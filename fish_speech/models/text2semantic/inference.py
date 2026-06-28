import os
import queue
import re
import threading
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple, Union

import click
import numpy as np
import torch
import torch._inductor.config
from loguru import logger
from tqdm import tqdm

from fish_speech.content_sequence import (
    TextPart,
    VQPart,
)
from fish_speech.conversation import Conversation, Message
from fish_speech.tokenizer import IM_END_TOKEN

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    torch._inductor.config.fx_graph_cache = True


from torch.nn.attention import SDPBackend, sdpa_kernel

from fish_speech.models.text2semantic.llama import (
    BaseTransformer,
    DualARTransformer,
    NaiveTransformer,
)


def multinomial_sample_one_no_sync(probs_sort):
    q = torch.rand_like(probs_sort)
    q = -torch.log(q)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


RAS_WIN_SIZE = 10  # window for Repetition Aware Sampling
RAS_HIGH_TEMP = 1.0
RAS_HIGH_TOP_P = 0.9


def logits_to_probs(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,  # 注意: 我看到你传进来的是 int，这很关键
) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

    indices = torch.arange(sorted_logits.shape[-1], device=sorted_logits.device)
    top_k_mask = indices >= top_k
    sorted_indices_to_remove = (cum_probs > top_p) | top_k_mask
    sorted_indices_to_remove[0] = False  # 单元素修改问题不大，或者写成 | (indices != 0)

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = torch.where(
        indices_to_remove, float("-Inf"), logits
    )  # 同样替换 masked_fill_ 为 torch.where
    logits = logits / torch.clip(temperature, min=1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[0, -1],
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def decode_one_token_ar(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
    semantic_logit_bias: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    forward_result = model.forward_generate(
        x,
        input_pos,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
    )
    logits = forward_result.logits  # (1, 1, vocab_size)
    hidden_states = forward_result.hidden_states

    # Apply constrained decoding: only allow semantic tokens + im_end
    biased_logits = logits + semantic_logit_bias

    # Normal sample
    main_token_normal = sample(
        biased_logits, temperature=temperature, top_p=top_p, top_k=top_k
    )[0]

    # RAS: also sample with high temp to use as fallback if token repeats
    high_temp = torch.tensor(
        RAS_HIGH_TEMP, device=temperature.device, dtype=temperature.dtype
    )
    high_top_p = torch.tensor(RAS_HIGH_TOP_P, device=top_p.device, dtype=top_p.dtype)
    main_token_high = sample(
        biased_logits, temperature=high_temp, top_p=high_top_p, top_k=top_k
    )[0]

    # Use high-temp sample if: token is semantic AND token is in previous window
    if previous_tokens is not None:
        in_window = (previous_tokens[0] == main_token_normal).any()
        # Use tensor ops (&, torch.where) instead of Python (and, if) — torch.compile requires no data-dependent branching
        is_semantic = (main_token_normal >= model.config.semantic_begin_id) & (
            main_token_normal <= model.config.semantic_end_id
        )
        should_use_high = in_window & is_semantic
        main_token_normal = torch.where(
            should_use_high, main_token_high, main_token_normal
        )

    codebooks = [main_token_normal]

    input_pos = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
    model.forward_generate_fast(hidden_states, input_pos)

    a = codebooks[0] - model.config.semantic_begin_id
    a = torch.clamp(a, min=0, max=model.config.codebook_size - 1)

    hidden_states = model.fast_embeddings(a)
    codebooks.append(a)

    for codebook_idx in range(1, model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)

        short_logits = logits  # DualAR predicts config.codebook_size number of tokens

        # Convert logits to probs (no constrain for fast codebooks)
        a = sample(
            short_logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )[0]

        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)

    codebooks = torch.stack(codebooks, dim=1)

    # Only delete references, let Python GC handle cleanup
    del logits, hidden_states, forward_result

    return codebooks.T


def decode_n_tokens(
    model: DualARTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
    semantic_logit_bias: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    decode_one_token=decode_one_token_ar,
):
    # Rolling window for RAS (Repetition Aware Sampling)
    previous_tokens = torch.zeros(
        (model.config.num_codebooks + 1, RAS_WIN_SIZE),
        dtype=torch.int,
        device=cur_token.device,
    )
    # Accumulate all generated tokens (the actual output)
    new_tokens = []

    # [MODIFIED] Pre-fetch ID for efficiency loop
    im_end_id = model.tokenizer.get_token_id(IM_END_TOKEN)

    for i in tqdm(range(num_new_tokens)):
        with sdpa_kernel(SDPBackend.MATH):
            next_token = decode_one_token(
                model=model,
                x=cur_token,
                input_pos=input_pos,
                previous_tokens=previous_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                semantic_logit_bias=semantic_logit_bias,
                audio_masks=audio_masks,
                audio_parts=audio_parts,
            ).clone()

        input_pos += 1
        cur_token = next_token.view(1, model.config.num_codebooks + 1, -1)
        # Roll RAS window left and insert new token at end
        previous_tokens = previous_tokens.roll(-1, dims=1)
        previous_tokens[:, -1] = next_token.view(model.config.num_codebooks + 1, -1)[
            :, 0
        ]
        new_tokens.append(next_token)

        if cur_token[0, 0, -1] == im_end_id:
            break

    del cur_token

    return torch.cat(new_tokens, dim=1)


@torch.no_grad()
@torch.inference_mode()
def generate(
    *,
    model: DualARTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    decode_one_token=decode_one_token_ar,
    num_samples: int = 1,
    **sampling_kwargs,
):
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(1)
    prompt = prompt[None].repeat(num_samples, 1, 1)

    # Bound generation by the actual KV-cache width (model.max_seq_len, set by
    # setup_caches) rather than the model's config max_seq_len, which may be far
    # larger than the allocated cache.
    seq_limit = (
        model.max_seq_len
        if getattr(model, "max_seq_len", -1) and model.max_seq_len > 0
        else model.config.max_seq_len
    )

    if T >= seq_limit:
        raise ValueError(
            f"Input sequence length {T} exceeds cache max_seq_len {seq_limit}"
        )

    if max_new_tokens:
        if T + max_new_tokens > seq_limit:
            max_new_tokens = seq_limit - T

        T_new = T + max_new_tokens
    else:
        T_new = seq_limit
        max_new_tokens = T_new - T

    device = prompt.device
    dtype = next(
        model.parameters()
    ).dtype  # model weight dtype (bfloat16), NOT prompt dtype (int32)

    # Critical fix: Only set up cache on first run or when necessary
    if not hasattr(model, "_cache_setup_done") or not model._cache_setup_done:
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,  # Fixed to 1, avoid dynamic changes
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        model._cache_setup_done = True

    codebook_dim = 1 + model.config.num_codebooks

    # Create new tensor each time, but try to reuse memory
    input_pos = torch.arange(0, T, device=device, dtype=torch.long)
    empty = torch.empty(
        (codebook_dim, model.config.max_seq_len), dtype=prompt.dtype, device=device
    )
    empty[:, :T] = prompt
    seq = empty

    temp_val = sampling_kwargs.get("temperature", 1.0)
    top_p_val = sampling_kwargs.get("top_p", 0.9)
    top_k_val = sampling_kwargs.get("top_k", 30)

    temperature = torch.tensor(temp_val, device=device, dtype=dtype)
    top_p = torch.tensor(top_p_val, device=device, dtype=dtype)

    # Build semantic logit bias: 0 for semantic tokens + im_end, -inf for all others
    vocab_size = model.config.vocab_size
    semantic_logit_bias = torch.full(
        (1, 1, vocab_size), float("-inf"), device=device, dtype=dtype
    )

    # [MODIFIED] Use config for semantic range
    semantic_logit_bias[
        0, 0, model.config.semantic_begin_id : model.config.semantic_end_id + 1
    ] = 0.0

    # [MODIFIED] Use tokenizer.get_token_id (Wrapper method)
    semantic_logit_bias[0, 0, model.tokenizer.get_token_id(IM_END_TOKEN)] = 0.0

    prefill_decode = decode_one_token_ar

    first_token = prefill_decode(
        model,
        prompt.view(1, codebook_dim, -1),
        input_pos,
        temperature,
        top_p,
        top_k_val,
        semantic_logit_bias,
        audio_masks,
        audio_parts,
    )
    seq[:, T : T + 1] = first_token

    # Recreate input_pos
    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    x = decode_n_tokens(
        model,
        first_token.view(1, codebook_dim, -1),
        input_pos,
        max_new_tokens - 1,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k_val,
        semantic_logit_bias=semantic_logit_bias,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
        decode_one_token=decode_one_token,
    )
    seq = seq[:, : T + 1 + x.size(1)]
    seq[:, T + 1 :] = x

    # Clean up temporary variables
    del first_token, x, prompt, empty, input_pos

    return seq


# ===========================================================================
# Batched (parallel sentence-chunk) generation
#
# Splits one utterance into sentence chunks and generates them in a single
# forward batch (lockstep decode over the batch dim). On the GB10 this turns the
# ~2.0 single-stream RTF into ~0.65 (faster than realtime) because the GPU is
# heavily underutilized at batch 1. Critically it requires a SMALL KV cache: with
# an attention mask, each step attends over the full cache width, so a large
# max_seq_len dominates the cost and kills batch scaling.
# ===========================================================================

_SENTENCE_RE = re.compile(r"[^.!?。！？\n]+[.!?。！？]?", re.UNICODE)


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-ish chunks for parallel synthesis."""
    parts = [s.strip() for s in _SENTENCE_RE.findall(text)]
    return [p for p in parts if p]


def logits_to_probs_batched(logits, temperature, top_p, top_k: int):
    # logits: (B, V)
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    idx = torch.arange(sorted_logits.shape[-1], device=logits.device)
    remove = (cum > top_p) | (idx >= top_k)[None, :]
    remove[:, 0] = False
    remove = remove.scatter(-1, sorted_idx, remove)
    logits = torch.where(remove, float("-inf"), logits) / torch.clip(
        temperature, min=1e-5
    )
    return torch.softmax(logits, dim=-1)


def sample_batched(logits, temperature, top_p, top_k: int):
    probs = logits_to_probs_batched(logits, temperature, top_p, top_k)
    q = -torch.log(torch.rand_like(probs))
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(torch.int)  # (B,1)


def decode_one_token_ar_batched(
    model,
    x,
    input_pos,
    temperature,
    top_p,
    top_k: int,
    valid_mask,
    sb: int,
    se: int,
    im_end: int,
    nc: int,
):
    """One lockstep batched decode step. x: (B, codebook_dim, S) -> (B, codebook_dim).

    The main token can only be a semantic token or im_end, so we sample over just
    that candidate set (~4097) instead of the full 155776 vocab — equivalent to the
    -inf bias but far cheaper and avoids an inductor cumsum codegen bug. sb/se/im_end/nc
    are passed as ints (constants) so this compiles fullgraph.
    """
    res = model.forward_generate(
        x, input_pos, key_padding_mask=valid_mask.logical_not()
    )
    logits = res.logits[:, -1]  # (B, V)
    hidden = res.hidden_states  # (B, 1, fast_dim)
    cand = torch.cat([logits[:, sb : se + 1], logits[:, im_end : im_end + 1]], dim=1)
    n_sem = se + 1 - sb
    idx = sample_batched(cand, temperature, top_p, top_k)  # (B,1) index into cand
    main = torch.where(idx >= n_sem, torch.full_like(idx, im_end), idx + sb)

    codebooks = [main]
    model.forward_generate_fast(hidden, torch.tensor([0], device=x.device))
    a = torch.clamp(main - sb, 0, model.config.codebook_size - 1)
    h = model.fast_embeddings(a)
    codebooks.append(a)
    for cb in range(1, nc):
        flogits = model.forward_generate_fast(h, torch.tensor([cb], device=x.device))
        fa = sample_batched(flogits[:, -1], temperature, top_p, top_k)
        h = model.fast_embeddings(fa)
        codebooks.append(fa)
    return torch.cat(codebooks, dim=1)  # (B, codebook_dim)


@torch.no_grad()
def generate_batched(
    *,
    model,
    prompts,
    decode_step,
    im_end,
    sb,
    se,
    nc,
    temperature,
    top_p,
    top_k,
    max_new_tokens,
):
    """prompts: list of (codebook_dim, T_i) tensors (left-padded to a common length
    internally). Returns list of (nc, L_i) code tensors, one per prompt, trimmed at
    each row's im_end."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    B = len(prompts)
    cb_dim = nc + 1
    T_max = max(p.size(1) for p in prompts)
    max_seq = model.max_seq_len

    if T_max >= max_seq:
        raise ValueError(
            f"Batched prompt length {T_max} exceeds cache width {max_seq}; "
            f"use a longer cache or shorter reference/chunks."
        )

    batch = torch.zeros((B, cb_dim, T_max), dtype=torch.int, device=device)
    valid = torch.zeros((B, max_seq), dtype=torch.bool, device=device)
    for i, p in enumerate(prompts):
        Ti = p.size(1)
        pad = T_max - Ti
        batch[i, :, pad:] = p.to(device)
        valid[i, pad:T_max] = True

    temp_t = torch.tensor(temperature, device=device, dtype=dtype)
    topp_t = torch.tensor(top_p, device=device, dtype=dtype)

    # prefill (eager, variable length)
    input_pos = torch.arange(0, T_max, device=device)
    first = decode_one_token_ar_batched(
        model, batch, input_pos, temp_t, topp_t, top_k, valid, sb, se, im_end, nc
    )

    max_steps = min(max_new_tokens, max_seq - T_max)
    all_tokens = torch.zeros((B, cb_dim, max_steps), dtype=torch.int, device=device)
    all_tokens[:, :, 0] = first
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    end_step = torch.full((B,), max_steps, dtype=torch.long, device=device)

    cur = first.unsqueeze(-1)
    pos = T_max
    pos_t = torch.tensor([pos], device=device)
    nsteps = 1
    for step in range(1, max_steps):
        valid[:, pos] = True
        pos_t.fill_(pos)
        nxt = decode_step(
            model, cur, pos_t, temp_t, topp_t, top_k, valid, sb, se, im_end, nc
        )
        all_tokens[:, :, step] = nxt
        newly = (nxt[:, 0] == im_end) & (~finished)
        end_step = torch.where(newly, torch.tensor(step, device=device), end_step)
        finished = finished | newly
        cur = nxt.unsqueeze(-1)
        pos += 1
        nsteps = step + 1
        if bool(finished.all()) or pos >= max_seq - 1:
            break

    end_step = end_step.clamp(max=nsteps).tolist()
    results = []
    for i in range(B):
        codes = all_tokens[i, 1:, : end_step[i]].clone()  # (nc, L)
        results.append(torch.clamp(codes, min=0))
    return results


def _build_batched_system_parts(prompt_text, prompt_tokens):
    """Mirror generate_long's system prompt (reference text + VQ codes, or generic)."""
    use_prompt = bool(prompt_text) and bool(prompt_tokens)
    if use_prompt:
        tagged = []
        for i, t in enumerate(prompt_text):
            tagged.append(
                t if re.search(r"<\|speaker:\d+\|>", t) else f"<|speaker:{i}|>{t}"
            )
        return [
            TextPart(
                text="convert the provided text to speech reference to the following:\n\nText:\n",
                cal_loss=False,
            ),
            TextPart(text="\n".join(tagged), cal_loss=False),
            TextPart(text="\n\nSpeech:\n", cal_loss=False),
            VQPart(codes=torch.cat([c for c in prompt_tokens], dim=1), cal_loss=False),
        ]
    return [TextPart(text="convert the provided text to speech", cal_loss=False)]


def generate_long_batched(
    *,
    model,
    decode_step,
    batch_size,
    device,
    text,
    max_new_tokens=0,
    top_p=0.8,
    top_k=30,
    temperature=0.8,
    prompt_text=None,
    prompt_tokens=None,
    **_ignored,
):
    """Sentence-chunk an utterance and synthesize chunks in parallel batches.

    Yields GenerateResponse(action="sample", codes=...) per chunk in order, then a
    final GenerateResponse(action="next") — same protocol as generate_long."""
    if isinstance(prompt_text, str):
        prompt_text = [prompt_text]
    if prompt_tokens is not None and not isinstance(prompt_tokens, list):
        prompt_tokens = [prompt_tokens]
    if prompt_tokens:
        prompt_tokens = [t.cpu() for t in prompt_tokens]

    tokenizer = model.tokenizer
    nc = model.config.num_codebooks
    sb = model.config.semantic_begin_id
    se = model.config.semantic_end_id
    im_end = tokenizer.get_token_id(IM_END_TOKEN)
    system_parts = _build_batched_system_parts(prompt_text, prompt_tokens)

    def build_prompt(chunk_text):
        conv = Conversation()
        conv.append(
            Message(
                role="system",
                parts=list(system_parts),
                cal_loss=False,
                add_im_start=True,
                add_im_end=True,
            )
        )
        conv.append(
            Message(
                role="user",
                parts=[TextPart(text=chunk_text, cal_loss=False)],
                cal_loss=False,
                add_im_start=True,
                add_im_end=True,
            )
        )
        conv.append(
            Message(
                role="assistant",
                parts=[],
                cal_loss=False,
                modality="voice",
                add_im_start=True,
                add_im_end=False,
            )
        )
        encoded, _, _ = conv.encode_for_inference(tokenizer, num_codebooks=nc)
        return encoded.to(device)

    sentences = split_sentences(text) or [text]
    logger.info(f"[batched] {len(sentences)} sentence chunks, batch_size={batch_size}")

    for g in range(0, len(sentences), batch_size):
        group = sentences[g : g + batch_size]
        n_real = len(group)
        # pad to a fixed batch size so torch.compile sees one shape
        padded = group + ["."] * (batch_size - n_real)
        prompts = [build_prompt(s) for s in padded]
        results = generate_batched(
            model=model,
            prompts=prompts,
            decode_step=decode_step,
            im_end=im_end,
            sb=sb,
            se=se,
            nc=nc,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens or 1024,
        )
        for i in range(n_real):
            yield GenerateResponse(action="sample", codes=results[i], text=group[i])

    yield GenerateResponse(action="next")


def init_model(checkpoint_path, device, precision, compile=False):
    model = DualARTransformer.from_pretrained(checkpoint_path, load_weights=True)

    model = model.to(device=device, dtype=precision)
    logger.info(f"Restored model from checkpoint")

    if isinstance(model, DualARTransformer):
        decode_one_token = decode_one_token_ar
        # prefill_n_tokens = decode_one_token_ar
        logger.info("Using DualARTransformer")
    else:
        raise ValueError("Unsupported model type")

    # Pre-create fixed parameter tensors to avoid runtime creation
    model.fixed_temperature = torch.tensor(0.7, device=device, dtype=torch.float)
    model.fixed_top_p = torch.tensor(0.7, device=device, dtype=torch.float)
    model.fixed_repetition_penalty = torch.tensor(1.5, device=device, dtype=torch.float)

    # Mark whether cache has been initialized
    model._cache_setup_done = False

    if compile:
        logger.info("Compiling function...")
        decode_one_token = torch.compile(
            decode_one_token,
            backend="inductor" if torch.cuda.is_available() else "aot_eager",
            mode="default" if torch.cuda.is_available() else None,
            fullgraph=True,
        )

    return model.eval(), decode_one_token


@torch.inference_mode()
def load_codec_model(codec_checkpoint_path, device, precision=torch.bfloat16):
    """Load the DAC codec model for audio encoding/decoding."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config_path = Path(__file__).parent.parent.parent / "configs" / "modded_dac_vq.yaml"
    cfg = OmegaConf.load(str(config_path))
    codec = instantiate(cfg)

    state_dict = torch.load(codec_checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }
    codec.load_state_dict(state_dict, strict=False)
    codec.eval()
    codec.to(device=device, dtype=precision)
    return codec


@torch.inference_mode()
def encode_audio(audio_path, codec, device):
    """Encode an audio file to VQ codes."""
    import torchaudio

    wav, sr = torchaudio.load(str(audio_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = torchaudio.functional.resample(wav.to(device), sr, codec.sample_rate)[0]

    # Match codec model dtype (e.g. bfloat16)
    model_dtype = next(codec.parameters()).dtype
    audios = wav[None, None].to(dtype=model_dtype)  # (1, 1, T)
    audio_lengths = torch.tensor([len(wav)], device=device, dtype=torch.long)

    indices, feature_lengths = codec.encode(audios, audio_lengths)
    return indices[0, :, : feature_lengths[0]]  # (num_codebooks, T)


@torch.inference_mode()
def decode_to_audio(codes, codec):
    """Decode VQ codes to audio waveform."""
    # codes: (num_codebooks, T) -> (1, num_codebooks, T)
    audio = codec.from_indices(codes[None])
    return audio[0, 0]  # (T,) mono waveform


@dataclass
class GenerateResponse:
    action: Literal["sample", "next"]
    codes: Optional[torch.Tensor] = None
    text: Optional[str] = None


def split_text_by_speaker(text: str) -> list[str]:
    """
    Split text into turns based on <|speaker:X|> tags.

    Args:
        text: The full text with speaker tags

    Returns:
        List of speaker turns, each starting with <|speaker:X|>
    """
    pattern = r"(<\|speaker:\d+\|>)"
    parts = re.split(pattern, text)

    turns = []
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if re.match(pattern, part):
            if i + 1 < len(parts):
                turn = part + parts[i + 1]
                turns.append(turn.strip())
                i += 2
            else:
                turns.append(part)
                i += 1
        else:
            i += 1

    return turns


def group_turns_into_batches(
    turns: list[str], max_speakers: int = 3, max_bytes: int = 300
) -> list[str]:
    """
    Group turns into batches based on speaker count or byte limit.

    Args:
        turns: List of speaker turns
        max_speakers: Maximum number of speakers per batch (default 3)
        max_bytes: Maximum UTF-8 bytes per batch (default 300)

    Returns:
        List of batched text strings
    """
    batches = []
    current_batch = []
    current_bytes = 0

    for turn in turns:
        turn_bytes = len(turn.encode("utf-8"))

        would_exceed_speakers = len(current_batch) >= max_speakers
        would_exceed_bytes = current_bytes + turn_bytes > max_bytes and current_batch

        if would_exceed_speakers or would_exceed_bytes:
            batches.append("\n".join(current_batch))
            current_batch = [turn]
            current_bytes = turn_bytes
        else:
            current_batch.append(turn)
            current_bytes += turn_bytes

    if current_batch:
        batches.append("\n".join(current_batch))

    return batches


def generate_long(
    *,
    model,
    device: Union[str, torch.device],
    decode_one_token: Callable,
    text: str,
    num_samples: int = 1,
    max_new_tokens: int = 0,
    top_p: float = 0.9,
    top_k: int = 30,
    repetition_penalty: float = 1.1,
    temperature: float = 1.0,
    compile: bool = False,
    iterative_prompt: bool = True,
    chunk_length: int = 512,
    prompt_text: Optional[Union[str, list[str]]] = None,
    prompt_tokens: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
):
    assert 0 < top_p <= 1, "top_p must be in (0, 1]"
    assert 0 < temperature < 2, "temperature must be in (0, 2)"

    use_prompt = bool(prompt_text) and bool(prompt_tokens)
    if use_prompt and isinstance(prompt_text, str):
        prompt_text = [prompt_text]
        prompt_tokens = [prompt_tokens]

    if use_prompt:
        assert len(prompt_text) == len(
            prompt_tokens
        ), "Prompt text and tokens must have the same length"

    if prompt_tokens:
        prompt_tokens = [i.cpu() for i in prompt_tokens]

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokenizer = model.tokenizer
    max_length = model.config.max_seq_len

    # Build base conversation with system message
    base_conversation = Conversation()

    if use_prompt:
        # Auto-add speaker tags to prompt texts that don't have them
        tagged_prompt_text = []
        for i, t in enumerate(prompt_text):
            if not re.search(r"<\|speaker:\d+\|>", t):
                tagged_prompt_text.append(f"<|speaker:{i}|>{t}")
            else:
                tagged_prompt_text.append(t)

        system_parts = [
            TextPart(
                text="convert the provided text to speech reference to the following:\n\nText:\n",
                cal_loss=False,
            ),
        ]
        reference_text = "\n".join(tagged_prompt_text)
        system_parts.append(TextPart(text=reference_text, cal_loss=False))
        system_parts.append(TextPart(text="\n\nSpeech:\n", cal_loss=False))
        all_codes = torch.cat([c for c in prompt_tokens], dim=1)
        system_parts.append(VQPart(codes=all_codes, cal_loss=False))
        # torch.save(all_codes, "debug_vq_codes.pt")
    else:
        system_parts = [
            TextPart(text="convert the provided text to speech", cal_loss=False)
        ]

    base_conversation.append(
        Message(
            role="system",
            parts=system_parts,
            cal_loss=False,
            add_im_start=True,
            add_im_end=True,
        )
    )

    # Split text by speaker and group into batches
    turns = split_text_by_speaker(text)
    if turns:
        batches = group_turns_into_batches(
            turns, max_speakers=5, max_bytes=chunk_length
        )
    else:
        batches = [text]

    logger.info(f"Split into {len(turns)} turns, grouped into {len(batches)} batches")

    for sample_idx in range(num_samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        # Deep copy base conversation for this sample
        conversation = deepcopy(base_conversation)

        for batch_idx, batch_text in enumerate(batches):
            logger.info(
                f"--- Sample {sample_idx}, Batch {batch_idx} "
                f"({len(batch_text.encode('utf-8'))} bytes) ---"
            )
            logger.info(f"Batch text: {batch_text}")

            # Add user message
            conversation.append(
                Message(
                    role="user",
                    parts=[TextPart(text=batch_text, cal_loss=False)],
                    cal_loss=False,
                    add_im_start=True,
                    add_im_end=True,
                )
            )

            # Deep copy for generation (don't pollute original conversation)
            conversation_gen = deepcopy(conversation)
            conversation_gen.append(
                Message(
                    role="assistant",
                    parts=[],
                    cal_loss=False,
                    modality="voice",
                    add_im_start=True,
                    add_im_end=False,
                )
            )

            logger.info("Visualizing prompt structure:")
            conversation_gen.visualize(
                tokenizer,
                merge_audio_tokens=True,
                merge_semantic_tokens=True,
            )

            encoded, audio_masks, audio_parts = conversation_gen.encode_for_inference(
                tokenizer, num_codebooks=model.config.num_codebooks
            )

            logger.info(f"Encoded prompt shape: {encoded.shape}")
            if audio_parts is not None:
                logger.info(f"Audio parts shape: {audio_parts.shape}")
            if audio_masks is not None:
                logger.info(
                    f"Audio masks non-zero count: {torch.count_nonzero(audio_masks)}"
                )

            if encoded.size(1) > max_length - 2048:
                raise ValueError(
                    f"Prompt is too long: {encoded.size(1)} > {max_length - 2048}"
                )

            encoded = encoded.to(device=device)
            prompt_length = encoded.size(1)

            y = generate(
                model=model,
                prompt=encoded,
                max_new_tokens=max_new_tokens,
                audio_masks=audio_masks,
                audio_parts=audio_parts,
                decode_one_token=decode_one_token,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

            if sample_idx == 0 and batch_idx == 0 and compile:
                logger.info(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_batch = time.perf_counter() - t0
            tokens_generated = y.size(1) - prompt_length
            tokens_sec = tokens_generated / t_batch if t_batch > 0 else 0
            logger.info(
                f"Batch {batch_idx}: Generated {tokens_generated} tokens in "
                f"{t_batch:.02f} seconds, {tokens_sec:.02f} tokens/sec"
            )
            logger.info(
                f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s"
            )

            # Extract generated codes
            codes = y[1:, prompt_length:-1].clone()
            assert (codes >= 0).all(), f"Negative code found: {codes}"

            # Add assistant message with generated codes back to conversation
            conversation.append(
                Message(
                    role="assistant",
                    parts=[VQPart(codes=codes.cpu(), cal_loss=False)],
                    cal_loss=False,
                    modality="voice",
                    add_im_start=True,
                    add_im_end=True,
                )
            )

            yield GenerateResponse(action="sample", codes=codes, text=batch_text)

            # Cleanup
            del y, encoded

        if torch.cuda.is_available():
            logger.info(
                f"GPU Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB"
            )

        yield GenerateResponse(action="next")


@dataclass
class WrappedGenerateResponse:
    status: Literal["success", "error"]
    response: Optional[Union[GenerateResponse, Exception]] = None


@dataclass
class GenerateRequest:
    request: dict
    response_queue: queue.Queue


def launch_thread_safe_queue(
    checkpoint_path,
    device,
    precision,
    compile: bool = False,
):
    input_queue = queue.Queue()
    init_event = threading.Event()

    def worker():
        model, decode_one_token = init_model(
            checkpoint_path, device, precision, compile=compile
        )
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        init_event.set()

        while True:
            item: GenerateRequest | None = input_queue.get()
            if item is None:
                break

            kwargs = item.request
            response_queue = item.response_queue

            try:
                for chunk in generate_long(
                    model=model, decode_one_token=decode_one_token, **kwargs
                ):
                    response_queue.put(
                        WrappedGenerateResponse(status="success", response=chunk)
                    )

                # Only clear cache after complete request batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(traceback.format_exc())
                response_queue.put(WrappedGenerateResponse(status="error", response=e))
                # Clear cache on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()

    return input_queue


def launch_batched_queue(
    checkpoint_path,
    device,
    precision,
    compile: bool = False,
    batch_size: int = 4,
    cache_len: int = 2048,
):
    """Worker for the parallel sentence-chunk (batched) path.

    Uses its own model instance with a SMALL KV cache (cache_len), which is what makes
    batched attention cheap and the whole utterance faster-than-realtime. Separate from
    the sequential worker so the existing path (long text / streaming, which accumulates
    conversation and needs the large cache) is unaffected.
    """
    input_queue = queue.Queue()
    init_event = threading.Event()

    def worker():
        model, _ = init_model(checkpoint_path, device, precision, compile=False)
        with torch.device(device):
            model.setup_caches(
                max_batch_size=batch_size,
                max_seq_len=cache_len,
                dtype=next(model.parameters()).dtype,
            )
        if compile:
            logger.info("Compiling batched decode step...")
            decode_step = torch.compile(
                decode_one_token_ar_batched,
                backend="inductor",
                mode="default",
                fullgraph=True,
            )
        else:
            decode_step = decode_one_token_ar_batched
        init_event.set()

        while True:
            item: GenerateRequest | None = input_queue.get()
            if item is None:
                break

            kwargs = item.request
            response_queue = item.response_queue

            try:
                for chunk in generate_long_batched(
                    model=model,
                    decode_step=decode_step,
                    batch_size=batch_size,
                    **kwargs,
                ):
                    response_queue.put(
                        WrappedGenerateResponse(status="success", response=chunk)
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error(traceback.format_exc())
                response_queue.put(WrappedGenerateResponse(status="error", response=e))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()

    return input_queue


@click.command()
@click.option(
    "--text",
    type=str,
    default="<|speaker:0|>你说的对, 但是原神是一款由米哈游自主研发的开放世界手游.",
)
@click.option("--prompt-text", type=str, default=None, multiple=True)
@click.option(
    "--prompt-tokens",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    multiple=True,
)
@click.option(
    "--prompt-audio",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    multiple=True,
)
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option("--num-samples", type=int, default=1)
@click.option("--max-new-tokens", type=int, default=0)
@click.option("--top-p", type=float, default=0.9)
@click.option("--top-k", type=int, default=30)
@click.option("--temperature", type=float, default=1.0)
@click.option(
    "--checkpoint-path",
    type=click.Path(path_type=Path, exists=True),
    default="checkpoints/s2-pro",
)
@click.option("--device", type=str, default="cuda")
@click.option("--compile/--no-compile", default=False)
@click.option("--seed", type=int, default=42)
@click.option("--half/--no-half", default=False)
@click.option("--iterative-prompt/--no-iterative-prompt", default=True)
@click.option("--chunk-length", type=int, default=300)
@click.option("--output-dir", type=Path, default="output")
def main(
    text: str,
    prompt_text: Optional[tuple[str, ...]],
    prompt_tokens: Optional[tuple[Path, ...]],
    prompt_audio: Optional[tuple[Path, ...]],
    output: Optional[Path],
    num_samples: int,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    temperature: float,
    checkpoint_path: Path,
    device: str,
    compile: bool,
    seed: int,
    half: bool,
    iterative_prompt: bool,
    chunk_length: int,
    output_dir: Path,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    precision = torch.half if half else torch.bfloat16

    if prompt_text and not prompt_audio and not prompt_tokens:
        raise ValueError(
            "--prompt-text requires either --prompt-audio or --prompt-tokens"
        )
    if prompt_text and prompt_tokens and len(prompt_text) != len(prompt_tokens):
        raise ValueError(
            f"Number of prompt text ({len(prompt_text)}) and prompt tokens ({len(prompt_tokens)}) should be the same"
        )
    if prompt_text and prompt_audio and len(prompt_text) != len(prompt_audio):
        raise ValueError(
            f"Number of prompt text ({len(prompt_text)}) and prompt audio ({len(prompt_audio)}) should be the same"
        )

    logger.info("Loading model ...")
    t0 = time.time()
    model, decode_one_token = init_model(
        checkpoint_path, device, precision, compile=compile
    )
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

    codec = None
    codec_checkpoint = checkpoint_path / "codec.pth"

    # Handle prompt: --prompt-audio takes priority over --prompt-tokens
    prompt_tokens_list = None
    if prompt_audio:
        logger.info("Loading codec model for audio encoding...")
        codec = load_codec_model(codec_checkpoint, device, precision)
        prompt_tokens_list = [
            encode_audio(p, codec, device).cpu() for p in prompt_audio
        ]
        logger.info(f"Encoded {len(prompt_audio)} audio file(s) to VQ codes")
    elif prompt_tokens is not None:
        prompt_tokens_list = [torch.from_numpy(np.load(p)) for p in prompt_tokens]

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    generator = generate_long(
        model=model,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        compile=compile,
        iterative_prompt=iterative_prompt,
        chunk_length=chunk_length,
        prompt_text=list(prompt_text) if prompt_text else None,
        prompt_tokens=prompt_tokens_list,
    )

    idx = 0
    codes = []

    for response in generator:
        if response.action == "sample":
            codes.append(response.codes)
            logger.info(f"Sampled text: {response.text}")
        elif response.action == "next":
            if codes:
                merged_codes = torch.cat(codes, dim=1)
                codes_npy_path = os.path.join(output_dir, f"codes_{idx}.npy")
                np.save(codes_npy_path, merged_codes.cpu().numpy())
                logger.info(f"Saved codes to {codes_npy_path}")

                # Decode to wav if --output is specified
                if output:
                    if codec is None:
                        logger.info("Loading codec model for audio decoding...")
                        codec = load_codec_model(codec_checkpoint, device, precision)
                    audio = decode_to_audio(merged_codes.to(device), codec)
                    import soundfile as sf

                    out_path = (
                        str(output)
                        if num_samples == 1
                        else str(output.with_stem(f"{output.stem}_{idx}"))
                    )
                    sf.write(out_path, audio.cpu().float().numpy(), codec.sample_rate)
                    logger.info(f"Saved audio to {out_path}")

            logger.info(f"Next sample")
            codes = []
            idx += 1
        else:
            logger.error(f"Error: {response}")


if __name__ == "__main__":
    main()
