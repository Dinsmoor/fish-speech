import os

import torch
from loguru import logger

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import (
    launch_batched_queue,
    launch_thread_safe_queue,
)
from fish_speech.utils.schema import ServeTTSRequest
from tools.server.inference import inference_wrapper as inference

# Parallel sentence-chunk path (faster-than-realtime on GB10). Set
# FISH_BATCHED=0 to disable (saves ~9GB for a second model instance).
BATCHED_ENABLED = os.environ.get("FISH_BATCHED", "1") != "0"
BATCHED_SIZE = int(os.environ.get("FISH_BATCHED_SIZE", "4"))
BATCHED_CACHE_LEN = int(os.environ.get("FISH_BATCHED_CACHE_LEN", "2048"))


class ModelManager:
    def __init__(
        self,
        mode: str,
        device: str,
        half: bool,
        compile: bool,
        llama_checkpoint_path: str,
        decoder_checkpoint_path: str,
        decoder_config_name: str,
    ) -> None:

        self.mode = mode
        self.device = device
        self.half = half
        self.compile = compile

        self.precision = torch.half if half else torch.bfloat16

        # Check if MPS or CUDA is available
        if torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("mps is available, running on mps.")
        elif not torch.cuda.is_available():
            self.device = "cpu"
            logger.info("CUDA is not available, running on CPU.")

        # Load the TTS models
        self.load_llama_model(
            llama_checkpoint_path, self.device, self.precision, self.compile, self.mode
        )
        self.load_decoder_model(
            decoder_config_name, decoder_checkpoint_path, self.device
        )
        self.tts_inference_engine = TTSInferenceEngine(
            llama_queue=self.llama_queue,
            batched_queue=self.batched_queue,
            decoder_model=self.decoder_model,
            precision=self.precision,
            compile=self.compile,
        )

        # Warm up the models
        if self.mode == "tts":
            self.warm_up(self.tts_inference_engine)

    def load_llama_model(
        self, checkpoint_path, device, precision, compile, mode
    ) -> None:

        if mode == "tts":
            self.llama_queue = launch_thread_safe_queue(
                checkpoint_path=checkpoint_path,
                device=device,
                precision=precision,
                compile=compile,
            )
            self.batched_queue = None
            if BATCHED_ENABLED and device != "cpu":
                logger.info(
                    f"Launching batched (parallel sentence-chunk) worker "
                    f"(batch_size={BATCHED_SIZE}, cache_len={BATCHED_CACHE_LEN})..."
                )
                self.batched_queue = launch_batched_queue(
                    checkpoint_path=checkpoint_path,
                    device=device,
                    precision=precision,
                    compile=compile,
                    batch_size=BATCHED_SIZE,
                    cache_len=BATCHED_CACHE_LEN,
                )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        logger.info("LLAMA model loaded.")

    def load_decoder_model(self, config_name, checkpoint_path, device) -> None:
        self.decoder_model = load_decoder_model(
            config_name=config_name,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        logger.info("Decoder model loaded.")

    def warm_up(self, tts_inference_engine) -> None:
        request = ServeTTSRequest(
            text="Hello world.",
            references=[],
            reference_id=None,
            max_new_tokens=1024,
            chunk_length=200,
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7,
            format="wav",
        )
        list(inference(request, tts_inference_engine))
        # The batched worker self-warms (and compiles) at startup in
        # launch_batched_queue, so no separate batched warmup is needed here.
        logger.info("Models warmed up.")
