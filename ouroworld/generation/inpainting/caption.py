"""Text prompt for the video model: a BLIP-2 caption of the reference video."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class Blip2Captioner:
    """Captions one frame with BLIP-2 and appends a fixed quality suffix."""

    def __init__(self, model_dir: Path, suffix: str) -> None:
        """Use the BLIP-2 checkpoint in ``model_dir``; ``suffix`` is appended verbatim."""
        self.model_dir = Path(model_dir)
        self.suffix = suffix

    def caption(self, image: np.ndarray) -> str:
        """Return the prompt for the ``(H, W, 3)`` uint8 ``image``."""
        from transformers import AutoProcessor, Blip2ForConditionalGeneration

        processor = AutoProcessor.from_pretrained(self.model_dir)
        model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_dir, torch_dtype=torch.float16
        ).to("cuda")
        inputs = processor(images=Image.fromarray(image), return_tensors="pt").to(
            "cuda", torch.float16
        )
        with torch.inference_mode():
            tokens = model.generate(**inputs)
        text = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        del processor, model, inputs, tokens
        gc.collect()
        torch.cuda.empty_cache()
        return text + self.suffix
