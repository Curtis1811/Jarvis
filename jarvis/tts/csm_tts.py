"""Local conversational voice with Sesame CSM-1B, via Hugging Face transformers.

The closest open model to the "talking to a person" feel of Sesame Maya or Dessa.
Needs an NVIDIA GPU (around 8 GB VRAM or more) to keep up with conversation.
The model is gated: accept the terms at https://huggingface.co/sesame/csm-1b and
run `huggingface-cli login` once before first use.
"""

from __future__ import annotations

import numpy as np


class SesameCSMTTS:
    RATE = 24_000

    def __init__(self, model_id: str = "sesame/csm-1b", device: str = "auto", speaker: str = "0") -> None:
        import torch
        from transformers import AutoProcessor, CsmForConditionalGeneration

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)
        self._speaker = speaker

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        conversation = [{"role": self._speaker, "content": [{"type": "text", "text": text}]}]
        inputs = self._processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True
        ).to(self._device)
        audio = self._model.generate(**inputs, output_audio=True)
        wave = audio[0].to("cpu").float().numpy()
        return wave.astype(np.float32), self.RATE
