from __future__ import annotations

from typing import Sequence

import numpy as np


class HFNextTokenLogits:
    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        device: str | None = None,
        dtype: str = "auto",
        batch_size: int = 8,
        revision: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, revision=revision)
        if adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.to(self.device).eval()

    def __call__(self, prompts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        rows = []
        with torch.inference_mode():
            for start in range(0, len(prompts), self.batch_size):
                batch = list(prompts[start : start + self.batch_size])
                enc = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.device)
                logits = self.model(**enc).logits[:, -1, :].float()
                rows.append(logits.cpu().numpy())
        return np.concatenate(rows, axis=0)
