from __future__ import annotations
from functools import lru_cache
from src.utils.config import AppConfig
import os

LIGHTWEIGHT_DEFAULT = "distilgpt2"  # small CPU friendly model

try:  # defer heavy imports, handle missing deps
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
    import torch  # type: ignore
    _TRANS_AVAILABLE = True
except Exception:  # pragma: no cover
    _TRANS_AVAILABLE = False


@lru_cache(maxsize=1)
def _get_pipe(model_name: str, temperature: float):  # pragma: no cover - heavy
    if not _TRANS_AVAILABLE:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, "float16", None) if torch.cuda.is_available() else getattr(torch, "float32", None),
            device_map="auto" if torch.cuda.is_available() else None,
        )
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception:
        if model_name != LIGHTWEIGHT_DEFAULT:
            # retry with lightweight model
            tokenizer = AutoTokenizer.from_pretrained(LIGHTWEIGHT_DEFAULT)
            model = AutoModelForCausalLM.from_pretrained(LIGHTWEIGHT_DEFAULT)
            return pipeline("text-generation", model=model, tokenizer=tokenizer)
        raise


class LocalLLM:
    """Graceful local model wrapper; falls back to template stub if transformers unavailable."""

    def __init__(self, config: AppConfig):
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        preferred = config.local_llm_model or LIGHTWEIGHT_DEFAULT
        if os.getenv("LOCAL_LLM_SMALL", "false").lower() == "true":
            preferred = LIGHTWEIGHT_DEFAULT
        self.pipe = None
        if _TRANS_AVAILABLE:
            try:
                self.pipe = _get_pipe(preferred, self.temperature)
            except Exception:
                self.pipe = None

    def generate(self, prompt: str) -> str:
        if not self.pipe:
            # minimal heuristic summary / answer fallback
            tail = prompt.splitlines()[-8:]
            return "Fallback (no local model). Context signals: " + " ".join(t[:60] for t in tail)[:400]
        try:
            out = self.pipe(
                prompt,
                max_new_tokens=min(self.max_tokens, 256),
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                num_return_sequences=1,
                pad_token_id=getattr(self.pipe.tokenizer, "eos_token_id", None),
            )
            text = out[0]["generated_text"]
            return text[len(prompt):].strip() if text.startswith(prompt) else text
        except Exception:  # pragma: no cover
            return "Local generation error; please provide a Gemini API key for higher quality responses."
