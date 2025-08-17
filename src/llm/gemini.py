from __future__ import annotations
import google.generativeai as genai
import os
import time
from typing import List
from src.utils.config import AppConfig

class GeminiClient:
    def __init__(self, config: AppConfig):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)
        self.config = config
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        last_err = None
        for attempt in range(max_retries):
            try:
                rsp = self.model.generate_content(prompt, generation_config={"temperature": self.config.temperature, "max_output_tokens": self.config.max_tokens})
                return rsp.text
            except Exception as e:  # pragma: no cover - external API
                last_err = e
                time.sleep(1 + attempt)
        raise RuntimeError(f"Gemini generation failed: {last_err}")
