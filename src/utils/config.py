from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass(frozen=True)
class AppConfig:
    use_gemini: bool = True
    embed_model: str = "intfloat/e5-small-v2"
    local_llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    max_tokens: int = 2048
    temperature: float = 0.3
    confidence_threshold: int = 65
    workspace_dir: str = "workspace_tmp"
    use_small_local: bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv()
        return cls(
            use_gemini=os.getenv("USE_GEMINI", "true").lower() == "true",
            embed_model=os.getenv("EMBED_MODEL", "intfloat/e5-small-v2"),
            local_llm_model=os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            max_tokens=int(os.getenv("MAX_TOKENS", "2048")),
            temperature=float(os.getenv("TEMPERATURE", "0.3")),
            confidence_threshold=int(os.getenv("CONFIDENCE_THRESHOLD", "65")),
            workspace_dir=os.getenv("WORKSPACE_DIR", "workspace_tmp"),
            use_small_local=os.getenv("LOCAL_LLM_SMALL", "false").lower() == "true",
        )
