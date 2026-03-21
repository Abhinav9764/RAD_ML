"""
Code_Generator/RAD-ML/core/llm_client.py
=========================================
Thin wrapper around Google Gemini Flash (free tier, no cost).
Provides generate(prompt) → str  with retry + timeout.
"""
from __future__ import annotations
import logging
import time

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config: dict):
        llm_cfg        = config.get("llm", {})
        self._api_key  = llm_cfg.get("gemini_api_key", "")
        self._model    = llm_cfg.get("gemini_model", "gemini-1.5-flash")
        self._retries  = int(llm_cfg.get("max_retries", 3))
        self._timeout  = int(llm_cfg.get("timeout_seconds", 90))
        self._client   = None
        self._init()

    def _init(self) -> None:
        if not self._api_key or self._api_key.startswith("YOUR_"):
            logger.warning("Gemini API key not configured — LLM calls will fail.")
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(self._model)
            logger.info("LLMClient ready: %s", self._model)
        except Exception as exc:
            logger.error("LLM init failed: %s", exc)

    def generate(self, prompt: str) -> str:
        """Generate text. Raises RuntimeError on total failure."""
        if self._client is None:
            raise RuntimeError("LLM not initialised. Check gemini_api_key in config.yaml.")

        last_exc: Exception | None = None
        for attempt in range(1, self._retries + 1):
            try:
                response = self._client.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.2,
                        "max_output_tokens": 8192,
                    },
                )
                return response.text
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning("LLM attempt %d/%d failed: %s — retrying in %ds",
                               attempt, self._retries, exc, wait)
                time.sleep(wait)

        raise RuntimeError(f"LLM failed after {self._retries} attempts: {last_exc}")
