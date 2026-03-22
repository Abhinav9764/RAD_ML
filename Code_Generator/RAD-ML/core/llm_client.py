"""
Code_Generator/RAD-ML/core/llm_client.py
========================================
Thin wrapper around Google Gemini Flash.
Provides generate(prompt) -> str with retry + timeout.
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config: dict):
        llm_cfg = config.get("llm", {}) or {}
        gemini_cfg = config.get("gemini", {}) or {}
        merged_cfg = {**gemini_cfg, **llm_cfg}
        self._api_key = os.getenv("GEMINI_API_KEY") or merged_cfg.get("gemini_api_key") or merged_cfg.get("api_key", "")
        self._model = merged_cfg.get("gemini_model") or merged_cfg.get("model", "gemini-2.0-flash")
        self._retries = max(1, int(merged_cfg.get("max_retries", 2)))
        self._timeout = max(1, int(merged_cfg.get("timeout_seconds", 20)))
        self._fail_fast = str(os.getenv("RADML_FAIL_FAST_LLM", "1")).strip().lower() not in {"0", "false", "no"}
        self._fast_timeout = max(1, min(self._timeout, int(merged_cfg.get("fast_timeout_seconds", 12))))
        self._fast_retries = max(1, min(self._retries, int(merged_cfg.get("fast_retries", 1))))
        self._client = None
        self._init()

    def _init(self) -> None:
        if not self._api_key or self._api_key.startswith("YOUR_"):
            logger.warning("Gemini API key not configured; LLM calls will fail.")
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

        timeout = self._fast_timeout if self._fail_fast else self._timeout
        retries = self._fast_retries if self._fail_fast else self._retries
        last_exc: Exception | None = None

        for attempt in range(1, retries + 1):
            executor = ThreadPoolExecutor(max_workers=1)
            future = None
            try:
                future = executor.submit(
                    self._client.generate_content,
                    prompt,
                    generation_config={
                        "temperature": 0.2,
                        "max_output_tokens": 4096,
                    },
                )
                response = future.result(timeout=timeout)
                text = getattr(response, "text", "")
                if not text:
                    raise RuntimeError("LLM returned an empty response")
                return text
            except FutureTimeoutError as exc:
                last_exc = exc
                logger.warning("LLM attempt %d/%d timed out after %ss", attempt, retries, timeout)
            except Exception as exc:
                last_exc = exc
                if attempt >= retries:
                    logger.warning("LLM attempt %d/%d failed: %s", attempt, retries, exc)
                else:
                    wait = 2 ** attempt
                    logger.warning("LLM attempt %d/%d failed: %s; retrying in %ds", attempt, retries, exc, wait)
                    time.sleep(wait)
            finally:
                if future is not None and not future.done():
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)

        raise RuntimeError(f"LLM failed after {retries} attempts: {last_exc}")
