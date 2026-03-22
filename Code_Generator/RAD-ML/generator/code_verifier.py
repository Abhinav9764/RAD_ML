"""
generator/code_verifier.py - Gemini API Code Verifier (Streamlit edition)
==========================================================================
Sends generated Python (app.py / test_app.py) to Gemini for a logic + syntax
check. HTML/CSS verification is removed because Streamlit renders the UI.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Optional

log = logging.getLogger(__name__)

try:
    import google.generativeai as genai  # type: ignore

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    log.warning("google-generativeai not installed; CodeVerifier will skip API verification.")


VERIFICATION_SYSTEM = """\
You are an expert Python code reviewer specialising in Streamlit applications.
You will be given one generated Python file (app.py or test_app.py).

Your job:
1. Check for syntax errors.
2. Check for logic errors (missing imports, undefined variables, broken logic).
3. Ensure the code uses ONLY Streamlit for the UI - reject any Flask, FastAPI,
   render_template, jsonify, or @app.route usage.
4. Preserve the Streamlit framework and style already used.
5. If the code is correct, reply with exactly: OK
6. If the code has issues, reply with ONLY the corrected Python code and nothing else.
   No explanations, no markdown, just the fixed Python code.
"""


class CodeVerifier:
    """
    Verifies and auto-corrects generated Streamlit Python code via Gemini API.
    """

    def __init__(self, cfg: dict):
        gemini_cfg = cfg.get("gemini", {})
        self.api_key = os.getenv("GEMINI_API_KEY") or gemini_cfg.get("api_key", "")
        self.model_name = gemini_cfg.get("model", "gemini-1.5-pro-latest")
        self._model = None
        self._disabled = False
        self._fail_fast = str(os.getenv("RADML_FAIL_FAST_LLM", "1")).strip().lower() not in {"0", "false", "no"}
        self._timeout = max(1, int(gemini_cfg.get("verifier_timeout_seconds", 10)))

    def verify(self, python_code: str, artifact_name: str = "app.py") -> str:
        """
        Verify and optionally auto-correct the given Streamlit Python code.
        """
        syntax_error = self._local_syntax_check(python_code)
        if syntax_error:
            log.warning("Local syntax error in %s: %s", artifact_name, syntax_error)

        if self._contains_flask(python_code):
            log.warning("%s contains Flask patterns; returning original code for downstream fallback.", artifact_name)
            return python_code

        if self._disabled:
            return python_code

        if not GEMINI_AVAILABLE or not self.api_key or "YOUR" in self.api_key:
            log.info("Gemini verification skipped (API unavailable or unconfigured).")
            return python_code

        if self._fail_fast and str(os.getenv("RADML_SKIP_REMOTE_VERIFY", "0")).strip().lower() in {"1", "true", "yes"}:
            log.info("Gemini verification skipped because RADML_SKIP_REMOTE_VERIFY is enabled.")
            return python_code

        executor = ThreadPoolExecutor(max_workers=1)
        future = None
        try:
            future = executor.submit(self._call_gemini, python_code, artifact_name)
            corrected = future.result(timeout=self._timeout if self._fail_fast else max(self._timeout, 30))
            if corrected.strip().upper().startswith("OK"):
                log.info("Gemini verified %s with no issues.", artifact_name)
                return python_code
            log.info("Gemini applied corrections to %s.", artifact_name)
            return corrected
        except FutureTimeoutError:
            self._disabled = True
            log.warning("Gemini verification timed out for %s after %ss. Skipping remote verification.", artifact_name, self._timeout)
            return python_code
        except Exception as exc:
            self._disabled = True
            log.warning("Gemini verification failed: %s. Disabling verifier for this run.", exc)
            return python_code
        finally:
            if future is not None and not future.done():
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _local_syntax_check(code: str) -> Optional[str]:
        try:
            ast.parse(code)
            return None
        except SyntaxError as exc:
            return str(exc)

    @staticmethod
    def _contains_flask(code: str) -> bool:
        src = str(code or "").lower()
        return any(
            pattern in src
            for pattern in (
                "from flask",
                "import flask",
                "flask(",
                "@app.route",
                "render_template",
                "jsonify(",
            )
        )

    def _call_gemini(self, python_code: str, artifact_name: str) -> str:
        if self._model is None:
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=VERIFICATION_SYSTEM,
            )

        prompt = (
            f"Review this generated Streamlit Python file ({artifact_name}). "
            "Ensure it uses ONLY Streamlit for the UI - no Flask allowed.\n\n"
            f"```python\n{python_code}\n```"
        )
        response = self._model.generate_content(prompt)
        raw = response.text or ""
        raw = re.sub(r"```(?:python)?\\s*", "", raw).strip().rstrip("```").strip()
        return raw
