"""
generator/code_verifier.py — Gemini API Code Verifier (Streamlit edition)
==========================================================================
Sends generated Python (app.py / test_app.py) to the Gemini API for a
logic + syntax check.  HTML/CSS verification is removed — Streamlit
generates its own UI at runtime.

Gemini returns either:
  - The corrected code (if issues were found)
  - "OK" (if no issues were found; original code is returned unchanged)

A fast AST pre-check runs before the API call to skip trivial issues
and conserve API quota.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from typing import Optional

log = logging.getLogger(__name__)

try:
    import google.generativeai as genai  # type: ignore
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    log.warning("google-generativeai not installed — CodeVerifier will skip API verification.")


VERIFICATION_SYSTEM = """\
You are an expert Python code reviewer specialising in Streamlit applications.
You will be given one generated Python file (app.py or test_app.py).

Your job:
1. Check for syntax errors.
2. Check for logic errors (missing imports, undefined variables, broken logic).
3. Ensure the code uses ONLY Streamlit for the UI — reject any Flask, FastAPI,
   render_template, jsonify, or @app.route usage.
4. Preserve the Streamlit framework and style already used.
5. If the code is correct, reply with exactly: OK
6. If the code has issues, reply with ONLY the corrected Python code and nothing else.
   No explanations, no markdown, just the fixed Python code.
"""


class CodeVerifier:
    """
    Verifies and auto-corrects generated Streamlit Python code via Gemini API.

    Args:
        cfg: Full config dict (reads [gemini] section).
    """

    def __init__(self, cfg: dict):
        gemini_cfg = cfg.get("gemini", {})
        self.api_key = os.getenv("GEMINI_API_KEY") or gemini_cfg.get("api_key", "")
        self.model_name = gemini_cfg.get("model", "gemini-1.5-pro-latest")
        self._model = None
        self._disabled = False

    # ── Public API ────────────────────────────────────────────────────────────
    def verify(self, python_code: str, artifact_name: str = "app.py") -> str:
        """
        Verify and optionally auto-correct the given Streamlit Python code.

        Args:
            python_code:   Content of the generated Python file.
            artifact_name: File label passed to Gemini for context.

        Returns:
            Corrected (or unchanged) Python code string.
        """
        # Fast local AST check first
        syntax_error = self._local_syntax_check(python_code)
        if syntax_error:
            log.warning("Local syntax error in %s: %s", artifact_name, syntax_error)

        # Reject code that still contains Flask patterns
        if self._contains_flask(python_code):
            log.warning(
                "%s contains Flask patterns — flagging for LLM refinement.", artifact_name
            )
            # Return as-is so the refinement loop sees the real code and feeds it back
            return python_code

        if self._disabled:
            return python_code

        if not GEMINI_AVAILABLE or not self.api_key or "YOUR" in self.api_key:
            log.info("Gemini verification skipped (API unavailable or unconfigured).")
            return python_code

        try:
            corrected = self._call_gemini(python_code, artifact_name)
            if corrected.strip().upper().startswith("OK"):
                log.info("✓ Gemini verified %s — no issues found.", artifact_name)
                return python_code
            log.info("✓ Gemini applied corrections to %s.", artifact_name)
            return corrected
        except Exception as exc:
            self._disabled = True
            log.warning("Gemini verification failed: %s. Disabling verifier for this run.", exc)
            return python_code

    # ── Internal ──────────────────────────────────────────────────────────────
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
            "Ensure it uses ONLY Streamlit for the UI — no Flask allowed.\n\n"
            f"```python\n{python_code}\n```"
        )
        response = self._model.generate_content(prompt)
        raw = response.text or ""
        # Strip markdown fences if Gemini added them
        raw = re.sub(r"```(?:python)?\s*", "", raw).strip().rstrip("```").strip()
        return raw
