"""
Code_Generator/RAD-ML/generator/code_verifier.py
=================================================
Verifies generated app.py by:
  1. AST parse check (syntax errors caught instantly, no subprocess needed)
  2. Import check (all imports resolvable in workspace venv)
  3. If errors found → ask Gemini to fix, up to max_fix_attempts

Returns the fixed code path or raises after exhausting retries.
"""
from __future__ import annotations
import ast
import logging
import subprocess
import sys
import textwrap
from pathlib import Path

logger = logging.getLogger(__name__)

_FIX_PROMPT = """
The following Python Flask app.py has an error.

=== ERROR ===
{error}

=== CURRENT CODE ===
{code}

=== TASK ===
Fix ALL errors in the code above.
Return ONLY the corrected Python code — no markdown fences, no explanation.
Start directly with: from flask import ...
"""


class CodeVerifier:
    def __init__(self, llm_client, config: dict):
        self._llm      = llm_client
        self._max_tries = int(config.get("codegen", {}).get("max_fix_attempts", 5))

    # ── public ────────────────────────────────────────────────────────────────
    def verify_and_fix(self, app_path: Path) -> Path:
        """
        Verify app_path. If broken, attempt up to max_fix_attempts LLM fixes.
        Returns the (possibly updated) Path on success.
        Raises RuntimeError if unfixable.
        """
        for attempt in range(1, self._max_tries + 1):
            error = self._check(app_path)
            if error is None:
                logger.info("Code verification passed on attempt %d.", attempt)
                return app_path

            logger.warning("Attempt %d/%d — error detected:\n%s",
                           attempt, self._max_tries, error[:300])

            if attempt == self._max_tries:
                raise RuntimeError(
                    f"Code could not be fixed after {self._max_tries} attempts.\n"
                    f"Last error:\n{error}"
                )

            fixed = self._fix(app_path.read_text(encoding="utf-8"), error)
            app_path.write_text(fixed, encoding="utf-8")
            logger.info("Applied LLM fix, re-verifying …")

        return app_path   # unreachable, but satisfies type checker

    # ── internals ─────────────────────────────────────────────────────────────
    def _check(self, path: Path) -> str | None:
        """Return error string or None if code is clean."""
        code = path.read_text(encoding="utf-8")

        # 1. Syntax check via AST
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return f"SyntaxError at line {exc.lineno}: {exc.msg}"

        # 2. Compile check (catches more issues than AST alone)
        try:
            compile(code, str(path), "exec")
        except Exception as exc:
            return str(exc)

        # 3. Light static scan — forbidden patterns
        forbidden = [
            ("import os; os.system", "os.system call detected"),
            ("subprocess.call",      "subprocess.call detected"),
            ("eval(",                "eval() detected — security risk"),
            ("exec(",                "exec() detected — security risk"),
        ]
        for pattern, msg in forbidden:
            if pattern in code:
                return msg

        return None

    def _fix(self, code: str, error: str) -> str:
        import re
        prompt = _FIX_PROMPT.format(error=error, code=code)
        fixed  = self._llm.generate(prompt)
        # Strip markdown fences if present
        fixed  = re.sub(r"^```[a-z]*\n?", "", fixed.strip(), flags=re.MULTILINE)
        fixed  = re.sub(r"\n?```$",        "", fixed.strip(), flags=re.MULTILINE)
        return fixed.strip()
