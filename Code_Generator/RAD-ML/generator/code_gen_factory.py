"""
generator/code_gen_factory.py
==============================
Layer 3 — Code Generator

Generates each file in the project one at a time, using:
  - The full ProjectSpec from Layer 1
  - The Architecture Plan from Layer 2

Files generated (always):
  app.py          - Flask web app with form and prediction route
  predictor.py    - SageMaker endpoint caller (clean separation of concerns)
  train.py        - Training utilities and metric computation
  requirements.txt
  README.md
  tests/test_app.py

Each file is generated with a focused, file-specific prompt so the LLM
produces complete, production-grade code — not placeholder pseudocode.
"""
from __future__ import annotations
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Per-file generation prompts ───────────────────────────────────────────────

_FILE_PROMPT = """\
You are a senior Python engineer writing production-grade code.

=== PROJECT SPEC ===
{spec_json}

=== ARCHITECTURE PLAN ===
{plan_json}

=== YOUR TASK ===
Generate ONLY the file: `{filename}`

Requirements:
- Python 3.11
- Type hints on ALL functions
- Docstrings on ALL classes and functions
- Logging with `logging.getLogger(__name__)`
- Proper error handling (try/except with specific exceptions)
- NO placeholder comments like "# TODO" or "# implement this"
- NO pseudocode — real, runnable implementation
- Only import packages listed in the plan's "dependencies" plus stdlib
- Match the function signatures defined in the plan exactly

{extra_instructions}

Return ONLY the file contents. No markdown fences. No explanation.
"""

_EXTRA = {
    "app.py": """\
- Flask app listening on host='0.0.0.0', port={flask_port}
- GET /  → use `render_template('index.html')` with no variables
- POST / → read form data, call predictor.predict(), use `render_template('index.html', result=...)`
- The HTML form must have EXACTLY these input fields (no more, no fewer): {feature_list}
- Validate all inputs before calling the endpoint (return 400 or render error if invalid)
- Import predictor from predictor.py in the same directory
""",

    "templates/index.html": """\
- Read the PROJECT SPEC above to understand the model's purpose.
- Create a beautiful, modern HTML5 web page for the model using embedded CSS (dark-mode, clean typography).
- Include an HTML `<form>` that POSTs to `/`.
- The form MUST have exactly these input fields (name attributes must match exactly): {feature_list}
- Add a beautiful section to display the prediction result (e.g. `{{{{% if result %}}}}...{{{{% endif %}}}}`).
- Provide the full, production-ready HTML code containing all inputs.
""",

    "predictor.py": """\
- Function `format_features(inputs: dict[str, float]) -> str`:
  Converts input dict to a comma-separated CSV string in this EXACT column order:
  {feature_list}
  This order MUST match the SageMaker XGBoost training data column order.
- Function `call_endpoint(csv_row: str, endpoint_name: str, region: str) -> float`:
  Calls boto3 SageMaker Runtime invoke_endpoint with ContentType='text/csv'.
  Parses and returns the float prediction.
- Function `predict(inputs: dict[str, float], endpoint_name: str, region: str) -> dict`:
  Full pipeline: validate → format → call → return {{'prediction': value, 'status': 'ok'}}
- Endpoint name: {endpoint_name}
- AWS region: {aws_region}
- Handle boto3.ClientError, ValueError, and generic Exception separately
""",

    "train.py": """\
- Function `load_dataset(path_or_uri: str) -> pd.DataFrame`:
  Load CSV from local path or s3:// URI (use boto3 for S3)
- Function `compute_metrics(y_true, y_pred, task_type: str) -> dict`:
  For regression: return {{'rmse': float, 'mae': float, 'r2': float}}
  For classification: return {{'accuracy': float, 'f1': float}}
- Function `log_training_summary(metrics: dict, endpoint_name: str) -> None`:
  Log all metrics and the endpoint URL in a clean formatted block
- This is a utilities module, NOT a training entry point (SageMaker runs training)
""",

    "requirements.txt": """\
Output a plain requirements.txt with these packages (no versions, one per line):
flask, boto3, pandas, numpy
Nothing else.
""",

    "README.md": """\
Write a clear README with these sections:
1. Project Overview (what it does, what it predicts)
2. Setup (pip install, AWS credentials)
3. Running the App (python app.py command)
4. API Usage (what inputs to provide, what output to expect)
5. Dataset Info (feature names and target)
6. Architecture (one paragraph)
Use markdown formatting.
""",

    "tests/test_app.py": """\
- Use pytest (no unittest)
- Mock boto3 SageMaker calls with unittest.mock.patch
- Test `format_features` produces correct CSV string in correct column order
- Test `predict` returns a dict with 'prediction' key on mocked success
- Test `predict` raises or returns error dict on endpoint failure
- Test Flask GET / returns 200 with an HTML form
- Test Flask POST / with valid data returns 200
- No network calls — all external calls must be mocked
- Import from app and predictor modules using relative path setup in conftest or sys.path
""",
}


def _clean(text: str) -> str:
    """Strip markdown code fences."""
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$",        "", text.strip(), flags=re.MULTILINE)
    return text.strip()


class CodeGenFactory:
    """Generate all project files from ProjectSpec + Architecture Plan."""

    def __init__(self, llm_client, config: dict):
        self._llm    = llm_client
        self._cfg    = config
        self._ws_dir = Path(config.get("codegen", {})
                           .get("workspace_dir",
                                "Code_Generator/RAD-ML/workspace/current_app"))
        self._ws_dir.mkdir(parents=True, exist_ok=True)
        self._port   = int(config.get("codegen", {}).get("flask_port", 7000))

    # ── public ────────────────────────────────────────────────────────────────
    def generate_all(self, project_spec: dict, plan: dict) -> dict[str, Path]:
        """
        Generate every file in project_spec['deliverables'].

        Returns
        -------
        dict mapping filename → absolute Path of the written file
        """
        import json
        spec_json = json.dumps(project_spec, indent=2)
        plan_json = json.dumps(plan, indent=2)

        features      = project_spec.get("feature_cols", project_spec.get("features", []))
        feature_list  = ", ".join(features)
        endpoint_name = project_spec.get("endpoint_name", "radml-endpoint")
        aws_region    = project_spec.get("aws_region",    "us-east-1")
        target        = project_spec.get("target_col",    project_spec.get("target", "output"))
        flask_port    = project_spec.get("flask_port",    self._port)

        deliverables = project_spec.get("deliverables", list(_EXTRA.keys()))
        written: dict[str, Path] = {}

        for filename in deliverables:
            logger.info("Generating %s …", filename)

            extra_tmpl = _EXTRA.get(filename, "")
            extra = extra_tmpl.format(
                feature_list  = feature_list,
                endpoint_name = endpoint_name,
                aws_region    = aws_region,
                target        = target,
                flask_port    = flask_port,
            ) if extra_tmpl else ""

            prompt = _FILE_PROMPT.format(
                spec_json = spec_json,
                plan_json = plan_json,
                filename  = filename,
                extra_instructions = extra,
            )

            try:
                raw   = self._llm.generate(prompt)
                code  = _clean(raw)
            except Exception as exc:
                logger.error("Failed to generate %s: %s", filename, exc)
                code = f"# Generation failed: {exc}\n"

            out_path = self._ws_dir / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(code, encoding="utf-8")
            written[filename] = out_path
            logger.info("  Written: %s (%d lines)", out_path, code.count("\n"))

        return written
