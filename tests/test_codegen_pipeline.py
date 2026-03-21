"""
tests/test_codegen_pipeline.py
================================
Unit tests for the 5-layer code generation pipeline.
All LLM calls are mocked — no API key needed.
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Code_Generator" / "RAD-ML"))

from generator.prompt_understanding import PromptUnderstandingLayer
from generator.planner import Planner
from generator.code_gen_factory import CodeGenFactory
from generator.validator import Validator, FileReport
from generator.repair_loop import RepairLoop

# ── Shared fixtures ───────────────────────────────────────────────────────────

SPEC = {
    "task":          "housing price prediction",
    "language":      "Python",
    "framework":     "Flask",
    "task_type":     "regression",
    "model_type":    "XGBoost regressor",
    "features":      ["bedrooms", "bathrooms", "sqft"],
    "feature_cols":  ["bedrooms", "bathrooms", "sqft"],
    "target":        "price",
    "target_col":    "price",
    "deliverables":  ["app.py", "predictor.py", "requirements.txt"],
    "constraints":   ["use SageMaker endpoint"],
    "coding_style":  "production",
    "endpoint_name": "test-endpoint",
    "aws_region":    "us-east-1",
    "flask_port":    7000,
}

PLAN = {
    "architecture_overview": "Flask app calling SageMaker",
    "file_structure": {"app.py": "web app", "predictor.py": "endpoint caller"},
    "key_functions": {
        "app.py": ["index() -> str", "predict() -> str"],
        "predictor.py": ["format_features(inputs) -> str",
                         "call_endpoint(csv_row, ep, region) -> float",
                         "predict(inputs, ep, region) -> dict"],
    },
    "dependencies": ["flask", "boto3"],
    "validation_strategy": "AST + pytest",
    "edge_cases": ["invalid input"],
    "feature_order_note": "bedrooms, bathrooms, sqft",
}

CFG = {
    "codegen": {
        "workspace_dir":       tempfile.mkdtemp(),
        "flask_port":          7000,
        "max_fix_attempts":    2,
        "test_timeout_seconds": 5,
    }
}


# ── Layer 1: PromptUnderstandingLayer ────────────────────────────────────────

def test_understanding_uses_llm_output():
    llm = MagicMock()
    llm.generate.return_value = json.dumps(SPEC)

    layer = PromptUnderstandingLayer(llm)
    result = layer.build_spec(
        prompt="predict house price",
        parsed_spec={"task_type": "regression", "input_params": ["bedrooms"],
                     "target_param": "price", "raw": "predict house price"},
        dataset_info={"columns": ["bedrooms", "price"], "row_count": 600},
        sm_meta={"endpoint_name": "test-endpoint"},
        preprocess_result={"feature_cols": ["bedrooms"], "target_col": "price"},
        config=CFG,
    )
    assert result["task_type"] == "regression"
    assert "endpoint_name" in result


def test_understanding_fallback_on_llm_failure():
    llm = MagicMock()
    llm.generate.side_effect = RuntimeError("LLM down")

    layer = PromptUnderstandingLayer(llm)
    result = layer.build_spec(
        prompt="predict house price",
        parsed_spec={"task_type": "regression", "input_params": ["bedrooms"],
                     "target_param": "price", "raw": "predict house price"},
        dataset_info={"columns": ["bedrooms", "price"], "row_count": 600},
        sm_meta={"endpoint_name": "test-endpoint"},
        preprocess_result={"feature_cols": ["bedrooms"], "target_col": "price"},
        config=CFG,
    )
    # Fallback spec must have all required keys
    for key in ("task_type", "features", "target", "endpoint_name", "deliverables"):
        assert key in result, f"Missing key: {key}"


# ── Layer 2: Planner ──────────────────────────────────────────────────────────

def test_planner_returns_plan():
    llm = MagicMock()
    llm.generate.return_value = json.dumps(PLAN)

    planner = Planner(llm)
    plan = planner.plan(SPEC)
    assert "file_structure" in plan
    assert "dependencies" in plan


def test_planner_fallback_on_failure():
    llm = MagicMock()
    llm.generate.side_effect = RuntimeError("LLM down")

    planner = Planner(llm)
    plan = planner.plan(SPEC)
    assert "file_structure" in plan
    assert "key_functions" in plan
    assert "dependencies" in plan


# ── Layer 3: CodeGenFactory ───────────────────────────────────────────────────

def _minimal_app_code():
    return """\
from flask import Flask, request, render_template_string
import predictor
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        inputs = {k: float(request.form[k]) for k in ["bedrooms","bathrooms","sqft"]}
        result = predictor.predict(inputs, "test-endpoint", "us-east-1")
    return render_template_string("<html><body>{{ result }}</body></html>", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000)
"""


def test_codegen_writes_files():
    llm = MagicMock()
    llm.generate.return_value = _minimal_app_code()

    factory = CodeGenFactory(llm, CFG)
    spec    = dict(SPEC)
    spec["deliverables"] = ["app.py"]

    written = factory.generate_all(spec, PLAN)
    assert "app.py" in written
    assert written["app.py"].exists()
    assert written["app.py"].read_text()


def test_codegen_handles_llm_failure():
    llm = MagicMock()
    llm.generate.side_effect = RuntimeError("LLM down")

    factory = CodeGenFactory(llm, CFG)
    spec    = dict(SPEC)
    spec["deliverables"] = ["app.py"]

    written = factory.generate_all(spec, PLAN)
    # File still written (with error comment)
    assert "app.py" in written
    assert written["app.py"].exists()


# ── Layer 4: Validator ────────────────────────────────────────────────────────

def test_validator_passes_good_app():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "app.py"
        p.write_text(_minimal_app_code())

        cfg = dict(CFG)
        cfg["codegen"] = dict(CFG["codegen"])
        cfg["codegen"]["workspace_dir"] = d

        validator = Validator(SPEC, cfg)
        report = validator.validate({"app.py": p})
        assert report.file_reports["app.py"].passed, \
            report.file_reports["app.py"].errors


def test_validator_catches_syntax_error():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "app.py"
        p.write_text("def broken(:\n    pass\n")   # deliberate syntax error

        cfg = dict(CFG)
        cfg["codegen"] = dict(CFG["codegen"])
        cfg["codegen"]["workspace_dir"] = d

        validator = Validator(SPEC, cfg)
        report = validator.validate({"app.py": p})
        assert not report.file_reports["app.py"].passed
        assert any("SyntaxError" in e for e in report.file_reports["app.py"].errors)


def test_validator_catches_missing_predictor_import():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "app.py"
        # Missing `import predictor` and missing route
        p.write_text("from flask import Flask\napp = Flask(__name__)\n")

        cfg = dict(CFG)
        cfg["codegen"] = dict(CFG["codegen"])
        cfg["codegen"]["workspace_dir"] = d

        validator = Validator(SPEC, cfg)
        report = validator.validate({"app.py": p})
        assert not report.file_reports["app.py"].passed


# ── Layer 5: RepairLoop ───────────────────────────────────────────────────────

def test_repair_loop_fixes_file():
    with tempfile.TemporaryDirectory() as d:
        # Start with broken file
        p = Path(d) / "app.py"
        p.write_text("def broken(:\n    pass\n")

        cfg = dict(CFG)
        cfg["codegen"] = dict(CFG["codegen"])
        cfg["codegen"]["workspace_dir"] = d

        # Mock LLM to return valid code
        llm = MagicMock()
        llm.generate.return_value = _minimal_app_code()

        validator = Validator(SPEC, cfg)
        # Build initial report showing the file fails
        report = validator.validate({"app.py": p})
        assert not report.all_passed

        loop = RepairLoop(llm, SPEC, PLAN, cfg)
        _, final_report = loop.repair({"app.py": p}, report, validator)

        assert final_report.all_passed


def test_repair_loop_stops_after_max_attempts():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "app.py"
        p.write_text("def broken(:\n    pass\n")

        cfg = dict(CFG)
        cfg["codegen"] = dict(CFG["codegen"])
        cfg["codegen"]["workspace_dir"] = d
        cfg["codegen"]["max_fix_attempts"] = 2

        # LLM keeps returning broken code
        llm = MagicMock()
        llm.generate.return_value = "def still_broken(:\n    pass\n"

        validator = Validator(SPEC, cfg)
        report = validator.validate({"app.py": p})

        loop = RepairLoop(llm, SPEC, PLAN, cfg)
        _, final_report = loop.repair({"app.py": p}, report, validator)

        # Should not crash — just report remaining failures
        assert not final_report.all_passed
        # LLM should have been called at most max_fix_attempts times
        assert llm.generate.call_count <= 2
