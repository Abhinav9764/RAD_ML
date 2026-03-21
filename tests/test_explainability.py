"""
tests/test_explainability.py
==============================
Tests for the ExplainabilityEngine — all LLM calls mocked, no API key needed.
"""
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Code_Generator" / "RAD-ML"))

from explainability.engine import ExplainabilityEngine, _ALGO_KB, _build_diagram_png

# ── Fixtures ──────────────────────────────────────────────────────────────────
SAMPLE_JOB_RESULT = {
    "deploy_url":    "http://localhost:7000",
    "endpoint_name": "test-ep-abc123",
    "sm_meta": {
        "endpoint_name": "test-ep-abc123",
        "job_name":      "radml-job-abc",
        "status":        "mock_completed",
    },
    "dataset": {
        "row_count":    1200,
        "columns":      ["bedrooms", "bathrooms", "sqft", "location", "price"],
        "merged":       False,
        "source_count": 1,
        "s3_uri":       "s3://rad-ml-datasets/collected_data/test01/data.csv",
        "preview_rows": [{"bedrooms": 3, "bathrooms": 2, "sqft": 1500,
                          "location": "urban", "price": 350000}],
    },
    "model": {
        "task_type":    "regression",
        "feature_cols": ["bedrooms", "bathrooms", "sqft", "location"],
        "target_col":   "price",
        "stats": {"train_rows": 960, "val_rows": 240, "num_features": 4},
    },
}

SAMPLE_DB_RESULTS = {
    "prompt":   "Predict housing price based on bedrooms, bathrooms and location",
    "job_id":   "test01",
    "spec": {
        "task_type":    "regression",
        "input_params": ["bedrooms", "bathrooms", "sqft", "location"],
        "target_param": "price",
        "keywords":     ["housing", "price", "bedrooms"],
    },
    "top_sources": [
        {"title": "House Prices Dataset", "source": "kaggle",
         "url": "https://www.kaggle.com/datasets/test",
         "row_count": 1200, "final_score": 0.85},
    ],
}

CFG = {
    "codegen": {"workspace_dir": tempfile.mkdtemp(), "flask_port": 7000},
    "llm": {"gemini_api_key": ""},
}

_NARRATIVE = "## Your model is ready!\nIt predicts **price** from bedrooms, bathrooms, sqft."


def _make_engine(narrative=_NARRATIVE):
    llm = MagicMock()
    llm.generate.return_value = narrative
    return ExplainabilityEngine(llm, CFG), llm


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_explain_returns_all_keys():
    engine, _ = _make_engine()
    result = engine.explain(SAMPLE_JOB_RESULT, SAMPLE_DB_RESULTS)
    required = {"narrative", "algorithm_card", "usage_guide",
                "data_story", "architecture_diagram_b64", "code_preview"}
    assert required.issubset(set(result.keys())), \
        f"Missing keys: {required - set(result.keys())}"


def test_narrative_uses_llm():
    engine, llm = _make_engine()
    result = engine.explain(SAMPLE_JOB_RESULT, SAMPLE_DB_RESULTS)
    assert llm.generate.called
    assert "price" in result["narrative"] or len(result["narrative"]) > 10


def test_narrative_fallback_on_llm_failure():
    engine, llm = _make_engine()
    llm.generate.side_effect = RuntimeError("LLM down")
    result = engine.explain(SAMPLE_JOB_RESULT, SAMPLE_DB_RESULTS)
    # Fallback should still produce meaningful text
    assert len(result["narrative"]) > 50
    assert "price" in result["narrative"].lower() or "model" in result["narrative"].lower()


def test_algorithm_card_regression():
    engine, _ = _make_engine()
    result = engine.explain(SAMPLE_JOB_RESULT, SAMPLE_DB_RESULTS)
    card = result["algorithm_card"]
    assert card["name"] == "XGBoost Regressor"
    assert len(card["strengths"]) > 0
    assert len(card["limitations"]) > 0
    assert len(card["metrics"]) > 0
    assert "why_chosen" in card
    assert "how_it_works" in card


def test_algorithm_card_classification():
    engine, _ = _make_engine()
    job = dict(SAMPLE_JOB_RESULT)
    job["model"] = dict(SAMPLE_JOB_RESULT["model"])
    job["model"]["task_type"] = "classification"
    db = dict(SAMPLE_DB_RESULTS)
    db["spec"] = dict(SAMPLE_DB_RESULTS["spec"])
    db["spec"]["task_type"] = "classification"
    result = engine.explain(job, db)
    assert "Classifier" in result["algorithm_card"]["name"]


def test_algorithm_card_clustering():
    engine, _ = _make_engine()
    job = dict(SAMPLE_JOB_RESULT)
    job["model"] = dict(SAMPLE_JOB_RESULT["model"])
    job["model"]["task_type"] = "clustering"
    db = dict(SAMPLE_DB_RESULTS)
    db["spec"] = dict(SAMPLE_DB_RESULTS["spec"])
    db["spec"]["task_type"] = "clustering"
    result = engine.explain(job, db)
    assert "clustering" in result["algorithm_card"]["name"].lower() or \
           "similarity" in result["algorithm_card"]["name"].lower()


def test_usage_guide_has_five_steps():
    engine, _ = _make_engine()
    result = engine.explain(SAMPLE_JOB_RESULT, SAMPLE_DB_RESULTS)
    guide = result["usage_guide"]
    assert len(guide) == 5
    for step in guide:
        assert "step" in step
        assert "title" in step
        assert "detail" in step
        assert "icon" in step


def test_usage_guide_references_inputs():
    engine, _ = _make_engine()
    result = engine.explain(SAMPLE_JOB_RESULT, SAMPLE_DB_RESULTS)
    step2 = result["usage_guide"][1]   # "Fill in the input form"
    assert "bedrooms" in step2["detail"] or "4" in step2["detail"]


def test_data_story_structure():
    engine, _ = _make_engine()
    result = engine.explain(SAMPLE_JOB_RESULT, SAMPLE_DB_RESULTS)
    story = result["data_story"]
    assert "summary" in story
    assert "sources" in story
    assert "search_strategy" in story
    assert "1,200" in story["summary"] or "1200" in story["summary"]


def test_data_story_sources_detail():
    engine, _ = _make_engine()
    result = engine.explain(SAMPLE_JOB_RESULT, SAMPLE_DB_RESULTS)
    sources = result["data_story"]["sources"]
    assert len(sources) >= 1
    assert sources[0]["source"] in ("KAGGLE", "UCI", "OPENML")
    assert "score" in sources[0]


def test_code_preview_returns_dict():
    with tempfile.TemporaryDirectory() as d:
        # Write a sample file
        p = Path(d) / "app.py"
        p.write_text("from flask import Flask\napp = Flask(__name__)\n")

        engine, _ = _make_engine()
        result = engine.explain(
            SAMPLE_JOB_RESULT, SAMPLE_DB_RESULTS,
            written_files={"app.py": str(p)},
        )
        assert "app.py" in result["code_preview"]
        assert "flask" in result["code_preview"]["app.py"].lower()


def test_code_preview_truncates_at_60_lines():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "big_file.py"
        p.write_text("\n".join([f"line_{i} = {i}" for i in range(100)]))

        engine, _ = _make_engine()
        result = engine.explain(
            SAMPLE_JOB_RESULT, SAMPLE_DB_RESULTS,
            written_files={"big_file.py": str(p)},
        )
        preview = result["code_preview"]["big_file.py"]
        assert "40 more lines" in preview   # 100 - 60 = 40


def test_diagram_skips_gracefully_without_graphviz(monkeypatch):
    """When diagrams library raises, explanation should still succeed."""
    import explainability.engine as eng_mod
    original = eng_mod._build_diagram_png

    def _fail(*args, **kwargs):
        raise RuntimeError("graphviz not installed")
    monkeypatch.setattr(eng_mod, "_build_diagram_png", _fail)

    engine, _ = _make_engine()
    result = engine.explain(SAMPLE_JOB_RESULT, SAMPLE_DB_RESULTS)
    # diagram_b64 is empty string but everything else still works
    assert result["architecture_diagram_b64"] == ""
    assert result["narrative"]
    assert result["algorithm_card"]


def test_algo_kb_has_all_task_types():
    for task in ("regression", "classification", "clustering"):
        assert task in _ALGO_KB
        card = _ALGO_KB[task]
        for key in ("name", "family", "why_chosen", "how_it_works",
                    "strengths", "limitations", "metrics"):
            assert key in card, f"Missing '{key}' in _ALGO_KB['{task}']"
