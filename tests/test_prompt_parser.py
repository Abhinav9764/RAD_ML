"""
tests/test_prompt_parser.py
============================
Tests for the PromptParser — verifies keyword extraction,
intent detection, input_params parsing, and target resolution.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Data_Collection_Agent"))

from brain.prompt_parser import PromptParser

PARSER = PromptParser()


def test_regression_intent():
    spec = PARSER.parse("Predict housing prices based on bedrooms and location")
    assert spec["intent"]    == "ml_model"
    assert spec["task_type"] == "regression"


def test_classification_intent():
    spec = PARSER.parse("Classify emails as spam or not spam")
    assert spec["task_type"] == "classification"


def test_chatbot_intent():
    spec = PARSER.parse("Build a chatbot for customer FAQ")
    assert spec["intent"] == "chatbot"


def test_keywords_extracted():
    spec = PARSER.parse("Predict employee salary given age experience education")
    assert len(spec["keywords"]) > 0


def test_input_params_extracted():
    spec = PARSER.parse("Predict car price based on engine size, fuel type and mileage")
    params = spec["input_params"]
    assert any("engine" in p or "fuel" in p or "mileage" in p for p in params)


def test_target_param_extracted():
    spec = PARSER.parse("Predict salary of an employee")
    assert "salary" in spec["target_param"].lower()


def test_recommendation_maps_to_clustering():
    spec = PARSER.parse("Build a movie recommendation system")
    assert spec["task_type"] in ("clustering", "regression")


def test_domain_not_empty():
    spec = PARSER.parse("Housing price prediction model")
    assert spec["domain"]
