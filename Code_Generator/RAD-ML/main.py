"""
Code_Generator/RAD-ML/main.py
==============================
Code Generator pipeline — full 5-layer architecture + Explainability.

FIX (v6):
  - Moved 'explanation' variable assignment BEFORE it is referenced in 'result' dict.
    Previously: result dict was built referencing `explanation` (undefined at that point),
    and Layer 6 (explainability) ran AFTER — causing NameError / stale reference.
  - Fixed by initialising explanation = {} before result dict, then computing it
    in Layer 6 and updating result["explanation"] afterwards.
"""
from __future__ import annotations
import json
import logging
import sys
import uuid
from pathlib import Path

ROOT         = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
for p in (str(ROOT), str(ROOT.parent), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_config(path: str | None = None) -> dict:
    import yaml
    for c in [path, ROOT / "config.yaml", PROJECT_ROOT / "config.yaml"]:
        if c and Path(c).exists():
            with open(c, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}


# ── main callable ─────────────────────────────────────────────────────────────
def run_codegen(db_results: dict, config: dict, job_id: str,
                log_fn=None) -> dict:
    """
    Full 5-layer code generation pipeline + explainability.

    Parameters
    ----------
    db_results : payload from Data_Collection_Agent run_collection()
    config     : loaded YAML config dict
    job_id     : unique job identifier
    log_fn     : optional callable(step, message) for live UI streaming

    Returns
    -------
    {
      "endpoint_name", "deploy_url", "app_path",
      "sm_meta", "preprocess", "feature_order",
      "generated_files", "validation_summary", "explanation"
    }
    """
    from engines.ml_engine.data_preprocessor import DataPreprocessor
    from engines.ml_engine.sagemaker_handler import SageMakerHandler
    from core.llm_client import LLMClient
    from generator.prompt_understanding import PromptUnderstandingLayer
    from generator.planner import Planner
    from generator.code_gen_factory import CodeGenFactory
    from generator.validator import Validator
    from generator.repair_loop import RepairLoop
    from explainability.engine import ExplainabilityEngine

    log = logging.getLogger(__name__)

    def step(name: str, msg: str) -> None:
        log.info("[%s] %s", name, msg)
        if log_fn:
            log_fn(name, msg)

    spec     = db_results.get("spec", {})
    ds_info  = db_results.get("dataset", {})
    csv_path = Path(ds_info.get("local_path", ""))

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found at '{csv_path}'. "
            "Run the Data Collection Agent first."
        )

    # ── Preprocessing + SageMaker ─────────────────────────────────────────────
    step("preprocess", f"Preprocessing {csv_path.name} …")
    preprocessor = DataPreprocessor(config)
    pre_result   = preprocessor.preprocess(csv_path, spec, job_id)
    step("preprocess",
         f"Train: {pre_result['stats']['train_rows']} rows | "
         f"Features ({len(pre_result['feature_cols'])}): {pre_result['feature_cols']} | "
         f"Target: {pre_result['target_col']}")

    step("sagemaker", "Uploading training data to S3 …")
    sm = SageMakerHandler(config)
    train_s3, val_s3 = sm.upload_data(
        pre_result["train_path"], pre_result["val_path"], job_id
    )
    step("sagemaker", f"Train data → {train_s3}")

    step("sagemaker", "Launching SageMaker XGBoost training job …")
    sm_meta = sm.run_training(
        s3_input_uri      = train_s3,
        target_column     = pre_result["target_col"],
        preprocess_result = pre_result,
    )
    step("sagemaker",
         f"Job: {sm_meta['job_name']} | "
         f"Status: {sm_meta['status']} | "
         f"Endpoint: {sm_meta['endpoint_name']}")

    # ── Layer 1: Prompt Understanding ─────────────────────────────────────────
    step("understand", "Building structured ProjectSpec from prompt …")
    llm           = LLMClient(config)
    understanding = PromptUnderstandingLayer(llm)
    project_spec  = understanding.build_spec(
        prompt            = db_results.get("prompt", ""),
        parsed_spec       = spec,
        dataset_info      = ds_info,
        sm_meta           = sm_meta,
        preprocess_result = pre_result,
        config            = config,
    )
    step("understand",
         f"Spec: task='{project_spec.get('task')}' | "
         f"type={project_spec.get('task_type')} | "
         f"deliverables={project_spec.get('deliverables', [])}")

    # ── Layer 2: Planner ──────────────────────────────────────────────────────
    step("plan", "Producing architecture plan (no code yet) …")
    planner = Planner(llm)
    plan    = planner.plan(project_spec)
    step("plan",
         f"Plan: {len(plan.get('file_structure', {}))} files | "
         f"deps: {plan.get('dependencies', [])}")

    # ── Layer 3: Code Generation ──────────────────────────────────────────────
    step("codegen", "Generating project files via Gemini (file-by-file) …")
    factory       = CodeGenFactory(llm, config)
    written_files = factory.generate_all(project_spec, plan)
    step("codegen",
         f"Generated {len(written_files)} files: {list(written_files.keys())}")

    # ── Layer 4: Validation ───────────────────────────────────────────────────
    step("validate", "Running multi-stage validation …")
    validator  = Validator(project_spec, config)
    val_report = validator.validate(written_files)
    step("validate", val_report.summary())

    for fr in val_report.failed_files():
        step("validate", f"  FAIL {fr.filename}: {'; '.join(fr.errors[:2])}")

    # ── Layer 5: Repair Loop ──────────────────────────────────────────────────
    if not val_report.all_passed:
        n_fail = len(val_report.failed_files())
        step("repair", f"Repairing {n_fail} failing file(s) …")
        repair_loop = RepairLoop(llm, project_spec, plan, config)
        written_files, val_report = repair_loop.repair(
            written_files, val_report, validator
        )
        step("repair", f"After repair: {val_report.summary()}")
    else:
        step("repair", "No repairs needed — all files passed ✓")

    flask_port = int(config.get("codegen", {}).get("flask_port", 7000))
    app_path   = written_files.get(
        "app.py", list(written_files.values())[0] if written_files else ""
    )

    # ── Layer 6: Explainability Engine ────────────────────────────────────────
    # NOTE: explanation must be initialised BEFORE being referenced in result dict.
    explanation: dict = {}
    step("explain", "Generating explanation and architecture diagram …")
    try:
        explain_engine = ExplainabilityEngine(llm, config)
        explanation = explain_engine.explain(
            job_result={
                "deploy_url":    f"http://localhost:{flask_port}",
                "endpoint_name": sm_meta.get("endpoint_name"),
                "sm_meta":       sm_meta,
                "dataset": {
                    "row_count":    int(ds_info.get("row_count", 0)),
                    "columns":      ds_info.get("columns", []),
                    "merged":       ds_info.get("merged", False),
                    "source_count": ds_info.get("source_count", 1),
                    "s3_uri":       ds_info.get("s3_uri"),
                    "preview_rows": ds_info.get("preview_rows", []),
                },
                "model": {
                    "task_type":    pre_result.get("task_type"),
                    "feature_cols": pre_result.get("feature_cols", []),
                    "target_col":   pre_result.get("target_col"),
                    "stats":        pre_result.get("stats", {}),
                },
            },
            db_results    = db_results,
            written_files = {k: str(v) for k, v in written_files.items()},
        )
        step("explain", "Explanation generated ✓")
    except Exception as exc:
        log.warning("Explainability engine failed: %s", exc)
        explanation = {}

    # ── Final result ──────────────────────────────────────────────────────────
    result = {
        "endpoint_name":      sm_meta.get("endpoint_name"),
        "deploy_url":         f"http://localhost:{flask_port}",
        "app_path":           str(app_path),
        "sm_meta":            sm_meta,
        "preprocess": {
            "feature_cols": pre_result["feature_cols"],
            "target_col":   pre_result["target_col"],
            "task_type":    pre_result["task_type"],
            "stats":        pre_result["stats"],
        },
        "feature_order":      pre_result["feature_cols"],
        "generated_files":    {k: str(v) for k, v in written_files.items()},
        "validation_summary": val_report.summary(),
        "explanation":        explanation,
        "project_spec":       project_spec,
    }

    step("done",
         f"Code generation complete → http://localhost:{flask_port}  "
         f"({val_report.summary()})")

    # ── Launch Flask App ──────────────────────────────────────────────────────
    if app_path:
        import subprocess
        try:
            step("deploy", f"Starting Flask app on port {flask_port} ...")
            subprocess.Popen(
                [sys.executable, str(app_path)],
                cwd=str(Path(app_path).parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            log.warning("Failed to start Flask app: %s", e)

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAD-ML Code Generator")
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--job-id",  default=None)
    parser.add_argument("--config",  default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    with open(args.results_json, encoding="utf-8") as f:
        db_results = json.load(f)

    result = run_codegen(db_results, config,
                         args.job_id or str(uuid.uuid4())[:8])
    print(json.dumps(result, indent=2, default=str))
