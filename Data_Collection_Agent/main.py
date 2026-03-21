"""
Data collection entrypoint for RAD-ML.

Pipeline:
1. Parse prompt into a dataset search spec.
2. Search Kaggle, UCI, and OpenML.
3. Rank the candidates.
4. Download and score usable tabular files.
5. Merge the final dataset and upload results metadata.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
for import_path in (str(ROOT), str(PROJECT_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)


def _load_config(path=None) -> dict:
    import yaml

    for candidate in [path, ROOT / "config.yaml", PROJECT_ROOT / "config.yaml"]:
        if candidate and Path(candidate).exists():
            with open(candidate, encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
    return {}


def _setup_logging(config: dict) -> None:
    log_cfg = config.get("logging", {})
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("console_level", "INFO").upper(), logging.INFO),
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_cfg.get("log_file", "logs/rad_ml.log")),
        ],
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def run_collection(prompt: str, config: dict, job_id: str, log_fn=None) -> dict:
    from brain.prompt_parser import PromptParser
    from collectors.kaggle_collector import KaggleCollector
    from collectors.openml_collector import OpenMLCollector
    from collectors.uci_collector import UCICollector
    from collectors.huggingface_collector import HuggingFaceCollector
    from utils.dataset_merger import DatasetMerger
    from utils.dataset_scorer import DatasetScorer
    from utils.s3_uploader import S3Uploader

    log = logging.getLogger(__name__)

    def step(name: str, msg: str) -> None:
        log.info("[%s] %s", name, msg)
        if log_fn:
            log_fn(name, msg)

    step("parse", f"Parsing prompt: {prompt[:80]}")
    spec = PromptParser().parse(prompt)
    step(
        "parse",
        (
            f"Intent classified -> intent={spec['intent'].upper()}  "
            f"task={spec['task_type'].upper()}  domain='{spec['domain']}'"
        ),
    )
    step(
        "parse",
        (
            f"Search keywords: {spec['keywords']}  |  "
            f"Inputs: {spec['input_params']}  |  "
            f"Target: {spec['target_param']}"
        ),
    )
    if spec.get("fallback_refs"):
        step("parse", f"Fallback datasets ready: {spec['fallback_refs'][:3]}")

    keywords = spec["keywords"]
    fallback_refs = spec.get("fallback_refs", [])

    kaggle = KaggleCollector(config)
    uci = UCICollector(config)
    openml = OpenMLCollector(config)
    huggingface = HuggingFaceCollector(config)
    scorer = DatasetScorer(config)

    all_metas: list[dict] = []
    seen_refs: set[str] = set()

    def add_unique(metas: list[dict]) -> None:
        for meta in metas:
            ref = meta.get("ref", "")
            if ref and ref not in seen_refs:
                seen_refs.add(ref)
                all_metas.append(meta)

    step("search", "Tier 1 - searching Kaggle + UCI + OpenML + HuggingFace")
    for keyword in keywords[:4]:
        step("search", f"  query: '{keyword}'")
        kaggle_results = kaggle.search(keyword)
        uci_results = uci.search(keyword)
        openml_results = openml.search(keyword, spec)
        hf_results = huggingface.search(keyword, spec)
        step(
            "search",
            (
                f"  '{keyword}' -> Kaggle:{len(kaggle_results)}  "
                f"UCI:{len(uci_results)}  OpenML:{len(openml_results)}  "
                f"HuggingFace:{len(hf_results)}"
            ),
        )
        add_unique(kaggle_results)
        add_unique(uci_results)
        add_unique(openml_results)
        add_unique(hf_results)
        if len(all_metas) >= 5:
            break

    # NEW: Smart Tier 2 - check Kaggle availability before attempting
    if not all_metas and fallback_refs:
        kaggle_available = kaggle.are_credentials_available()
        
        if not kaggle_available:
            step(
                "search",
                f"Tier 2 - SKIPPED (Kaggle credentials not available; moving to Tier 3)",
            )
        else:
            step(
                "search",
                f"Tier 2 - live search returned 0; resolving {len(fallback_refs)} fallback dataset(s)",
            )
            for dataset_ref in fallback_refs[:5]:
                step("search", f"  resolving fallback: kaggle/{dataset_ref}")
                meta_list = kaggle.search_by_ref(dataset_ref)
                step("search", f"    -> resolved to {len(meta_list)} metadata record(s)")
                add_unique(meta_list)
            if all_metas:
                step("search", f"Tier 2 success: resolved {len(all_metas)} fallback dataset(s)")

    if not all_metas:
        step("search", "Tier 3 - Kaggle unavailable or exhausted; trying OpenML deep search")
        for keyword in keywords[:5]:
            step("search", f"  OpenML search: '{keyword}'")
            openml_results = openml.search(keyword, spec)
            if openml_results:
                step("search", f"  OpenML found {len(openml_results)} dataset(s) for '{keyword}'")
                add_unique(openml_results)
            else:
                step("search", f"  OpenML found 0 dataset(s) for '{keyword}'")
            if len(all_metas) >= 5:
                break

        if not all_metas and spec.get("task_type"):
            task = spec["task_type"].lower()
            step("search", f"Tier 3b - OpenML task-based search: '{task}'")
            openml_results = openml.search(task, spec)
            if openml_results:
                step("search", f"  OpenML found {len(openml_results)} dataset(s) for task '{task}'")
                add_unique(openml_results)
            else:
                step("search", f"  OpenML found 0 dataset(s) for task '{task}'")

        if not all_metas and spec.get("domain"):
            domain = spec["domain"].lower()
            step("search", f"Tier 3c - OpenML domain-based search: '{domain}'")
            openml_results = openml.search(domain, spec)
            if openml_results:
                step("search", f"  OpenML found {len(openml_results)} dataset(s) for domain '{domain}'")
                add_unique(openml_results)

        if all_metas:
            step("search", f"Tier 3 success: found {len(all_metas)} dataset(s) on OpenML")

    if not all_metas:
        step("search", "Tier 4 - HuggingFace Hub search (no API key required)")
        for keyword in keywords[:4]:
            step("search", f"  HuggingFace search: '{keyword}'")
            hf_results = huggingface.search(keyword, spec)
            if hf_results:
                step("search", f"  HuggingFace found {len(hf_results)} dataset(s) for '{keyword}'")
                add_unique(hf_results)
            else:
                step("search", f"  HuggingFace found 0 dataset(s) for '{keyword}'")
            if len(all_metas) >= 5:
                break
        
        if all_metas:
            step("search", f"Tier 4 success: found {len(all_metas)} dataset(s) on HuggingFace")

    if not all_metas:
        kg_user = config.get("kaggle", {}).get("username", "")
        kg_key = config.get("kaggle", {}).get("key", "")
        error_msg = (
            "No datasets found after exhausting all search tiers:\n"
            "  x Tier 1: Kaggle/UCI/OpenML live search\n"
            "  x Tier 2: Kaggle fallback dataset refs\n"
            "  x Tier 3: OpenML keyword/task/domain/general search\n"
            "  x Tier 4: HuggingFace Hub search\n\n"
            "Diagnosis:\n"
            f"  Kaggle configured : {'no' if not kg_user else 'yes'}\n"
            f"  Kaggle API key    : {'not set' if not kg_key else 'set'}\n"
            f"  Intent detected   : {spec.get('intent', '?').upper()}\n"
            f"  Task type         : {spec.get('task_type', '?').upper()}\n"
            f"  Domain detected   : '{spec.get('domain', 'n/a')}'\n"
            f"  Keywords tried    : {keywords}\n\n"
            "Fixes to try:\n"
            "  1. Use the 'Upload CSV' button if you already have a dataset.\n"
            "  2. Verify Kaggle username and API key in config.yaml.\n"
            "  3. Regenerate your Kaggle API token.\n"
            "  4. Check internet connectivity.\n"
            "  5. Try broader search terms.\n"
            "  6. Retry later if provider APIs are unavailable.\n"
        )
        raise RuntimeError(error_msg)

    for meta in all_metas:
        scorer.score_metadata(meta, spec)
    all_metas.sort(key=lambda meta: meta.get("pre_score", 0), reverse=True)

    top_k = all_metas[:8]
    step(
        "score",
        (
            f"Best candidate: '{top_k[0]['title']}'  "
            f"source={top_k[0]['source'].upper()}  "
            f"score={top_k[0].get('pre_score', 0):.3f}"
        ),
    )

    min_rows = int(config.get("collection", {}).get("min_row_threshold", 500))
    scored_csvs: list[tuple] = []
    download_failures: list[str] = []

    for meta in top_k:
        source = meta.get("source", "kaggle")
        ref = meta.get("ref", "")
        step("download", f"[{source.upper()}] {meta.get('title', '')[:55]}")

        try:
            if source == "kaggle":
                csv_paths = kaggle.download(ref)
                source_error = kaggle.last_error
            elif source == "uci":
                csv_paths = uci.download(ref)
                source_error = ""
            elif source == "openml":
                csv_paths = openml.download(ref)
                source_error = ""
            elif source == "huggingface":
                csv_paths = huggingface.download(ref)
                source_error = huggingface.last_error
            else:
                csv_paths = []
                source_error = f"Unsupported source '{source}'."
        except Exception as exc:
            csv_paths = []
            source_error = str(exc)
            step("download", f"  -> Download error: {exc}")

        if not csv_paths:
            if source_error:
                step("download", f"  -> {source_error}")
                download_failures.append(f"{source.upper()} {ref}: {source_error}")
            else:
                step("download", "  -> No usable tabular files extracted, skipping.")
                download_failures.append(
                    f"{source.upper()} {ref}: no usable tabular files extracted"
                )
            continue

        csv_path = max(csv_paths, key=lambda file_path: file_path.stat().st_size)
        final_score = scorer.score_csv(csv_path, meta, spec)
        meta["local_path"] = str(csv_path)
        scored_csvs.append((csv_path, final_score, meta))

        step(
            "download",
            f"  -> {csv_path.name}  rows={meta.get('row_count', '?')}  score={final_score:.3f}",
        )

        if meta.get("row_count", 0) >= min_rows and len(scored_csvs) >= 2:
            step("download", "Row threshold met - stopping early.")
            break

    if not scored_csvs:
        detail_block = "\n".join(f"  -> {message}" for message in download_failures[:5])
        message = "Found dataset metadata but all downloads failed.\n"
        if detail_block:
            message += f"{detail_block}\n"
        message += (
            "  -> Check Kaggle credentials and network connectivity\n"
            "  -> Use 'Upload CSV' to supply your own dataset"
        )
        raise RuntimeError(message)

    scored_csvs.sort(key=lambda item: item[1], reverse=True)

    step("merge", "Building final dataset")
    merger = DatasetMerger(config)
    ds_info = merger.build_final_dataset(scored_csvs, job_id)
    step(
        "merge",
        f"Final dataset: {ds_info['row_count']:,} rows x {len(ds_info['columns'])} cols  merged={ds_info['merged']}",
    )

    step("upload", "Uploading to S3")
    uploader = S3Uploader(config)
    s3_uri = uploader.upload_dataset(ds_info["path"], job_id)
    step("upload", f"S3 URI: {s3_uri or '(disabled/skipped)'}")

    payload = {
        "job_id": job_id,
        "prompt": prompt,
        "spec": spec,
        "dataset": {
            "local_path": str(ds_info["path"]),
            "s3_uri": s3_uri,
            "columns": ds_info["columns"],
            "row_count": ds_info["row_count"],
            "source_count": ds_info["source_count"],
            "preview_rows": ds_info["preview_rows"],
            "merged": ds_info["merged"],
        },
        "top_sources": [
            {
                "source": meta.get("source"),
                "title": meta.get("title"),
                "url": meta.get("url"),
                "final_score": meta.get("final_score", meta.get("pre_score", 0)),
                "row_count": meta.get("row_count", 0),
            }
            for _, _, meta in scored_csvs[:3]
        ],
    }

    results_path = Path(
        config.get("storage", {}).get("results_json", "Data_Collection_Agent/db_results.json")
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    uploader.upload_results_json(payload, job_id)
    step(
        "done",
        f"Collection complete - {ds_info['row_count']:,} rows from {ds_info['source_count']} source(s)",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    _setup_logging(config)
    job_id = args.job_id or str(uuid.uuid4())[:8]

    try:
        result = run_collection(args.prompt, config, job_id)
        print(f"\nDataset : {result['dataset']['local_path']}")
        print(f"Rows    : {result['dataset']['row_count']:,}")
        print(f"Columns : {len(result['dataset']['columns'])}")
    except Exception as exc:
        logging.getLogger(__name__).critical("Collection failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
