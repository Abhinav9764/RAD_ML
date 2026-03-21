"""
explainability/engine.py
=========================
Explainability Engine — generates a rich human-readable explanation
of everything the pipeline did, using LLM + deterministic rules.

Produces:
  1. narrative               : LLM-written plain-English explanation
  2. algorithm_card          : why this algorithm, how it works, pros/cons
  3. usage_guide             : step-by-step "how to use your model"
  4. data_story              : what data was collected, from where, why
  5. architecture_diagram_b64: base64 PNG of pipeline architecture
  6. code_preview            : dict filename → first 60 lines
"""
from __future__ import annotations
import base64
import json
import logging
import textwrap
from pathlib import Path

# Add debugging support
try:
    from .debugger import DebugLogger
except ImportError:
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from debugger import DebugLogger
    except ImportError:
        DebugLogger = None

logger = logging.getLogger(__name__)

# ── Algorithm knowledge base (deterministic, no LLM needed) ──────────────────
_ALGO_KB = {
    "regression": {
        "name":        "XGBoost Regressor",
        "family":      "Gradient Boosted Decision Trees",
        "why_chosen": (
            "XGBoost was selected because your task requires predicting a continuous "
            "numeric value. It handles mixed data types without manual normalisation, "
            "is robust to missing values, and consistently outperforms linear models "
            "on tabular data. Regularisation (α=0.1, λ=1.0) prevents overfitting."
        ),
        "how_it_works": (
            "XGBoost builds an ensemble of decision trees sequentially. Each new tree "
            "corrects errors of the previous ones by fitting the residuals. "
            "With learning rate η=0.2 and max_depth=6, the model balances "
            "expressiveness against generalisation. Early stopping (10 rounds) "
            "automatically halts training when validation error stops improving."
        ),
        "strengths":   [
            "Handles missing values natively",
            "Works well on tabular data without scaling",
            "Built-in regularisation prevents overfitting",
            "Fast training on CPU with SageMaker ml.m5.large",
        ],
        "limitations": [
            "Black-box — predictions are not directly explainable",
            "May underperform on very small datasets (< 200 rows)",
            "Requires numeric or encoded inputs",
        ],
        "metrics":     ["RMSE (Root Mean Squared Error)", "MAE (Mean Absolute Error)", "R² Score"],
    },
    "classification": {
        "name":        "XGBoost Classifier",
        "family":      "Gradient Boosted Decision Trees",
        "why_chosen": (
            "XGBoost Classifier was selected for your categorical prediction task. "
            "It handles class imbalance well, supports binary and multi-class problems, "
            "and provides probability scores alongside hard predictions."
        ),
        "how_it_works": (
            "Each tree in the ensemble votes for a class. The final prediction is the "
            "class with the highest aggregate probability. Binary problems use sigmoid "
            "activation (binary:logistic); multi-class uses softmax."
        ),
        "strengths":   [
            "Handles class imbalance out of the box",
            "Provides probability confidence scores",
            "Robust to irrelevant features",
            "No need to scale input features",
        ],
        "limitations": [
            "Less interpretable than logistic regression",
            "Sensitive to hyperparameter tuning",
        ],
        "metrics":     ["Accuracy", "F1 Score", "AUC-ROC"],
    },
    "clustering": {
        "name":        "XGBoost Similarity Model",
        "family":      "Tree-Based Nearest Neighbour",
        "why_chosen": (
            "For recommendation/similarity tasks, XGBoost learns feature importance "
            "rankings. Similar items are found by computing feature-weighted distances "
            "in the trained model's leaf space — tree-based nearest neighbour."
        ),
        "how_it_works": (
            "The model learns which features matter most for your domain. At inference "
            "time, an input is passed through all trees; resulting leaf indices form a "
            "binary feature vector. Cosine similarity between these vectors identifies "
            "the most similar items in your dataset."
        ),
        "strengths":   [
            "No need for an explicit distance function",
            "Captures non-linear feature interactions",
            "Handles mixed numeric and categorical types",
        ],
        "limitations": [
            "Requires sufficient training data",
            "Memory-intensive for large item catalogues",
        ],
        "metrics":     ["Precision@K", "Recall@K", "NDCG"],
    },
}

# ── LLM prompt ────────────────────────────────────────────────────────────────
_EXPLAIN_PROMPT = """\
You are a friendly ML engineer explaining a completed pipeline to a non-technical user.

=== PIPELINE RESULTS ===
{results_json}

=== YOUR TASK ===
Write a clear, engaging explanation (400-600 words) covering:

1. **What your model does** — what input the user provides and what prediction is returned.
2. **Your dataset** — data sources, row count, columns, and why this data is relevant.
3. **Training process** — what happened during SageMaker training in plain English.
4. **How to use the live app** — step-by-step: open URL, fill the form, interpret results.
5. **Important limitations** — what the model can and cannot do reliably.
6. **Next steps** — how the user could improve results.

Rules:
- Write for a smart non-technical person. No unexplained jargon.
- Be warm, direct, specific. Use the actual feature names and target from the results.
- Use markdown headers (##), bold, and bullet lists.
- Return only the explanation text. No code blocks.
"""


class ExplainabilityEngine:
    """Generate human-readable explanations of the complete pipeline."""

    def __init__(self, llm_client, config: dict):
        self._llm    = llm_client
        self._cfg    = config
        self._ws_dir = Path(config.get("codegen", {})
                            .get("workspace_dir",
                                 "Code_Generator/RAD-ML/workspace/current_app"))
        self._ws_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize debugging
        if DebugLogger is not None:
            self._debug = DebugLogger("ExplainabilityEngine")
            self._debug.info("__init__", "ExplainabilityEngine initialized")
        else:
            self._debug = None

    # ── public ────────────────────────────────────────────────────────────────
    def explain(self, job_result: dict, db_results: dict,
                written_files: dict | None = None) -> dict:
        """
        Parameters
        ----------
        job_result    : orchestrator job.result dict
        db_results    : full db_results.json from data collection
        written_files : {filename: Path} of generated code files

        Returns
        -------
        {
          "narrative":                str,
          "algorithm_card":           dict,
          "usage_guide":              list[dict],
          "data_story":               dict,
          "architecture_diagram_b64": str,   # base64 PNG
          "code_preview":             dict,  # filename → first 60 lines
        }
        """
        if self._debug:
            self._debug.debug("explain", "Starting explanation generation")
        
        try:
            spec      = db_results.get("spec", {})
            dataset   = job_result.get("dataset", {})
            model     = job_result.get("model", {})
            sm_meta   = job_result.get("sm_meta", {})
            deploy    = job_result.get("deploy_url", "http://localhost:7000")
            sources   = db_results.get("top_sources", [])
            task_type = spec.get("task_type", "regression")

            if self._debug:
                self._debug.debug(
                    "explain",
                    f"Task type: {task_type}, Dataset rows: {dataset.get('row_count', 0)}"
                )

            algo_card   = _ALGO_KB.get(task_type, _ALGO_KB["regression"]).copy()
            narrative   = self._generate_narrative(job_result, db_results)
            usage_guide = self._build_usage_guide(deploy, spec, model, task_type)
            data_story  = self._build_data_story(dataset, sources, spec)
            diagram_b64 = self._generate_diagram(task_type, job_result)
            code_prev   = self._build_code_preview(written_files)

            if self._debug:
                self._debug.info(
                    "explain",
                    "Explanation generation completed successfully"
                )

            return {
                "narrative":               narrative,
                "algorithm_card":          algo_card,
                "usage_guide":             usage_guide,
                "data_story":              data_story,
                "architecture_diagram_b64": diagram_b64,
                "code_preview":            code_prev,
            }
        except Exception as e:
            if self._debug:
                self._debug.error(
                    "explain",
                    "Explanation generation failed",
                    exception=e
                )
            # Return safe defaults on error
            return {
                "narrative": f"Error generating explanation: {str(e)}",
                "algorithm_card": _ALGO_KB.get("regression", {}).copy(),
                "usage_guide": [],
                "data_story": {},
                "architecture_diagram_b64": "",
                "code_preview": {},
            }

    # ── LLM narrative ─────────────────────────────────────────────────────────
    def _generate_narrative(self, job_result: dict, db_results: dict) -> str:
        slim = {
            "prompt":       db_results.get("prompt", ""),
            "task_type":    db_results.get("spec", {}).get("task_type"),
            "input_params": db_results.get("spec", {}).get("input_params", []),
            "target_param": db_results.get("spec", {}).get("target_param"),
            "dataset": {
                "row_count":    job_result.get("dataset", {}).get("row_count"),
                "columns":      job_result.get("dataset", {}).get("columns", [])[:14],
                "merged":       job_result.get("dataset", {}).get("merged"),
                "source_count": job_result.get("dataset", {}).get("source_count"),
            },
            "model": {
                "task_type":    job_result.get("model", {}).get("task_type"),
                "feature_cols": job_result.get("model", {}).get("feature_cols", []),
                "target_col":   job_result.get("model", {}).get("target_col"),
                "stats":        job_result.get("model", {}).get("stats", {}),
            },
            "deploy_url":   job_result.get("deploy_url"),
            "top_sources":  db_results.get("top_sources", [])[:3],
        }
        try:
            if self._debug:
                self._debug.debug(
                    "_generate_narrative",
                    "Calling LLM to generate narrative"
                )
            result = self._llm.generate(
                _EXPLAIN_PROMPT.format(results_json=json.dumps(slim, indent=2))
            )
            if self._debug:
                self._debug.info(
                    "_generate_narrative",
                    f"Narrative generated: {len(result)} chars"
                )
            return result
        except Exception as exc:
            if self._debug:
                self._debug.warning(
                    "_generate_narrative",
                    f"LLM call failed: {str(exc)[:100]}",
                    context={"exception": type(exc).__name__}
                )
            logger.warning("Narrative LLM call failed (%s) — using fallback", exc)
            return self._fallback_narrative(db_results, job_result)

    # ── usage guide ───────────────────────────────────────────────────────────
    def _build_usage_guide(self, deploy_url: str, spec: dict,
                            model: dict, task_type: str) -> list[dict]:
        inputs = spec.get("input_params", model.get("feature_cols", []))
        target = spec.get("target_param", model.get("target_col", "prediction"))
        expected = {
            "regression":     f"a predicted **{target}** value (a number)",
            "classification": f"the predicted **{target}** category + confidence",
            "clustering":     f"recommendations ranked by similarity to your input",
        }.get(task_type, f"a predicted {target}")

        return [
            {"step": 1, "icon": "🌐", "title": "Open the live app",
             "detail": f"Visit [{deploy_url}]({deploy_url}) in your browser."},
            {"step": 2, "icon": "✏️", "title": "Fill in the input form",
             "detail": (
                 f"The form has **{len(inputs)} input fields**: "
                 + ("`" + "`, `".join(inputs) + "`" if inputs else "none detected") + ". "
                 "Enter realistic values — stay within the training data range for best accuracy."
             )},
            {"step": 3, "icon": "▶️", "title": "Click Predict",
             "detail": "Your inputs are sent to the AWS SageMaker endpoint. "
                       "Inference typically takes under 1 second."},
            {"step": 4, "icon": "📊", "title": "Read the result",
             "detail": f"The app returns {expected}. "
                       "This comes directly from the trained XGBoost model."},
            {"step": 5, "icon": "⚠️", "title": "Interpret responsibly",
             "detail": "ML models are probabilistic — not guaranteed to be correct. "
                       "Use the prediction as a guide, especially for edge-case inputs."},
        ]

    # ── data story ────────────────────────────────────────────────────────────
    def _build_data_story(self, dataset: dict, top_sources: list,
                           spec: dict) -> dict:
        sources_detail = []
        for s in top_sources[:3]:
            sources_detail.append({
                "name":   s.get("title", "Unknown"),
                "source": s.get("source", "kaggle").upper(),
                "url":    s.get("url", ""),
                "rows":   s.get("row_count", 0),
                "score":  round(float(s.get("final_score", 0)), 3),
            })
        return {
            "summary": (
                f"Collected **{dataset.get('row_count', 0):,} rows** × "
                f"**{len(dataset.get('columns', []))} columns** from "
                f"**{dataset.get('source_count', 1)} source(s)**."
                + (" Datasets were merged to reach the 500-row minimum threshold."
                   if dataset.get("merged") else "")
            ),
            "sources":         sources_detail,
            "columns":         dataset.get("columns", []),
            "row_count":       dataset.get("row_count", 0),
            "merged":          dataset.get("merged", False),
            "keywords_used":   spec.get("keywords", []),
            "search_strategy": (
                "The agent searched **Kaggle**, **UCI ML Repository**, and **OpenML** "
                "simultaneously. Candidates were scored on: keyword match (40%), "
                "row count (30%), column alignment with your input parameters (20%), "
                "and recency/popularity (10%). The top-scoring datasets were downloaded "
                "and merged if needed to meet the 500-row threshold."
            ),
        }

    # ── diagram ───────────────────────────────────────────────────────────────
    def _generate_diagram(self, task_type: str, job_result: dict) -> str:
        try:
            if self._debug:
                self._debug.debug(
                    "_generate_diagram",
                    f"Generating architecture diagram for {task_type} task"
                )
            
            result = _build_diagram_png(task_type, job_result, self._ws_dir)
            
            if self._debug:
                if result:
                    self._debug.info(
                        "_generate_diagram",
                        f"Diagram generated: {len(result)} chars"
                    )
                else:
                    self._debug.warning(
                        "_generate_diagram",
                        "Diagram generation returned empty (graphviz may be missing)"
                    )
            
            return result
            
        except FileNotFoundError as exc:
            if self._debug:
                self._debug.warning(
                    "_generate_diagram",
                    "Graphviz system tool not found - diagram skipped",
                    context={"exception": "FileNotFoundError"}
                )
            logger.warning("Graphviz not found (%s) — skipping diagram", exc)
            return ""
            
        except Exception as exc:
            if self._debug:
                self._debug.error(
                    "_generate_diagram",
                    f"Diagram generation failed: {str(exc)[:100]}",
                    exception=exc
                )
            logger.warning("Diagram generation failed (%s) — skipping", exc)
            return ""

    # ── code preview ──────────────────────────────────────────────────────────
    @staticmethod
    def _build_code_preview(written_files: dict | None) -> dict:
        if not written_files:
            return {}
        preview = {}
        for fname, path in written_files.items():
            try:
                p = Path(str(path))
                if not p.exists():
                    continue
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
                preview[fname] = "\n".join(lines[:60])
                if len(lines) > 60:
                    preview[fname] += f"\n\n# ... ({len(lines) - 60} more lines)"
            except Exception:
                pass
        return preview

    # ── fallback narrative ────────────────────────────────────────────────────
    @staticmethod
    def _fallback_narrative(db_results: dict, job_result: dict) -> str:
        spec    = db_results.get("spec", {})
        dataset = job_result.get("dataset", {})
        model   = job_result.get("model", {})
        inputs  = spec.get("input_params", [])
        target  = spec.get("target_param", "output")
        return textwrap.dedent(f"""
        ## Your RAD-ML Pipeline is Complete 🎉

        **What your model does:** Given {len(inputs)} input(s)
        ({', '.join(inputs[:5])}{'…' if len(inputs) > 5 else ''}),
        your model predicts **{target}**.

        **Dataset:** {dataset.get('row_count', 0):,} rows collected from
        {dataset.get('source_count', 1)} source(s).
        {'Datasets were merged to reach the 500-row threshold.' if dataset.get('merged') else ''}

        **Algorithm:** XGBoost ({model.get('task_type', 'regression')} mode)
        trained on AWS SageMaker with
        {model.get('stats', {}).get('train_rows', 0):,} training rows
        and {model.get('stats', {}).get('val_rows', 0):,} validation rows.

        **How to use:** Open the app link → fill the form → click Predict.
        The model returns a {target} estimate in under a second.
        """).strip()


# ── Diagram builder ───────────────────────────────────────────────────────────
def _build_diagram_png(task_type: str, job_result: dict, out_dir: Path) -> str:
    """Generate pipeline architecture diagram → base64 PNG."""
    try:
        from diagrams import Diagram, Cluster, Edge
        from diagrams.programming.flowchart import (
            StartEnd, Decision, Database, InputOutput, Merge,
        )
        from diagrams.generic.storage import Storage
        from diagrams.onprem.compute import Server
    except ImportError:
        logger.warning("diagrams library not installed — skipping diagram")
        return ""

    row_count    = job_result.get("dataset", {}).get("row_count", 0)
    task_label   = task_type.title()
    endpoint     = job_result.get("sm_meta", {}).get("endpoint_name", "SageMaker")
    out_path     = str(out_dir / "pipeline_architecture")

    graph_attr = {
        "fontname":  "Helvetica",
        "bgcolor":   "#0d0d1a",
        "fontcolor": "#e4e4f0",
        "pad":       "0.5",
        "splines":   "curved",
        "nodesep":   "0.4",
        "ranksep":   "0.6",
    }
    node_attr = {
        "fontname":  "Helvetica",
        "fontsize":  "11",
        "fontcolor": "#e4e4f0",
        "style":     "filled",
        "fillcolor": "#18182e",
        "color":     "#7c6dfa",
        "penwidth":  "1.5",
    }
    edge_attr = {
        "color":     "#7c6dfa88",
        "fontname":  "Helvetica",
        "fontsize":  "9",
        "fontcolor": "#9494bb",
        "penwidth":  "1.5",
    }

    try:
        with Diagram(
            "RAD-ML Pipeline Architecture",
            filename   = out_path,
            outformat  = "png",
            show       = False,
            graph_attr = graph_attr,
            node_attr  = node_attr,
            edge_attr  = edge_attr,
            direction  = "TB",
        ):
            user   = StartEnd("User\nPrompt")
            parser = Decision("Prompt Parser\nNLP · Intent\nExtraction")

            with Cluster("Data Collection  (Kaggle · UCI · OpenML)",
                          graph_attr={"bgcolor": "#0f1f20", "color": "#00e8c8",
                                      "fontcolor": "#00e8c8", "fontname": "Helvetica",
                                      "style": "rounded,filled"}):
                kg     = Database("Kaggle")
                uci    = Database("UCI ML\nRepository")
                oml    = Database("OpenML")
                scorer = Decision("Scorer\n4-factor\nWeighting")
                merger = Merge(f"Dataset Merger\n{row_count:,} rows")

            s3 = Storage("AWS S3\nDataset Store")

            with Cluster("Code Generator  (5 Layers)",
                          graph_attr={"bgcolor": "#1f0f1a", "color": "#f065ad",
                                      "fontcolor": "#f065ad", "fontname": "Helvetica",
                                      "style": "rounded,filled"}):
                l1 = Decision("Layer 1\nPrompt\nUnderstanding")
                l2 = Decision("Layer 2\nArchitect\nPlanner")
                l3 = InputOutput("Layer 3\nCode Generator\n(file-by-file)")
                l4 = Decision("Layer 4\nValidator\nAST · Tests")
                l5 = Merge("Layer 5\nRepair Loop\nLLM Fix")

            sm  = Server(f"SageMaker\n{task_label}")
            app = StartEnd(f"Flask App\n{endpoint[:20]}")
            out = StartEnd("User Gets\nPrediction")

            user   >> parser
            parser >> Edge(label="search") >> kg
            parser >> Edge(label="search") >> uci
            parser >> Edge(label="search") >> oml
            kg     >> scorer
            uci    >> scorer
            oml    >> scorer
            scorer >> merger
            merger >> Edge(label="final CSV") >> s3
            s3     >> l1
            l1     >> l2
            l2     >> l3
            l3     >> l4
            l4     >> Edge(label="errors?") >> l5
            l5     >> Edge(label="fixed") >> l4
            l4     >> Edge(label="✓ pass") >> sm
            sm     >> Edge(label="endpoint") >> app
            app    >> out

        png_path = Path(out_path + ".png")
        if not png_path.exists():
            return ""
        return base64.b64encode(png_path.read_bytes()).decode("ascii")
    except FileNotFoundError as e:
        logger.warning(f"graphviz system tool not found (do graphviz install manually): {e}")
        return ""
    except Exception as e:
        logger.warning(f"diagram generation failed: {e}")
        return ""
