"""
generator/algorithm_selector.py
================================
Intelligent algorithm selection based on task type.

Selects the best FREE/open-source algorithm for:
- Regression: LightGBM, XGBoost, Random Forest
- Classification: LightGBM, XGBoost, Random Forest
- Clustering: K-Means, DBSCAN, Hierarchical

Each algorithm includes:
- Why it's chosen
- Pros and cons
- Hyperparameters
- Performance metrics to track
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# ALGORITHM CONFIGURATIONS
# ────────────────────────────────────────────────────────────────────────────

_REGRESSION_ALGORITHMS = {
    "lightgbm": {
        "name": "LightGBM (Light Gradient Boosting)",
        "reason": "Fastest gradient boosting. Low memory. Best for large datasets.",
        "pros": ["Fast training", "Low memory", "Handles large data", "Built-in feature importance"],
        "cons": ["Overfitting risk", "Needs parameter tuning"],
        "package": "lightgbm",
        "hyperparams": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "random_state": 42,
        },
        "metrics": ["mse", "rmse", "mae", "r2_score", "mean_absolute_error"],
        "cross_validation": 5,
        "train_test_split": 0.2,
    },
    "xgboost": {
        "name": "XGBoost (Extreme Gradient Boosting)",
        "reason": "Most robust gradient boosting. Excellent generalization.",
        "pros": ["Highly accurate", "Robust", "Good regularization", "Parallel computation"],
        "cons": ["Slower than LightGBM", "More memory"],
        "package": "xgboost",
        "hyperparams": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        },
        "metrics": ["mse", "rmse", "mae", "r2_score"],
        "cross_validation": 5,
        "train_test_split": 0.2,
    },
    "random_forest": {
        "name": "Random Forest",
        "reason": "Simple, interpretable. Good for datasets < 100k rows.",
        "pros": ["Easy to use", "Interpretable", "Fast prediction", "Handles non-linear"],
        "cons": ["Slower training", "More memory than GB", "Less accurate than GB"],
        "package": "scikit-learn",
        "hyperparams": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
        },
        "metrics": ["mse", "rmse", "mae", "r2_score"],
        "cross_validation": 5,
        "train_test_split": 0.2,
    },
}

_CLASSIFICATION_ALGORITHMS = {
    "lightgbm": {
        "name": "LightGBM (Light Gradient Boosting)",
        "reason": "Fast classification. Best for large datasets.",
        "pros": ["Fast", "Low memory", "Handles imbalanced", "Feature importance"],
        "cons": ["Overfitting risk", "Needs tuning"],
        "package": "lightgbm",
        "hyperparams": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "random_state": 42,
        },
        "metrics": ["accuracy", "precision", "recall", "f1_score", "roc_auc", "confusion_matrix"],
        "cross_validation": 5,
        "train_test_split": 0.2,
    },
    "xgboost": {
        "name": "XGBoost (Extreme Gradient Boosting)",
        "reason": "Most accurate classification. Industry standard.",
        "pros": ["Highly accurate", "Handles imbalance", "Robust", "Good calibration"],
        "cons": ["Slower", "More memory"],
        "package": "xgboost",
        "hyperparams": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "scale_pos_weight": 1,  # Adjust for imbalanced classes
        },
        "metrics": ["accuracy", "precision", "recall", "f1_score", "roc_auc"],
        "cross_validation": 5,
        "train_test_split": 0.2,
    },
    "random_forest": {
        "name": "Random Forest",
        "reason": "Simple, interpretable classification.",
        "pros": ["Easy", "Interpretable", "Fast inference", "Inherent feature ranking"],
        "cons": ["Slower training", "More memory", "Less accurate"],
        "package": "scikit-learn",
        "hyperparams": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
        },
        "metrics": ["accuracy", "precision", "recall", "f1_score"],
        "cross_validation": 5,
        "train_test_split": 0.2,
    },
}

_CLUSTERING_ALGORITHMS = {
    "kmeans": {
        "name": "K-Means",
        "reason": "Fast clustering. Works for most datasets.",
        "pros": ["Fast", "Scalable", "Simple to understand", "Works well"],
        "cons": ["Need to specify k", "Sensitive to initialization"],
        "package": "scikit-learn",
        "hyperparams": {
            "n_clusters": 3,  # May be adjusted based on data
            "init": "k-means++",
            "random_state": 42,
            "n_init": 10,
        },
        "metrics": ["silhouette_score", "davies_bouldin_score", "calinski_harabasz_score", "inertia"],
        "cross_validation": None,  # Clustering doesn't use CV
        "train_test_split": None,
    },
    "dbscan": {
        "name": "DBSCAN",
        "reason": "Finds arbitrarily shaped clusters. No need to specify k.",
        "pros": ["No k needed", "Finds arbitrary shapes", "Outlier detection", "Density-based"],
        "cons": ["Parameter tuning hard", "Slow on large data"],
        "package": "scikit-learn",
        "hyperparams": {
            "eps": 0.5,  # Distance threshold
            "min_samples": 5,  # Min points to form cluster
        },
        "metrics": ["silhouette_score", "davies_bouldin_score"],
        "cross_validation": None,
        "train_test_split": None,
    },
    "hierarchical": {
        "name": "Hierarchical Clustering",
        "reason": "Produces dendrogram. Good for exploratory analysis.",
        "pros": ["Dendrogram visualization", "No k needed upfront", "Multiple linkage options"],
        "cons": ["Slow on large data", "Memory intensive"],
        "package": "scikit-learn",
        "hyperparams": {
            "n_clusters": 3,
            "linkage": "ward",  # Can also be complete, average, single
        },
        "metrics": ["silhouette_score", "davies_bouldin_score"],
        "cross_validation": None,
        "train_test_split": None,
    },
}


class AlgorithmSelector:
    """Intelligently select best algorithm based on task type."""

    @staticmethod
    def select(task_type: str, dataset_size: int | None = None) -> dict:
        """
        Select best algorithm for task type.

        Parameters
        ----------
        task_type : str
            One of: "regression", "classification", "clustering"
        dataset_size : int, optional
            Number of rows in dataset. Helps optimize choice.

        Returns
        -------
        dict with keys:
            - name: Algorithm name
            - reason: Why this algorithm
            - config: Algorithm configuration and hyperparameters
            - metrics: Performance metrics to track
        """
        task_type = task_type.lower().strip()

        if task_type == "regression":
            # For regression: LightGBM is fastest, XGBoost most accurate
            # Choose LightGBM for large datasets, XGBoost for smaller
            if dataset_size and dataset_size > 100000:
                algo_name = "lightgbm"
                reason = f"Dataset has {dataset_size:,} rows → LightGBM (fastest)"
            else:
                algo_name = "xgboost"
                reason = f"Dataset size optimal for XGBoost (most accurate)"

            config = _REGRESSION_ALGORITHMS[algo_name]
            return {
                "task_type": "regression",
                "algorithm": algo_name,
                "name": config["name"],
                "reason": reason,
                "pros": config["pros"],
                "cons": config["cons"],
                "package": config["package"],
                "hyperparams": config["hyperparams"],
                "metrics": config["metrics"],
                "cross_validation_folds": config["cross_validation"],
                "train_test_split": config["train_test_split"],
                "all_options": list(_REGRESSION_ALGORITHMS.keys()),
            }

        elif task_type == "classification":
            # Similar logic for classification
            if dataset_size and dataset_size > 100000:
                algo_name = "lightgbm"
                reason = f"Dataset has {dataset_size:,} rows → LightGBM (fastest)"
            else:
                algo_name = "xgboost"
                reason = f"Dataset size optimal for XGBoost (most accurate)"

            config = _CLASSIFICATION_ALGORITHMS[algo_name]
            return {
                "task_type": "classification",
                "algorithm": algo_name,
                "name": config["name"],
                "reason": reason,
                "pros": config["pros"],
                "cons": config["cons"],
                "package": config["package"],
                "hyperparams": config["hyperparams"],
                "metrics": config["metrics"],
                "cross_validation_folds": config["cross_validation"],
                "train_test_split": config["train_test_split"],
                "all_options": list(_CLASSIFICATION_ALGORITHMS.keys()),
            }

        elif task_type == "clustering":
            # For clustering: K-Means is fast and robust
            algo_name = "kmeans"
            reason = "K-Means is fast, robust, and works universally"

            config = _CLUSTERING_ALGORITHMS[algo_name]
            return {
                "task_type": "clustering",
                "algorithm": algo_name,
                "name": config["name"],
                "reason": reason,
                "pros": config["pros"],
                "cons": config["cons"],
                "package": config["package"],
                "hyperparams": config["hyperparams"],
                "metrics": config["metrics"],
                "cross_validation_folds": config["cross_validation"],
                "train_test_split": config["train_test_split"],
                "all_options": list(_CLUSTERING_ALGORITHMS.keys()),
            }

        else:
            raise ValueError(
                f"Unknown task_type: {task_type}. "
                f"Must be one of: regression, classification, clustering"
            )

    @staticmethod
    def get_algorithm_code(algo_selection: dict) -> str:
        """
        Generate Python code snippet for the selected algorithm.

        Parameters
        ----------
        algo_selection : dict
            Result from select()

        Returns
        -------
        Python code string that trains the model
        """
        task_type = algo_selection["task_type"]
        algo = algo_selection["algorithm"]
        name = algo_selection["name"]
        hyperparams = algo_selection["hyperparams"]

        if task_type == "regression":
            if algo == "lightgbm":
                return f"""\
# {name}
import lightgbm as lgb

# Initialize model
model = lgb.LGBMRegressor(
    {_format_params(hyperparams)}
)

# Train model
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10,
    verbose=10,
)

# Predictions
y_pred = model.predict(X_test)
"""
            elif algo == "xgboost":
                return f"""\
# {name}
import xgboost as xgb

# Initialize model
model = xgb.XGBRegressor(
    {_format_params(hyperparams)}
)

# Train model
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10,
    verbose=True,
)

# Predictions
y_pred = model.predict(X_test)
"""
            elif algo == "random_forest":
                return f"""\
# {name}
from sklearn.ensemble import RandomForestRegressor

# Initialize model
model = RandomForestRegressor(
    {_format_params(hyperparams)}
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
"""

        elif task_type == "classification":
            if algo == "lightgbm":
                return f"""\
# {name}
import lightgbm as lgb

# Initialize model
model = lgb.LGBMClassifier(
    {_format_params(hyperparams)}
)

# Train model
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10,
    verbose=10,
)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
"""
            elif algo == "xgboost":
                return f"""\
# {name}
import xgboost as xgb

# Initialize model
model = xgb.XGBClassifier(
    {_format_params(hyperparams)}
)

# Train model
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10,
    verbose=True,
)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
"""
            elif algo == "random_forest":
                return f"""\
# {name}
from sklearn.ensemble import RandomForestClassifier

# Initialize model
model = RandomForestClassifier(
    {_format_params(hyperparams)}
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
"""

        elif task_type == "clustering":
            if algo == "kmeans":
                return f"""\
# {name}
from sklearn.cluster import KMeans

# Initialize model
model = KMeans(
    {_format_params(hyperparams)}
)

# Fit model
model.fit(X)

# Get cluster assignments
labels = model.labels_
centers = model.cluster_centers_
"""
            elif algo == "dbscan":
                return f"""\
# {name}
from sklearn.cluster import DBSCAN

# Initialize model
model = DBSCAN(
    {_format_params(hyperparams)}
)

# Fit model
labels = model.fit_predict(X)

# Get cluster counts
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
"""
            elif algo == "hierarchical":
                return f"""\
# {name}
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

# Compute linkage
Z = linkage(X, method='{hyperparams.get("linkage", "ward")}')

# Get cluster assignments
labels = fcluster(Z, {hyperparams.get("n_clusters", 3)}, criterion='maxclust')
"""

        return "# Algorithm code generation failed"


def _format_params(params: dict) -> str:
    """Format hyperparameters for code generation."""
    lines = []
    for key, value in params.items():
        if isinstance(value, str):
            lines.append(f"    {key}='{value}',")
        else:
            lines.append(f"    {key}={value},")
    return "\n".join(lines)
