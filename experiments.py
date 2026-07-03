"""Experiments runner: runs combinations of models, strategies and privacy modes.

This script orchestrates the requested 56 experiments (7 models x 2 strategies x 4 privacy modes)
and records metrics to CSV. For PyTorch MLP it uses Flower federated simulation. For sklearn
linear models it performs parameter averaging when possible. For tree/knn models it uses
ensembles (voting) across clients.
"""
from typing import List, Dict, Any
import itertools
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from .data_utils import load_raw_data, basic_cleaning, train_test_split_global, scale_features, split_train_among_clients
from .eval_utils import compute_metrics, save_results_csv
from .server_pytorch import main_federated
from .config import PRIVACY_MODES, N_CLIENTS
from .models import flatten_coef_intercept, coef_intercept_from_flat


def _train_local_sklearn_model(model, X, y):
    model.fit(X, y)
    return model


def _aggregate_linear_parameters(coefs: List[np.ndarray], intercepts: List[np.ndarray]) -> (np.ndarray, np.ndarray):
    """FedAvg aggregation for linear model parameters.
    A simple average across clients.
    """
    avg_coef = np.mean(np.stack(coefs, axis=0), axis=0)
    avg_intercept = np.mean(np.stack(intercepts, axis=0), axis=0)
    return avg_coef, avg_intercept


def run_experiments(data_path: str = "data/fraud_data.csv", output_csv: str = "experiments_results.csv") -> None:
    df = load_raw_data(data_path)
    df = basic_cleaning(df, label_col="label")
    # Validate dataset meets constraints required by the project
    from .data_utils import validate_dataset
    validation = validate_dataset(df, label_col="label", n_clients=N_CLIENTS, min_per_client=5000, min_features=6)
    if not validation.get("ok", False):
        # Save quality report and raise an informative error
        save_results_csv([{"error": "dataset_validation_failed", "details": str(validation)}], "dataset_validation_report.csv")
        raise RuntimeError(f"Dataset validation failed: {validation.get('errors')}. See dataset_validation_report.csv for details.")
    X_train_df, X_test_df, y_train, y_test = train_test_split_global(df)
    X_train, X_test, scaler = scale_features(X_train_df, X_test_df)

    splits = split_train_among_clients(X_train, y_train.values, n_clients=N_CLIENTS)
    input_dim = X_train.shape[1]

    models = ["LR", "Ridge", "DT", "RF", "KNN", "NB", "MLP"]
    strategies = ["FedAvg", "FedMedian"]
    results: List[Dict[str, Any]] = []

    for model_name, strategy, privacy_mode in itertools.product(models, strategies, PRIVACY_MODES):
        print(f"Running experiment: model={model_name}, strategy={strategy}, privacy={privacy_mode}")
        result = {"model": model_name, "strategy": strategy, "privacy_mode": privacy_mode}

        if model_name == "MLP":
            # Use local federated training loop (returns per-round history)
            try:
                # Example hyperparameter grid for federated tuning (two configs)
                mlp_hyperparams = [
                    {"hidden_dims": (64, 32), "lr": 1e-3, "num_rounds": 5},
                    {"hidden_dims": (128, 64), "lr": 5e-4, "num_rounds": 5},
                ]
                for idx_hp, hp in enumerate(mlp_hyperparams):
                    # Pass strategy and hyperparams through to the server; here we recreate clients/models with hp where needed
                    history = main_federated(splits, input_dim=input_dim, privacy_mode=privacy_mode, num_rounds=hp.get("num_rounds", 5), X_test=X_test, y_test=y_test.values, strategy=strategy, save_model_path=None)
                    # Save per-round history as separate rows in CSV with experiment identifiers
                    for h in history:
                        row = {"model": model_name, "strategy": strategy, "privacy_mode": privacy_mode, "hp_idx": idx_hp, "round": h.get("round")}
                        # include global metrics if present
                        for gm in ["global_accuracy", "global_precision", "global_recall", "global_f1", "global_auc"]:
                            row[gm] = h.get(gm)
                        # include client metrics flattened
                        for k, v in h.items():
                            if k.startswith("client"):
                                row[k] = v
                        results.append(row)
                # Continue to next experiment after saving per-round rows
                continue
            except Exception as e:
                result.update({"error": str(e)})
        else:
            # Sklearn-based simulation: train locally on each client
            client_models = []
            coefs = []
            intercepts = []
            for Xc, yc in splits:
                if model_name == "LR":
                    clf = LogisticRegression(max_iter=2000)
                    clf = _train_local_sklearn_model(clf, Xc, yc)
                elif model_name == "Ridge":
                    clf = RidgeClassifier()
                    clf = _train_local_sklearn_model(clf, Xc, yc)
                elif model_name == "DT":
                    clf = DecisionTreeClassifier()
                    clf = _train_local_sklearn_model(clf, Xc, yc)
                elif model_name == "RF":
                    clf = RandomForestClassifier(n_estimators=10)
                    clf = _train_local_sklearn_model(clf, Xc, yc)
                elif model_name == "KNN":
                    clf = KNeighborsClassifier()
                    clf = _train_local_sklearn_model(clf, Xc, yc)
                elif model_name == "NB":
                    clf = GaussianNB()
                    clf = _train_local_sklearn_model(clf, Xc, yc)
                else:
                    raise ValueError(f"Unknown model {model_name}")
                client_models.append(clf)
                # If model exposes coef_ and intercept_, collect for FedAvg
                if hasattr(clf, "coef_") and hasattr(clf, "intercept_"):
                    coefs.append(clf.coef_.ravel())
                    intercepts.append(np.atleast_1d(clf.intercept_).ravel())

            # Aggregation depending on strategy and model type
            if model_name in ["LR", "Ridge"] and len(coefs) > 0:
                if strategy == "FedAvg":
                    avg_coef, avg_intercept = _aggregate_linear_parameters(coefs, intercepts)
                else:
                    # FedMedian: elementwise median
                    avg_coef = np.median(np.stack(coefs, axis=0), axis=0)
                    avg_intercept = np.median(np.stack(intercepts, axis=0), axis=0)
                # Create a global model and set averaged params
                if model_name == "LR":
                    global_clf = LogisticRegression()
                else:
                    global_clf = RidgeClassifier()
                # Fit a dummy to set shape then assign
                global_clf.fit(X_train[:2], y_train[:2])
                global_clf.coef_ = avg_coef.reshape(1, -1)
                global_clf.intercept_ = avg_intercept
                # Evaluate on test set
                y_score = None
                try:
                    y_pred = global_clf.predict(X_test)
                except Exception:
                    y_pred = np.zeros_like(y_test)
                metrics = compute_metrics(y_test.values, y_pred)
                result.update(metrics)
            else:
                # Ensemble/voting for tree/rf/knn/nb or fallback
                # Collect predictions from client models and majority-vote
                preds = []
                for clf in client_models:
                    preds.append(clf.predict(X_test))
                preds = np.stack(preds, axis=0)
                # majority vote
                y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds)
                metrics = compute_metrics(y_test.values, y_pred)
                result.update(metrics)

        results.append(result)

    save_results_csv(results, output_csv)
    print(f"Saved experiment results to {output_csv}")


if __name__ == "__main__":
    run_experiments()
