import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple


def render_model_report(
    pipeline,
    model_name: str,
    show_runs: Tuple[str, ...] = ('no_smote', 'smote'),
    save_dir: str = "models"
) -> Dict[str, Any]:
    """
    Print results for `model_name` and the runs specified in `show_runs`.
    Reports only CV AUC (from GridSearchCV/RandomizedSearchCV).
    """
    if model_name not in getattr(pipeline, "models", {}):
        raise KeyError(f"No runs stored for model '{model_name}' in pipeline.models")

    os.makedirs(save_dir, exist_ok=True)

    runs_data = {}

    print(f"\n==== Results for model: {model_name} ====\n")
    for run_key in show_runs:
        if run_key not in pipeline.models[model_name]:
            print(f"Run '{run_key}' not found — skipping.")
            continue

        run = pipeline.models[model_name][run_key]
        cv_score = run.get('cv_score')
        best_params = run.get('best_params', None)

        print(f"--- {run_key} ---")
        print(f"CV AUC: {cv_score:.4f}")
        print(f"Best params: {best_params}\n")

        runs_data[run_key] = {
            'cv_score': cv_score,
            'best_params': best_params
        }

    return {"model": model_name, "runs": runs_data}


def plot_aggregated_rocs(
    pipeline,
    save_dir: str = "models"
) -> None:
    """
    Aggregated barplot of CV AUC across all models and runs.
    """
    rows = []
    for model_name, runs in pipeline.models.items():
        for run_key, run in runs.items():
            cv_score = run.get("cv_score", None)
            if cv_score is None:
                continue
            rows.append({
                "model": model_name,
                "run": run_key,
                "cv_auc": cv_score
            })

    if not rows:
        print("No runs available for plotting.")
        return

    df = pd.DataFrame(rows)

    # pivot table for better visualization
    pivot = df.pivot(index="model", columns="run", values="cv_auc")

    pivot.plot(kind="bar", figsize=(10, 6))
    plt.ylabel("CV AUC")
    plt.title("Cross-validated AUC by Model and Run")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    savepath = os.path.join(save_dir, "aggregated_auc.png")
    plt.savefig(savepath, dpi=150)
    plt.show()
    print(f"Saved aggregated plot to: {savepath}")


def save_summary_csv(
    pipeline,
    save_path: str = "models/summary.csv"
) -> None:
    """
    Save a CSV summary with only CV AUC and best params.
    """
    rows = []
    for model_name, runs in pipeline.models.items():
        for run_key, run in runs.items():
            rows.append({
                "model": model_name,
                "run": run_key,
                "cv_auc": run.get("cv_score"),
                "best_params": run.get("best_params")
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved summary CSV to: {save_path}")
