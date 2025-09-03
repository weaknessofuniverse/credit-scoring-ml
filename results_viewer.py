import os
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, confusion_matrix, classification_report


def _get_proba_or_score(estimator, X) -> Optional[np.ndarray]:
    """Return 1D array of scores in [0,1] usable for ROC, or None if not available."""
    if estimator is None:
        return None
    try:
        return estimator.predict_proba(X)[:, 1]
    except Exception:
        try:
            scores = estimator.decision_function(X)
            if np.ptp(scores) == 0:
                return None
            return (scores - scores.min()) / (scores.max() - scores.min())
        except Exception:
            return None


def render_model_report(
    pipeline,
    model_name: str,
    X_train,
    y_train,
    show_runs: Tuple[str, ...] = ('no_smote', 'smote'),
    save_dir: str = "models",
    save_comparison_roc: bool = True
) -> Dict[str, Any]:
    """
    Print results for `model_name` and the runs specified in `show_runs`.
    Uses only train set (CV + train AUC).
    """
    if model_name not in getattr(pipeline, "models", {}):
        raise KeyError(f"No runs stored for model '{model_name}' in pipeline.models")

    os.makedirs(save_dir, exist_ok=True)
    runs_data, roc_entries = {}, []

    print(f"\n==== Results for model: {model_name} ====\n")
    for run_key in show_runs:
        if run_key not in pipeline.models[model_name]:
            continue

        run = pipeline.models[model_name][run_key]
        est = run.get('best_estimator')
        cv_score = run.get('cv_score')
        best_params = run.get('best_params', None)

        print(f"--- {run_key} ---")
        print(f"CV AUC: {cv_score}")
        print(f"Best params: {best_params}")

        proba = _get_proba_or_score(est, X_train)
        if proba is not None:
            auc = roc_auc_score(y_train, proba)
            fpr, tpr, _ = roc_curve(y_train, proba)
            RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
            plt.title(f"ROC — {model_name} ({run_key})  AUC={auc:.4f}")
            plt.show()

            y_pred = (proba >= 0.5).astype(int)
            print("Confusion matrix:")
            print(confusion_matrix(y_train, y_pred))
            print("\nClassification report:")
            print(classification_report(y_train, y_pred, digits=4))

            roc_entries.append({'run': run_key, 'fpr': fpr, 'tpr': tpr, 'auc': auc})

        runs_data[run_key] = {'cv_score': cv_score, 'best_params': best_params}
        print("\n")

    if save_comparison_roc and len(roc_entries) > 0:
        plt.figure(figsize=(8, 6))
        for rc in roc_entries:
            linestyle = '-' if rc['run'] == 'smote' else '--'
            label = f"{rc['run']} AUC={rc['auc']:.3f}"
            plt.plot(rc['fpr'], rc['tpr'], label=label, linestyle=linestyle, linewidth=2)
        plt.plot([0, 1], [0, 1], linestyle=':', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Comparison ROC — {model_name}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        savepath = os.path.join(save_dir, f"{model_name}_comparison_roc.png")
        plt.savefig(savepath, dpi=150)
        plt.show()
        print(f"Saved comparison ROC to: {savepath}")

    return {"model": model_name, "runs": runs_data, "roc_entries": roc_entries}


def _collect_all_roc_entries(pipeline, X_train, y_train) -> List[Dict[str, Any]]:
    """Collect ROC data for all stored runs on train set."""
    entries = []
    for model_name, runs in getattr(pipeline, "models", {}).items():
        for run_key, run in runs.items():
            est = run.get('best_estimator')
            proba = _get_proba_or_score(est, X_train)
            if proba is None:
                continue
            auc = roc_auc_score(y_train, proba)
            fpr, tpr, _ = roc_curve(y_train, proba)
            entries.append({'model': model_name, 'run': run_key, 'fpr': fpr, 'tpr': tpr, 'auc': auc})
    return entries


def plot_aggregated_rocs(
    pipeline,
    X_train,
    y_train,
    save_dir: str = "models",
    show_smote: bool = True,
    show_no_smote: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    save_png: bool = True
) -> Dict[str, str]:
    """Plot aggregated ROC curves across all models on train set."""
    os.makedirs(save_dir, exist_ok=True)
    smote_entries, nosmote_entries = [], []

    for model_name, runs in getattr(pipeline, "models", {}).items():
        for run_key, run in runs.items():
            est = run.get('best_estimator')
            proba = _get_proba_or_score(est, X_train)
            if proba is None:
                continue
            auc = roc_auc_score(y_train, proba)
            fpr, tpr, _ = roc_curve(y_train, proba)
            entry = {'model': model_name, 'run': run_key, 'fpr': fpr, 'tpr': tpr, 'auc': auc}
            (smote_entries if run_key == 'smote' else nosmote_entries).append(entry)

    saved_paths = {}

    def _plot_and_save(entries, title, fname):
        if not entries:
            return None
        plt.figure(figsize=figsize)
        for rc in entries:
            label = f"{rc['model']} AUC={rc['auc']:.3f}"
            plt.plot(rc['fpr'], rc['tpr'], label=label, linewidth=1.8)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right', fontsize='small')
        plt.tight_layout()
        path = os.path.join(save_dir, fname)
        if save_png:
            plt.savefig(path, dpi=150)
            print(f"Saved aggregated ROC to: {path}")
        plt.show()
        return path

    if show_smote:
        saved_paths['smote'] = _plot_and_save(smote_entries, "Aggregated ROC — SMOTE (train)", "roc_aggregated_smote.png")
    if show_no_smote:
        saved_paths['no_smote'] = _plot_and_save(nosmote_entries, "Aggregated ROC — no-SMOTE (train)", "roc_aggregated_no_smote.png")

    return saved_paths


def save_summary_csv(pipeline, path: str = "models/training_summary.csv") -> pd.DataFrame:
    """Save summary table of all runs (train only)."""
    rows = []
    for model_name, runs in getattr(pipeline, "models", {}).items():
        for run_key, run in runs.items():
            rows.append({
                'model': model_name,
                'run': run_key,
                'use_smote': run.get('metadata', {}).get('use_smote'),
                'cv_score': run.get('cv_score'),
                'best_params': run.get('best_params'),
                'fit_time_sec': run.get('fit_time_sec')
            })
    df = pd.DataFrame(rows).sort_values(['model', 'run']).reset_index(drop=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved summary CSV to: {path}")
    return df
