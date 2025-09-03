import os
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, confusion_matrix, classification_report

# helper to extract probability-like scores from an estimator
def _get_proba_or_score(estimator, X) -> Optional[np.ndarray]:
    """Return 1D array of scores in [0,1] usable for ROC, or None if not available."""
    if estimator is None:
        return None
    try:
        probs = estimator.predict_proba(X)[:, 1]
        return probs
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
    X_val,
    y_val,
    show_runs: Tuple[str, ...] = ('no_smote', 'smote'),
    save_dir: str = "models",
    save_comparison_roc: bool = True
) -> Dict[str, Any]:
    """
    Print results for `model_name` and the runs specified in `show_runs`.
    For each available run it prints CV/val AUC, fit time (if present), best params,
    draws per-run ROC and classification report, and returns a dictionary with results.
    Optionally saves a comparison ROC image with both runs.
    """
    if model_name not in getattr(pipeline, "models", {}):
        raise KeyError(f"No runs stored for model '{model_name}' in pipeline.models")

    os.makedirs(save_dir, exist_ok=True)

    runs_data = {}
    roc_entries = []

    print(f"\n==== Results for model: {model_name} ====\n")
    for run_key in show_runs:
        if run_key not in pipeline.models[model_name]:
            print(f"Run '{run_key}' not found — skipping.")
            continue

        run = pipeline.models[model_name][run_key]
        est = run.get('best_estimator')
        cv_score = run.get('cv_score')
        val_score = run.get('val_score')
        fit_time = run.get('fit_time_sec', None)
        best_params = run.get('best_params', None)

        print(f"--- {run_key} ---")
        print(f"CV AUC: {cv_score}")
        print(f"Validation AUC: {val_score}")
        if fit_time is not None:
            print(f"Fit time (s): {fit_time}")
        print(f"Best params: {best_params}")

        # compute score/proba for ROC and classification report
        proba = _get_proba_or_score(est, X_val)
        if proba is not None:
            try:
                auc = roc_auc_score(y_val, proba)
                fpr, tpr, _ = roc_curve(y_val, proba)
                # plot per-run ROC
                fig, ax = plt.subplots(figsize=(6, 4))
                RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax)
                ax.set_title(f"ROC — {model_name} ({run_key})  AUC={auc:.4f}")
                plt.show()

                # classification report with threshold 0.5
                y_pred = (proba >= 0.5).astype(int)
                print("Confusion matrix:")
                print(confusion_matrix(y_val, y_pred))
                print("\nClassification report:")
                print(classification_report(y_val, y_pred, digits=4))

                roc_entries.append({'run': run_key, 'fpr': fpr, 'tpr': tpr, 'auc': auc})
            except Exception as e:
                print("Failed to compute ROC/metrics for this run:", e)
        else:
            print("predict_proba/decision_function not available for this estimator; skipping ROC/AUC/CR.")

        # try to show feature importances via pipeline.feature_importance (best-effort)
        try:
            fi_df = pipeline.feature_importance(model_name, use_smote=(run_key == 'smote'), top_n=15)
            if fi_df is not None:
                print("\nTop feature importances:")
                display_df = fi_df.reset_index(drop=True)
                print(display_df.to_string(index=False))
        except Exception:
            # non-fatal: some estimators (Bagging with non-tree base) won't expose importances
            pass

        runs_data[run_key] = {
            'cv_score': cv_score,
            'val_score': val_score,
            'fit_time_sec': fit_time,
            'best_params': best_params
        }
        print("\n")

    # if requested, plot and save comparison ROC for this model
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


def _collect_all_roc_entries(pipeline) -> List[Dict[str, Any]]:
    """
    Collect ROC data for all stored runs in pipeline.models.
    Returns list of entries: {'model','run','fpr','tpr','auc'} for each run that has scores.
    """
    entries = []
    for model_name, runs in getattr(pipeline, "models", {}).items():
        for run_key, run in runs.items():
            est = run.get('best_estimator')
            entries.append({'model': model_name, 'run': run_key, 'estimator': est})
    return entries


def plot_aggregated_rocs(
    pipeline,
    X_val,
    y_val,
    save_dir: str = "models",
    show_smote: bool = True,
    show_no_smote: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    save_png: bool = True
) -> Dict[str, str]:
    """
    Plot two aggregated ROC figures:
     - one containing ROC curves for all SMOTE runs (if any)
     - one containing ROC curves for all no-SMOTE runs (if any)

    Returns a dict with paths to saved images (if saved), keys: 'smote', 'no_smote'.
    """
    os.makedirs(save_dir, exist_ok=True)
    smote_entries = []
    nosmote_entries = []

    for model_name, runs in getattr(pipeline, "models", {}).items():
        for run_key, run in runs.items():
            est = run.get('best_estimator')
            proba = _get_proba_or_score(est, X_val)
            if proba is None:
                # skip runs without score/proba
                continue
            try:
                auc = roc_auc_score(y_val, proba)
                fpr, tpr, _ = roc_curve(y_val, proba)
                entry = {'model': model_name, 'run': run_key, 'fpr': fpr, 'tpr': tpr, 'auc': auc}
                if run_key == 'smote':
                    smote_entries.append(entry)
                else:
                    nosmote_entries.append(entry)
            except Exception:
                continue

    saved_paths = {}
    def _plot_and_save(entries, title, fname):
        if len(entries) == 0:
            print(f"No entries to plot for: {title}")
            return None
        plt.figure(figsize=figsize)
        for rc in entries:
            label = f"{rc['model']} AUC={rc['auc']:.3f}"
            plt.plot(rc['fpr'], rc['tpr'], label=label, linewidth=1.8)
        plt.plot([0, 1], [0, 1], linestyle='--', linewidth=0.8, color='gray')
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
        saved_paths['smote'] = _plot_and_save(smote_entries, "Aggregated ROC — SMOTE (all models)", "roc_aggregated_smote.png")
    if show_no_smote:
        saved_paths['no_smote'] = _plot_and_save(nosmote_entries, "Aggregated ROC — no-SMOTE (all models)", "roc_aggregated_no_smote.png")

    return saved_paths


def save_summary_csv(pipeline, path: str = "models/training_summary.csv") -> pd.DataFrame:
    """
    Save a summary table of all runs stored in pipeline.models to CSV and return the DataFrame.
    Columns: model, run, use_smote, cv_score, val_score, best_params, fit_time_sec
    """
    rows = []
    for model_name, runs in getattr(pipeline, "models", {}).items():
        for run_key, run in runs.items():
            rows.append({
                'model': model_name,
                'run': run_key,
                'use_smote': run.get('metadata', {}).get('use_smote'),
                'cv_score': run.get('cv_score'),
                'val_score': run.get('val_score'),
                'best_params': run.get('best_params'),
                'fit_time_sec': run.get('fit_time_sec')
            })
    df = pd.DataFrame(rows).sort_values(['model', 'run']).reset_index(drop=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved summary CSV to: {path}")
    return df
