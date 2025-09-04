import os
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt


def render_model_report(
    pipeline,
    model_name: str,
    show_runs: Tuple[str, ...] = ('no_smote', 'smote'),
    print_output: bool = True
) -> Dict[str, Any]:
    """
    Print and return CV-only report for a single model.
    Expects pipeline.models[model_name][run_key]['cv_score'] and 'best_params'.
    """
    if model_name not in getattr(pipeline, "models", {}):
        raise KeyError(f"No runs stored for model '{model_name}' in pipeline.models")

    runs_data = {}
    if print_output:
        print(f"\n==== Results for model: {model_name} ====\n")

    for run_key in show_runs:
        if run_key not in pipeline.models[model_name]:
            if print_output:
                print(f"Run '{run_key}' not found — skipping.")
            continue

        run = pipeline.models[model_name][run_key]
        cv_score = run.get('cv_score')
        best_params = run.get('best_params')

        if print_output:
            print(f"--- {run_key} ---")
            print(f"CV AUC: {cv_score:.4f}" if cv_score is not None else "CV AUC: None")
            print(f"Best params: {best_params}\n")

        runs_data[run_key] = {'cv_score': cv_score, 'best_params': best_params}

    return {"model": model_name, "runs": runs_data}


def collect_summary_df(pipeline) -> pd.DataFrame:
    """
    Build a DataFrame with columns: model, run, cv_auc, best_params, plus metadata keys if present.
    """
    rows = []
    for model_name, runs in getattr(pipeline, "models", {}).items():
        for run_key, run in runs.items():
            row = {
                "model": model_name,
                "run": run_key,
                "cv_auc": run.get('cv_score'),
                "best_params": run.get('best_params')
            }
            # include metadata keys (sample_frac, speed_mode, use_smote) if exist
            meta = run.get('metadata', {})
            for mk, mv in meta.items():
                # avoid overwriting core columns
                if mk not in row:
                    row[mk] = mv
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # ensure consistent dtypes
    df = df.sort_values(['model', 'run']).reset_index(drop=True)
    return df


def render_all_models_report(
    pipeline,
    sort_by: str = 'cv_auc',
    ascending: bool = False,
    top_k: Optional[int] = None,
    print_table: bool = True
) -> pd.DataFrame:
    """
    Print and return a summary table for all models. By default sorts by cv_auc descending.
    Use top_k to show only top N rows.
    """
    df = collect_summary_df(pipeline)
    if df.empty:
        print("No runs stored in pipeline.models")
        return df

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    if top_k is not None:
        df = df.head(top_k)

    if print_table:
        pd.set_option('display.max_columns', None)
        print("\n===== All models summary =====\n")
        print(df.to_string(index=False))
        print("\n")

    return df


def plot_cv_auc_bar(
    pipeline,
    save_dir: str = "models",
    figsize: Tuple[int, int] = (10, 6),
    stacked: bool = False,
    show: bool = True
) -> Optional[str]:
    """
    Create a bar plot of CV AUC for each model/run.
    If both 'no_smote' and 'smote' present, they will appear as separate bars (grouped).
    """
    df = collect_summary_df(pipeline)
    if df.empty:
        print("No runs to plot.")
        return None

    # pivot so index=model, columns=run, values=cv_auc
    pivot = df.pivot(index='model', columns='run', values='cv_auc').fillna(0)

    plt.figure(figsize=figsize)
    pivot.plot(kind='bar', stacked=stacked, figsize=figsize)
    plt.ylabel('CV AUC')
    plt.title('Cross-validated AUC by Model and Run')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.0, 1.0)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'cv_auc_by_model.png')
    plt.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f"Saved CV AUC barplot to: {path}")
    return path


def save_summary_csv(pipeline, save_path: str = "models/summary.csv") -> str:
    """
    Save the summary DataFrame to CSV and return path.
    """
    df = collect_summary_df(pipeline)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved summary CSV to: {save_path}")
    return save_path
