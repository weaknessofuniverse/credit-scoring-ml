import numpy as np
import pandas as pd
from typing import Iterable, Tuple, Optional, Dict, Any
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
import math

warnings.filterwarnings("ignore")


class CreditScoringPipeline:
    def __init__(
        self,
        random_state: int = 42,
        default_search_method: str = "random",  # 'random' or 'grid'
        default_cv: int = 3,
        default_n_jobs: int = -1,
        speed_mode: str = "balanced",  # 'fast' | 'balanced' | 'thorough'
        run_both_smote_by_default: bool = False
    ):
        """
        Pipeline with speed presets. Use speed_mode to control how wide searches are:
          - fast:    very small search grids / n_iter (quick exploration)
          - balanced: middle ground
          - thorough: larger n_iter / full grids
        """
        self.models: Dict[str, Dict[str, Any]] = {}
        self.feature_names = None
        self.random_state = random_state
        self.default_search_method = default_search_method
        self.default_cv = default_cv
        self.default_n_jobs = default_n_jobs
        self.speed_mode = speed_mode
        self.run_both_smote_by_default = run_both_smote_by_default

        self.n_iter_defaults = {
            'LogisticRegression': 12,
            'RandomForest': 20,
            'GradientBoosting': 20,
            'HistGradientBoosting': 20,
            'SVM': 12,
            'Bagging': 12,
            'KNN': 12
        }

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Return model instances + readable param grids (kept compact)."""
        return {
            'LogisticRegression': {
                'model': LogisticRegression(random_state=self.random_state, class_weight='balanced', max_iter=2000),
                'param_grid': {
                    'logisticregression__C': [0.01, 0.1, 1, 10],
                    'logisticregression__solver': ['liblinear', 'saga'],
                    'logisticregression__penalty': ['l1', 'l2']
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=self.random_state, class_weight='balanced', n_jobs=1),
                'param_grid': {
                    'randomforestclassifier__n_estimators': [100, 200],
                    'randomforestclassifier__max_depth': [5, 10, 20, None],
                    'randomforestclassifier__min_samples_split': [2, 5]
                }
            },
            'HistGradientBoosting': {
                'model': HistGradientBoostingClassifier(random_state=self.random_state),
                'param_grid': {
                    'histgradientboostingclassifier__max_iter': [100, 200],
                    'histgradientboostingclassifier__max_depth': [3, 6, None],
                    'histgradientboostingclassifier__learning_rate': [0.05, 0.1]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'param_grid': {
                    'gradientboostingclassifier__n_estimators': [100, 200],
                    'gradientboostingclassifier__learning_rate': [0.05, 0.1],
                    'gradientboostingclassifier__max_depth': [3, 5]
                }
            },
            'SVM': {
                'model': LinearSVC(random_state=self.random_state, class_weight='balanced', max_iter=5000, dual=False),
                'param_grid': {
                    'linearsvc__C': [0.01, 0.1, 1, 10],
                }
            },
            'Bagging': {
                'model': BaggingClassifier(random_state=self.random_state),
                'param_grid': {
                    'baggingclassifier__n_estimators': [10, 20],
                    'baggingclassifier__max_samples': [0.5, 0.8, 1.0],
                    'baggingclassifier__max_features': [0.5, 0.8, 1.0]
                }
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'param_grid': {
                    'kneighborsclassifier__n_neighbors': [3, 5, 7, 9],
                    'kneighborsclassifier__weights': ['uniform', 'distance'],
                    'kneighborsclassifier__p': [1, 2]
                }
            }
        }

    def _shrink_param_grid(self, grid: Dict[str, Iterable], mode: str) -> Dict[str, Iterable]:
        """Reduce parameter grid size according to speed mode."""
        if mode == 'thorough':
            return grid
        keep_map = {'fast': 1, 'balanced': 2}
        keep = keep_map.get(mode, 2)
        new_grid = {}
        for k, v in grid.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                vals = list(dict.fromkeys(v))
                new_grid[k] = vals[:max(1, min(len(vals), keep))]
            else:
                new_grid[k] = v
        return new_grid

    def create_pipeline(self, model, use_smote: bool = True):
        steps = []
        if use_smote:
            steps.append(('smote', SMOTE(random_state=self.random_state)))
        model_step_name = type(model).__name__.lower()
        steps.append((model_step_name, model))
        return ImbPipeline(steps)

    def train_model(
        self,
        model_name: str,
        X_train,
        y_train,
        run_modes: Tuple[str, ...] = None,  # ('no_smote',) or ('no_smote','smote')
        sample_frac: float = 1.0,
        cv: Optional[int] = None,
        n_iter: Optional[int] = None,
        n_jobs: Optional[int] = None,
        search_method: Optional[str] = None,
        random_state: Optional[int] = None
    ):
        """
        Train one model (possibly multiple runs: no_smote / smote) using RandomizedSearchCV by default.
        Only CV metrics are stored (search.best_score_). Does not compute train/val ROC.
        """
        if random_state is None:
            random_state = self.random_state
        if cv is None:
            cv = self.default_cv
        if n_jobs is None:
            n_jobs = self.default_n_jobs
        if search_method is None:
            search_method = self.default_search_method

        available = self.get_available_models()
        if model_name not in available:
            raise KeyError(f"Model '{model_name}' not available. Choose from: {list(available.keys())}")

        if run_modes is None:
            run_modes = ('smote', 'no_smote') if self.run_both_smote_by_default else ('no_smote',)

        model_info = available[model_name]
        base_model = model_info['model']
        original_grid = model_info.get('param_grid', {})

        # choose n_iter
        if n_iter is None:
            n_iter = self.n_iter_defaults.get(model_name, 12)
            # adjust according to speed_mode
            if self.speed_mode == 'fast':
                n_iter = max(4, math.ceil(n_iter / 3))
            elif self.speed_mode == 'balanced':
                n_iter = max(8, math.ceil(n_iter / 1.5))
            else:  # thorough
                n_iter = n_iter

        # sample train if requested
        if sample_frac is not None and (sample_frac <= 0 or sample_frac > 1):
            raise ValueError("sample_frac must be in (0, 1]. Use 1.0 to use full train.")
        if sample_frac < 1.0:
            print(f"[INFO] Sampling train: frac={sample_frac}")
            Xs = X_train.sample(frac=sample_frac, random_state=random_state)

            if hasattr(y_train, 'loc'):
                ys = y_train.loc[Xs.index]
            else:
                try:
                    idx = Xs.index.to_numpy().astype(int)
                    ys = y_train[idx]
                except Exception:
                    Xs = Xs.reset_index(drop=True)
                    ys = np.asarray(y_train).reshape(-1)
                    ys = ys[: len(Xs)]

            Xs = Xs.reset_index(drop=True)
            if hasattr(ys, 'reset_index'):
                ys = ys.reset_index(drop=True)
            else:
                ys = np.asarray(ys)
        else:
            Xs, ys = X_train, y_train

        print(f"[INFO] Training model '{model_name}' | speed_mode={self.speed_mode} | search_method={search_method}")
        for run_key in run_modes:
            use_smote = (run_key == 'smote')
            # shrink grid based on speed_mode
            grid = self._shrink_param_grid(original_grid, self.speed_mode)

            # if RandomizedSearchCV, param_distributions should be provided
            if search_method == 'grid':
                # grid search (careful: can be expensive)
                searcher = GridSearchCV(
                    estimator=self.create_pipeline(base_model, use_smote=use_smote),
                    param_grid=grid,
                    cv=cv,
                    scoring='roc_auc',
                    n_jobs=n_jobs,
                    verbose=1
                )
            else:
                # randomized search: if grid contains single-value lists, RandomizedSearchCV still works
                searcher = RandomizedSearchCV(
                    estimator=self.create_pipeline(base_model, use_smote=use_smote),
                    param_distributions=grid,
                    n_iter=n_iter,
                    cv=cv,
                    scoring='roc_auc',
                    n_jobs=n_jobs,
                    verbose=1,
                    random_state=random_state
                )

            print(f"[RUN] {model_name} / {run_key} | n_iter={n_iter} | cv={cv} | use_smote={use_smote}")
            searcher.fit(Xs, ys)

            # store results (only CV metrics and best_estimator/best_params)
            if model_name not in self.models:
                self.models[model_name] = {}

            self.models[model_name][run_key] = {
                'best_estimator': searcher.best_estimator_,
                'best_params': searcher.best_params_,
                'cv_score': searcher.best_score_,
                'search': searcher,
                'metadata': {
                    'use_smote': use_smote,
                    'sample_frac': sample_frac,
                    'speed_mode': self.speed_mode
                }
            }

            print(f"[DONE] {model_name}/{run_key} | CV AUC={searcher.best_score_:.4f} | best_params={searcher.best_params_}")

        return self.models[model_name]

    def train_all(
        self,
        X_train,
        y_train,
        model_names: Optional[Iterable[str]] = None,
        run_modes: Tuple[str, ...] = None,
        sample_frac: float = 1.0,
        cv: Optional[int] = None,
        n_jobs: Optional[int] = None,
        search_method: Optional[str] = None
    ):
        """Train multiple models sequentially. Returns dict of results."""
        available = list(self.get_available_models().keys())
        if model_names is None:
            model_names = available
        results = {}
        for mn in model_names:
            res = self.train_model(
                model_name=mn,
                X_train=X_train,
                y_train=y_train,
                run_modes=run_modes,
                sample_frac=sample_frac,
                cv=cv,
                n_jobs=n_jobs,
                search_method=search_method
            )
            results[mn] = res
        return results

    def save_trained_model(self, model_name: str, path: str, use_smote: bool = True):
        """Save stored trained estimator to disk."""
        run_key = 'smote' if use_smote else 'no_smote'
        if model_name not in self.models or run_key not in self.models[model_name]:
            raise KeyError(f"No stored model for {model_name} with run '{run_key}'.")
        estimator = self.models[model_name][run_key]['best_estimator']
        joblib.dump(estimator, path)
        print(f"Saved: {path}")

    def load_model(self, path: str):
        model = joblib.load(path)
        print(f"Loaded: {path}")
        return model
