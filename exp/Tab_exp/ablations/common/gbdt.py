"""GBDT baselines (LightGBM, XGBoost, CatBoost) behind the runner's model interface.

Each adapter class looks like a mambular classifier to common/runner.py:
    cls(**params).fit(X, y, X_val=..., y_val=..., random_state=seed, **ignored)
    .predict_proba(X) -> (n, n_classes)
so Phase 1 (n_iter candidates x cv folds on the phase-1 sample, ROC-AUC selection),
Phase 2, chunked inference, saved predictions and paired tests are shared verbatim.

Fairness rules (so GBDTs and deep models are compared on equal terms):
- same split, same feature matrices as the mambular arms (built by features.py);
- CatBoost receives the label-encoded categorical columns as native `cat_features`;
  LightGBM / XGBoost get the very same integer codes as numeric columns;
- early stopping on validation AUC (the explicit val sample the runner passes);
- same search budget: the runner's n_iter / cv / phase1_frac;
- determinism: fixed seeds and a fixed thread count.
"""
import numpy as np

MAX_ROUNDS = 2000
EARLY_STOP = 50
THREADS = 4

GRIDS = {
    "LightGBM": {"num_leaves": [15, 31, 63, 127], "learning_rate": [0.03, 0.1, 0.3],
                 "min_child_samples": [20, 100, 500], "colsample_bytree": [0.7, 1.0]},
    "XGBoost": {"max_depth": [4, 6, 8, 10], "learning_rate": [0.03, 0.1, 0.3],
                "min_child_weight": [1, 10, 100], "colsample_bytree": [0.7, 1.0]},
    "CatBoost": {"depth": [4, 6, 8, 10], "learning_rate": [0.03, 0.1, 0.3],
                 "l2_leaf_reg": [1, 3, 10], "rsm": [0.7, 1.0]},
}


class _GBDT:
    """Base adapter. Subclasses set NAME and implement _make/_fit."""
    is_gbdt = True
    NAME = ""

    def __init__(self, cat_features=None, **params):
        self.cat_features = list(cat_features or [])
        self.params = params
        self.model = None
        self.best_iteration_ = None

    @classmethod
    def grid(cls):
        return GRIDS[cls.NAME]

    def fit(self, X, y, X_val=None, y_val=None, random_state=42, **_ignored):
        if X_val is None:
            raise ValueError(f"{self.NAME}: early stopping needs an explicit X_val (phase1_val_mode=explicit)")
        y, y_val = np.asarray(y), np.asarray(y_val)
        self.n_classes_ = int(len(np.unique(y)))
        self._fit(X, y, X_val, y_val, int(random_state))
        return self

    def predict_proba(self, X):
        return np.asarray(self.model.predict_proba(X), dtype=np.float64)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class LightGBMClassifier(_GBDT):
    NAME = "LightGBM"

    def _fit(self, X, y, Xv, yv, seed):
        import lightgbm as lgb
        self.model = lgb.LGBMClassifier(
            n_estimators=MAX_ROUNDS, random_state=seed, n_jobs=THREADS, deterministic=True,
            force_col_wise=True, verbose=-1, **self.params)
        self.model.fit(X, y, eval_set=[(Xv, yv)],
                       eval_metric="auc" if self.n_classes_ == 2 else "multi_logloss",
                       callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        self.best_iteration_ = self.model.best_iteration_


class XGBoostClassifier(_GBDT):
    NAME = "XGBoost"

    def _fit(self, X, y, Xv, yv, seed):
        import xgboost as xgb
        self.model = xgb.XGBClassifier(
            n_estimators=MAX_ROUNDS, tree_method="hist", random_state=seed, n_jobs=THREADS,
            eval_metric="auc" if self.n_classes_ == 2 else "mlogloss",
            early_stopping_rounds=EARLY_STOP, **self.params)
        self.model.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
        self.best_iteration_ = self.model.best_iteration


class CatBoostClassifier(_GBDT):
    NAME = "CatBoost"

    def _fit(self, X, y, Xv, yv, seed):
        from catboost import CatBoostClassifier as CB
        self.model = CB(
            iterations=MAX_ROUNDS, random_seed=seed, thread_count=THREADS,
            eval_metric="AUC" if self.n_classes_ == 2 else "MultiClass",
            od_type="Iter", od_wait=EARLY_STOP, use_best_model=True,
            verbose=False, allow_writing_files=False,
            cat_features=[c for c in self.cat_features if c in X.columns], **self.params)
        self.model.fit(X, y, eval_set=(Xv, yv))
        self.best_iteration_ = self.model.get_best_iteration()


MODELS = {c.NAME: c for c in (LightGBMClassifier, XGBoostClassifier, CatBoostClassifier)}
