"""
Supervised Feature Selection
──────────────────────────────
Label-aware selectors that rank features by predictive power w.r.t income target.

Classes:
  SHAPRanker            — LightGBM SHAP-value based ranking
  BorutaSelector        — Shadow feature comparison (all-relevant selection)
  MRMRSelector          — Minimum Redundancy Maximum Relevance
  PermutationImportance — Model-agnostic permutation importance

Each exposes: fit(X, y) → selected_features_ attribute + transform(X)
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Callable
import logging
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


# ── SHAP Ranker ──────────────────────────────────────────────────────────────

class SHAPRanker:
    """
    Trains a LightGBM regressor and ranks features by mean |SHAP| value.

    Parameters
    ----------
    top_k : int or float
        If int → keep top_k features. If float → keep top fraction. Default 0.5.
    n_estimators : int
        Trees for the base LightGBM. Default 300.
    random_state : int
        Seed. Default 42.
    """

    def __init__(self, top_k=0.5, n_estimators: int = 300, random_state: int = 42):
        self.top_k = top_k
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.shap_importance_: Optional[pd.Series] = None
        self.selected_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series,
            feature_cols: Optional[List[str]] = None) -> "SHAPRanker":
        try:
            import shap
            import lightgbm as lgb
        except ImportError:
            raise ImportError("Install shap and lightgbm: pip install shap lightgbm")

        cols = feature_cols or X.select_dtypes(include=[np.number]).columns.tolist()
        X_num = X[cols].fillna(0)

        logger.info(f"SHAPRanker: training LightGBM on {len(cols)} features...")
        model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_num, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_num)

        self.shap_importance_ = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=cols,
            name="mean_abs_shap"
        ).sort_values(ascending=False)

        k = self._resolve_k(len(cols))
        self.selected_features_ = self.shap_importance_.head(k).index.tolist()

        logger.info(f"SHAPRanker: selected {len(self.selected_features_)} / {len(cols)} features")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.selected_features_ if c in X.columns]
        return X[available]

    def _resolve_k(self, n_features: int) -> int:
        if isinstance(self.top_k, float):
            return max(1, int(n_features * self.top_k))
        return min(self.top_k, n_features)

    def report(self) -> pd.DataFrame:
        if self.shap_importance_ is None:
            return pd.DataFrame()
        df = self.shap_importance_.reset_index()
        df.columns = ["feature", "mean_abs_shap"]
        df["selected"] = df["feature"].isin(self.selected_features_)
        return df


# ── Boruta Selector ──────────────────────────────────────────────────────────

class BorutaSelector:
    """
    Boruta-inspired all-relevant feature selection.

    Algorithm:
      1. Create shadow features (randomly permuted copies of each real feature)
      2. Train LightGBM on real + shadow features
      3. Compare each real feature's importance to max shadow importance
      4. Features consistently beating max shadow → confirmed important
      5. Repeat n_trials times, select features important in > hit_rate fraction

    Parameters
    ----------
    n_trials : int
        Number of shadow-feature trials. Default 20.
    hit_rate : float
        Min fraction of trials where feature beats max shadow. Default 0.60.
    n_estimators : int
        Trees per trial. Default 100 (kept small for speed).
    random_state : int
        Seed. Default 42.
    """

    def __init__(self, n_trials: int = 20, hit_rate: float = 0.60,
                 n_estimators: int = 100, random_state: int = 42):
        self.n_trials = n_trials
        self.hit_rate = hit_rate
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.hit_counts_: Optional[pd.Series] = None
        self.selected_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series,
            feature_cols: Optional[List[str]] = None) -> "BorutaSelector":
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("Install lightgbm: pip install lightgbm")

        cols = feature_cols or X.select_dtypes(include=[np.number]).columns.tolist()
        X_num = X[cols].fillna(0).values
        y_arr = y.values

        rng = np.random.RandomState(self.random_state)
        hit_counts = np.zeros(len(cols))

        logger.info(f"BorutaSelector: running {self.n_trials} trials on {len(cols)} features...")

        for trial in range(self.n_trials):
            # Create shadow features
            shadow = X_num.copy()
            for j in range(shadow.shape[1]):
                rng.shuffle(shadow[:, j])

            X_combined = np.hstack([X_num, shadow])
            col_names = cols + [f"shadow_{c}" for c in cols]

            model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=0.1,
                max_depth=4,
                random_state=rng.randint(0, 10000),
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(X_combined, y_arr)

            importances = model.feature_importances_
            real_imp = importances[:len(cols)]
            shadow_imp = importances[len(cols):]
            max_shadow = shadow_imp.max()

            hit_counts += (real_imp > max_shadow).astype(int)

            if (trial + 1) % 5 == 0:
                logger.debug(f"  Trial {trial+1}/{self.n_trials} complete")

        self.hit_counts_ = pd.Series(hit_counts / self.n_trials, index=cols,
                                      name="hit_rate").sort_values(ascending=False)

        self.selected_features_ = self.hit_counts_[
            self.hit_counts_ >= self.hit_rate
        ].index.tolist()

        logger.info(f"BorutaSelector: selected {len(self.selected_features_)} / {len(cols)} features")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.selected_features_ if c in X.columns]
        return X[available]

    def report(self) -> pd.DataFrame:
        if self.hit_counts_ is None:
            return pd.DataFrame()
        df = self.hit_counts_.reset_index()
        df.columns = ["feature", "hit_rate"]
        df["selected"] = df["feature"].isin(self.selected_features_)
        return df


# ── mRMR Selector ────────────────────────────────────────────────────────────

class MRMRSelector:
    """
    Minimum Redundancy Maximum Relevance (mRMR) feature selection.

    Greedy algorithm that at each step selects the feature maximizing:
        score = relevance(f, y) - (1/|S|) * avg_redundancy(f, S)

    Relevance: absolute Pearson correlation with target (or mutual information proxy).
    Redundancy: absolute Pearson correlation with already-selected features.

    Parameters
    ----------
    k : int or float
        Number of features to select. If float, fraction of total.
    relevance : str
        'pearson' or 'mutual_info'. Default 'pearson'.
    """

    def __init__(self, k=0.5, relevance: str = "pearson"):
        self.k = k
        self.relevance = relevance
        self.selected_features_: List[str] = []
        self.scores_: List[float] = []

    def fit(self, X: pd.DataFrame, y: pd.Series,
            feature_cols: Optional[List[str]] = None) -> "MRMRSelector":
        cols = feature_cols or X.select_dtypes(include=[np.number]).columns.tolist()
        X_num = X[cols].fillna(0)
        k = self._resolve_k(len(cols))

        logger.info(f"MRMRSelector: selecting {k} from {len(cols)} features...")

        # Relevance of each feature with target
        if self.relevance == "pearson":
            rel = X_num.corrwith(y).abs().fillna(0)
        else:
            rel = self._mutual_info_relevance(X_num, y)

        selected = []
        remaining = list(cols)
        scores_log = []

        for step in range(k):
            if not remaining:
                break

            if not selected:
                best = rel[remaining].idxmax()
                selected.append(best)
                remaining.remove(best)
                scores_log.append(rel[best])
            else:
                # Redundancy: avg correlation with already selected
                redundancy = X_num[remaining].corrwith(
                    X_num[selected].mean(axis=1)
                ).abs().fillna(0)

                mrmr_scores = rel[remaining] - redundancy
                best = mrmr_scores.idxmax()
                selected.append(best)
                remaining.remove(best)
                scores_log.append(mrmr_scores[best])

        self.selected_features_ = selected
        self.scores_ = scores_log
        logger.info(f"MRMRSelector: selected {len(selected)} features")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.selected_features_ if c in X.columns]
        return X[available]

    def _resolve_k(self, n: int) -> int:
        if isinstance(self.k, float):
            return max(1, int(n * self.k))
        return min(self.k, n)

    def _mutual_info_relevance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        from sklearn.feature_selection import mutual_info_regression
        mi = mutual_info_regression(X.fillna(0), y, random_state=42)
        return pd.Series(mi, index=X.columns)

    def report(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature": self.selected_features_,
            "selection_order": range(1, len(self.selected_features_) + 1),
            "mrmr_score": self.scores_,
            "selected": True,
        })


# ── Permutation Importance ────────────────────────────────────────────────────

class PermutationImportance:
    """
    Model-agnostic permutation importance.

    For each feature: measure increase in MAE when that feature is randomly shuffled.
    Features with high importance degradation are most predictive.

    Parameters
    ----------
    model : fitted model with predict(X) method.
    top_k : int or float
        Features to select by importance.
    n_repeats : int
        Number of permutation repeats per feature. Default 5.
    """

    def __init__(self, model=None, top_k=0.5, n_repeats: int = 5,
                 random_state: int = 42):
        self.model = model
        self.top_k = top_k
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.importances_: Optional[pd.Series] = None
        self.selected_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series,
            feature_cols: Optional[List[str]] = None,
            model=None) -> "PermutationImportance":
        m = model or self.model
        if m is None:
            raise ValueError("Provide a fitted model via model= or constructor")

        cols = feature_cols or X.select_dtypes(include=[np.number]).columns.tolist()
        X_num = X[cols].fillna(0)
        rng = np.random.RandomState(self.random_state)

        baseline_preds = m.predict(X_num)
        baseline_mae = mean_absolute_error(y, baseline_preds)

        logger.info(f"PermutationImportance: baseline MAE = {baseline_mae:,.0f} | "
                    f"evaluating {len(cols)} features × {self.n_repeats} repeats...")

        importances = {}
        for col in cols:
            maes = []
            for _ in range(self.n_repeats):
                X_perm = X_num.copy()
                X_perm[col] = rng.permutation(X_perm[col].values)
                perm_mae = mean_absolute_error(y, m.predict(X_perm))
                maes.append(perm_mae)
            importances[col] = np.mean(maes) - baseline_mae

        self.importances_ = pd.Series(importances).sort_values(ascending=False)

        k = self._resolve_k(len(cols))
        self.selected_features_ = self.importances_.head(k).index.tolist()

        logger.info(f"PermutationImportance: selected {len(self.selected_features_)} features")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.selected_features_ if c in X.columns]
        return X[available]

    def _resolve_k(self, n: int) -> int:
        if isinstance(self.top_k, float):
            return max(1, int(n * self.top_k))
        return min(self.top_k, n)

    def report(self) -> pd.DataFrame:
        if self.importances_ is None:
            return pd.DataFrame()
        df = self.importances_.reset_index()
        df.columns = ["feature", "importance_delta_mae"]
        df["selected"] = df["feature"].isin(self.selected_features_)
        return df
