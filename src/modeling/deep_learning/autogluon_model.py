"""
AutoGluon Wrapper
──────────────────
AutoGluon automatically searches over LightGBM, XGBoost, CatBoost, Neural Nets,
TabPFN, Random Forest, and builds a stacked ensemble — all with a single fit() call.

For GDZ:
  - Set time_limit aggressively (default 300s = 5min) to cap compute
  - Use presets='medium_quality' to balance speed vs accuracy
  - Disable neural nets if torch unavailable: excluded_model_types=['NN_TORCH']
  - AutoGluon creates a local model directory — stays within GDZ

Install: pip install autogluon.tabular
         (Note: ~2GB install — confirm GDZ package policy before installing)

Fallback: If AutoGluon unavailable, falls back to a simple LightGBM ensemble
          so the pipeline never breaks.

Parameters
----------
time_limit : int
    Max training time in seconds. Default 300 (5 min) for GDZ constraints.
presets : str
    AutoGluon quality preset. Default 'medium_quality'.
    Options: 'best_quality', 'high_quality', 'medium_quality', 'optimize_for_deployment'
excluded_model_types : list
    Model types to skip. Default excludes heavy DL models for GDZ compute.
log_target : bool
    Train on log1p(income). Default True.
"""

import numpy as np
import pandas as pd
import logging
import pickle
import shutil
from pathlib import Path
from typing import Optional, List

from ..base import BaseIncomeModel

logger = logging.getLogger(__name__)


def _check_autogluon():
    try:
        from autogluon.tabular import TabularPredictor
        return TabularPredictor
    except ImportError:
        return None


class AutoGluonModel(BaseIncomeModel):
    """
    AutoGluon TabularPredictor wrapper for income estimation.

    Falls back to LightGBM if AutoGluon is not installed.

    Parameters
    ----------
    time_limit : int
        Seconds budget for AutoGluon search. Default 300.
    presets : str
        Quality preset. Default 'medium_quality'.
    excluded_model_types : list
        Skip these model types. Default: heavy DL models excluded for GDZ.
    log_target : bool
        Train on log1p(income). Default True.
    save_dir : str
        Directory for AutoGluon model artifacts.
    feature_cols : list, optional
        Feature columns to use.
    random_state : int
    """

    GDZ_EXCLUDED = ["NN_TORCH", "FASTAI"]   # Exclude heavy DL for GDZ compute

    def __init__(
        self,
        time_limit: int = 300,
        presets: str = "medium_quality",
        excluded_model_types: Optional[List[str]] = None,
        log_target: bool = True,
        save_dir: str = "artifacts/models/autogluon/",
        feature_cols: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.time_limit = time_limit
        self.presets = presets
        self.excluded_model_types = excluded_model_types or self.GDZ_EXCLUDED
        self.log_target = log_target
        self.save_dir = save_dir
        self.feature_cols = feature_cols
        self.random_state = random_state

        self._predictor = None
        self._fallback_model = None
        self._using_fallback = False
        self._feature_cols_fitted: List[str] = []
        self._label_col = "_income_target_"

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "AutoGluonModel":
        TabularPredictor = _check_autogluon()

        cols = self.feature_cols or X.select_dtypes(include=[np.number]).columns.tolist()
        self._feature_cols_fitted = cols
        X_train = X[cols].fillna(0).copy()

        y_target = np.log1p(y) if self.log_target else y
        X_train[self._label_col] = y_target.values

        if TabularPredictor is None:
            logger.warning("AutoGluon not installed — using LightGBM fallback")
            self._fit_fallback(X[cols], y)
            return self

        logger.info(f"AutoGluon: fitting with time_limit={self.time_limit}s, "
                    f"presets='{self.presets}', excluded={self.excluded_model_types}")
        logger.info(f"  Training set: {len(X_train):,} rows × {len(cols)} features")

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        self._predictor = TabularPredictor(
            label=self._label_col,
            problem_type="regression",
            eval_metric="mean_absolute_error",
            path=self.save_dir,
            verbosity=1,
        )
        self._predictor.fit(
            X_train,
            time_limit=self.time_limit,
            presets=self.presets,
            excluded_model_types=self.excluded_model_types,
        )

        logger.info("AutoGluon: fit complete")
        logger.info(f"  Leaderboard:\n{self._predictor.leaderboard(silent=True).head(5)}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cols = self._feature_cols_fitted
        X_score = X[cols].fillna(0)

        if self._using_fallback or self._predictor is None:
            preds = self._fallback_model.predict(X_score)
        else:
            preds = self._predictor.predict(X_score).values

        if self.log_target:
            preds = np.expm1(preds)
        return np.maximum(preds, 0)

    def leaderboard(self) -> Optional[pd.DataFrame]:
        """Return AutoGluon model leaderboard."""
        if self._predictor is None:
            return None
        return self._predictor.leaderboard(silent=True)

    def feature_importance(self) -> Optional[pd.DataFrame]:
        """Return AutoGluon feature importance."""
        if self._predictor is None:
            return None
        return self._predictor.feature_importance(silent=True)

    def _fit_fallback(self, X: pd.DataFrame, y: pd.Series):
        """LightGBM fallback when AutoGluon unavailable."""
        import lightgbm as lgb
        logger.info("AutoGluon fallback: training LightGBM ensemble...")
        y_target = np.log1p(y) if self.log_target else y
        models = []
        # Simple 3-model ensemble with different seeds
        for seed in [42, 123, 777]:
            m = lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=5,
                num_leaves=31, random_state=seed, n_jobs=-1, verbose=-1
            )
            m.fit(X.fillna(0), y_target)
            models.append(m)

        class _FallbackEnsemble:
            def __init__(self, models):
                self.models = models
            def predict(self, X):
                return np.mean([m.predict(X) for m in self.models], axis=0)

        self._fallback_model = _FallbackEnsemble(models)
        self._using_fallback = True

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # AutoGluon saves its own artifacts in save_dir; we pickle the wrapper
        meta = {
            "time_limit": self.time_limit,
            "presets": self.presets,
            "log_target": self.log_target,
            "save_dir": self.save_dir,
            "feature_cols_fitted": self._feature_cols_fitted,
            "using_fallback": self._using_fallback,
            "fallback_model": self._fallback_model,
        }
        with open(path, "wb") as f:
            pickle.dump(meta, f)
        logger.info(f"AutoGluonModel meta saved: {path}")

    @classmethod
    def load(cls, path: str) -> "AutoGluonModel":
        with open(path, "rb") as f:
            meta = pickle.load(f)

        obj = cls(
            time_limit=meta["time_limit"],
            presets=meta["presets"],
            log_target=meta["log_target"],
            save_dir=meta["save_dir"],
        )
        obj._feature_cols_fitted = meta["feature_cols_fitted"]
        obj._using_fallback = meta["using_fallback"]
        obj._fallback_model = meta["fallback_model"]

        if not meta["using_fallback"]:
            TabularPredictor = _check_autogluon()
            if TabularPredictor:
                obj._predictor = TabularPredictor.load(meta["save_dir"])
            else:
                logger.warning("AutoGluon not available — switching to fallback mode")
                obj._using_fallback = True

        return obj
