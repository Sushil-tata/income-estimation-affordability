"""
Model Ensemble
───────────────
Combines predictions from multiple models into a single income estimate.

Two ensemble strategies:
  1. WeightedEnsemble   — fixed or learned weights per model
  2. StackingEnsemble   — meta-learner (LightGBM) trained on OOF predictions
  3. SegmentEnsemble    — routes each customer to the best model for their segment

The SegmentEnsemble is the primary output of SegmentModelTrainer.
WeightedEnsemble and StackingEnsemble are useful when multiple models
perform similarly and combining reduces variance.
"""

import numpy as np
import pandas as pd
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


class WeightedEnsemble:
    """
    Weighted average of multiple model predictions.

    Weights can be fixed (uniform) or optimised on validation data
    using non-negative least squares.

    Parameters
    ----------
    models : dict
        {model_name: fitted_model} — each must have predict(X) → np.ndarray.
    weights : dict, optional
        {model_name: weight}. If None, equal weights.
        Set learn_weights=True to optimise from data.
    learn_weights : bool
        If True, optimise weights on val_X, val_y using NNLS. Default False.
    clip_min : float
        Clip predictions below this value. Default 0.
    """

    def __init__(
        self,
        models: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
        learn_weights: bool = False,
        clip_min: float = 0.0,
    ):
        self.models = models
        self.weights = weights
        self.learn_weights = learn_weights
        self.clip_min = clip_min
        self._fitted_weights: Dict[str, float] = {}

    def fit(
        self,
        val_X: pd.DataFrame,
        val_y: pd.Series,
    ) -> "WeightedEnsemble":
        """Optimise ensemble weights on validation data."""
        if not self.learn_weights:
            n = len(self.models)
            self._fitted_weights = {name: 1.0 / n for name in self.models}
            return self

        from scipy.optimize import nnls
        # Stack OOF/val predictions
        preds = np.column_stack([
            np.clip(m.predict(val_X), self.clip_min, None)
            for m in self.models.values()
        ])
        weights, _ = nnls(preds, val_y.values)
        # Normalise
        total = weights.sum()
        if total > 0:
            weights /= total
        self._fitted_weights = dict(zip(self.models.keys(), weights))

        logger.info("WeightedEnsemble weights:")
        for name, w in self._fitted_weights.items():
            logger.info(f"  {name}: {w:.3f}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted_weights:
            n = len(self.models)
            self._fitted_weights = {name: 1.0 / n for name in self.models}

        result = np.zeros(len(X))
        for name, model in self.models.items():
            w = self._fitted_weights.get(name, 1.0 / len(self.models))
            preds = np.clip(model.predict(X), self.clip_min, None)
            result += w * preds

        return result

    def evaluate(self, X: pd.DataFrame, y: pd.Series,
                 segment: Optional[pd.Series] = None) -> pd.DataFrame:
        from .loss_functions import evaluate_income_predictions
        preds = pd.Series(self.predict(X), index=X.index)
        return evaluate_income_predictions(y, preds, segment)


class StackingEnsemble:
    """
    Stacking ensemble with LightGBM meta-learner.

    Level-1 models make out-of-fold predictions on training data.
    Level-2 LightGBM is trained on these OOF predictions to predict income.

    At inference: Level-1 predicts → Level-2 combines.

    Parameters
    ----------
    base_models : dict
        {model_name: unfitted_model_factory} or {model_name: fitted_model}.
    cv_folds : int
        OOF folds. Default 5.
    meta_params : dict
        LightGBM params for meta-learner.
    """

    def __init__(
        self,
        base_models: Dict[str, Any],
        cv_folds: int = 5,
        meta_params: Optional[dict] = None,
        random_state: int = 42,
    ):
        self.base_models = base_models
        self.cv_folds = cv_folds
        self.meta_params = meta_params or {
            "n_estimators": 200, "learning_rate": 0.05,
            "max_depth": 3, "num_leaves": 15,
            "verbose": -1, "n_jobs": -1,
        }
        self.random_state = random_state
        self._fitted_base_: Dict[str, Any] = {}
        self._meta_model = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "StackingEnsemble":
        import lightgbm as lgb
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        n = len(X)
        oof_preds = np.zeros((n, len(self.base_models)))

        model_names = list(self.base_models.keys())

        # Generate OOF predictions per base model
        for col_idx, name in enumerate(model_names):
            logger.info(f"Stacking OOF: {name}...")
            model_factory = self.base_models[name]

            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
                X_val = X.iloc[val_idx]

                m = model_factory()
                m.fit(X_tr, y_tr)
                oof_preds[val_idx, col_idx] = np.clip(m.predict(X_val), 0, None)

        # Refit each base model on full training data
        for name, factory in self.base_models.items():
            logger.info(f"Stacking refit: {name}...")
            m = factory()
            m.fit(X, y)
            self._fitted_base_[name] = m

        # Train meta-learner on OOF predictions
        logger.info("Stacking: training meta-learner...")
        oof_df = pd.DataFrame(oof_preds, columns=model_names)
        meta = lgb.LGBMRegressor(**self.meta_params)
        meta.fit(oof_df, y)
        self._meta_model = meta

        logger.info("StackingEnsemble: fit complete")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base_preds = np.column_stack([
            np.clip(m.predict(X), 0, None)
            for m in self._fitted_base_.values()
        ])
        base_df = pd.DataFrame(base_preds, columns=list(self._fitted_base_.keys()))
        return np.maximum(self._meta_model.predict(base_df), 0)

    def evaluate(self, X: pd.DataFrame, y: pd.Series,
                 segment: Optional[pd.Series] = None) -> pd.DataFrame:
        from .loss_functions import evaluate_income_predictions
        preds = pd.Series(self.predict(X), index=X.index)
        return evaluate_income_predictions(y, preds, segment)


class SegmentEnsemble:
    """
    Routes each customer to the segment-specific best model.

    This is the primary production ensemble — it wraps SegmentModelTrainer
    and provides a unified predict() interface.

    Parameters
    ----------
    segment_models : dict
        {segment: fitted_model} — output of SegmentModelTrainer.fitted_models_.
    fallback_model : optional
        Model to use for unseen/missing segments.
    """

    def __init__(
        self,
        segment_models: Dict[str, Any],
        fallback_model: Optional[Any] = None,
    ):
        self.segment_models = segment_models
        self.fallback_model = fallback_model

    def predict(
        self,
        X: pd.DataFrame,
        segment_col: str = "segment",
        feature_cols: Optional[List[str]] = None,
    ) -> pd.Series:
        result = pd.Series(np.nan, index=X.index)

        for seg, model in self.segment_models.items():
            mask = X[segment_col] == seg if segment_col in X.columns else pd.Series(False, index=X.index)
            if mask.any():
                X_seg = X[mask]
                if feature_cols:
                    X_seg = X_seg[[c for c in feature_cols if c in X_seg.columns]]
                result[mask] = np.clip(model.predict(X_seg), 0, None)

        # Handle unseen segments
        nan_mask = result.isna()
        if nan_mask.any() and self.fallback_model is not None:
            X_fallback = X[nan_mask]
            if feature_cols:
                X_fallback = X_fallback[[c for c in feature_cols if c in X_fallback.columns]]
            result[nan_mask] = np.clip(self.fallback_model.predict(X_fallback), 0, None)

        return result.fillna(0)

    def predict_with_interval(
        self, X: pd.DataFrame, segment_col: str = "segment"
    ) -> pd.DataFrame:
        """Return Q25/Q50/Q75 if models support it, else derive from point estimate."""
        point = self.predict(X, segment_col)
        # Default uncertainty: ±15% band (override if models provide intervals)
        return pd.DataFrame({
            "income_q25": point * 0.85,
            "income_q50": point,
            "income_q75": point * 1.15,
            "income_estimate": point,
        }, index=X.index)

    def evaluate(
        self, X: pd.DataFrame, y: pd.Series,
        segment_col: str = "segment",
    ) -> pd.DataFrame:
        from .loss_functions import evaluate_income_predictions
        preds = self.predict(X, segment_col)
        segment = X[segment_col] if segment_col in X.columns else None
        return evaluate_income_predictions(y, preds, segment)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "SegmentEnsemble":
        with open(path, "rb") as f:
            return pickle.load(f)
