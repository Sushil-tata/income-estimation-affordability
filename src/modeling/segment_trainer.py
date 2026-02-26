"""
Segment Model Trainer
──────────────────────
Trains the best model per segment by:
  1. Applying label engineering (segment-specific label strategy)
  2. Applying feature selection (segment-specific features)
  3. Training multiple models with multiple loss functions
  4. Selecting the best (model, loss, label) combination by CV MAE

This is the central orchestrator for the advanced modeling workstream.

Architecture per segment:
  ─────────────────────────────────────────────────────────────────
  Candidate models:
    LightGBM × {huber_10k, quantile_p30/40/50, log_rmse, mape, tweedie_p15}
    VanillaLSTM, BiLSTM, LSTMWithAttention, TCN (if torch available)
    TabPFN v2 (if tabpfn available)
    AutoGluon (if autogluon available)

  Candidate label strategies:
    raw, robust, composite, shrunk_composite, quantile

  Selection criterion: 5-fold CV MAE on verified income (always evaluated vs raw y)
  ─────────────────────────────────────────────────────────────────

Output: per-segment (model_name, loss, label_strategy, cv_mae, model_object)
"""

import numpy as np
import pandas as pd
import logging
import pickle
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from sklearn.model_selection import KFold

import lightgbm as lgb

from .label_engineering import LabelEngineer
from .loss_functions import LossRegistry, evaluate_income_predictions
from ..feature_selection import FeatureSelectionPipeline

logger = logging.getLogger(__name__)


# ── LightGBM wrapper conforming to BaseIncomeModel interface ──────────────────

class LGBMIncomeModel:
    """Thin LightGBM wrapper with configurable objective."""

    def __init__(self, loss_name: str = "huber_10k", n_estimators: int = 500,
                 learning_rate: float = 0.05, max_depth: int = 5,
                 num_leaves: int = 31, random_state: int = 42):
        self.loss_name = loss_name
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.random_state = random_state
        self._model = None
        self.model_name = f"LightGBM_{loss_name}"

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "LGBMIncomeModel":
        obj_fn, eval_fn = LossRegistry.get(self.loss_name)
        X_num = X.select_dtypes(include=[np.number]).fillna(0)

        params = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        dtrain = lgb.Dataset(X_num, label=y.values)
        lgb_params = {k: v for k, v in params.items()
                      if k not in ("n_estimators",)}
        lgb_params["verbose"] = -1

        self._model = lgb.train(
            lgb_params, dtrain,
            num_boost_round=self.n_estimators,
            fobj=obj_fn,
            feval=eval_fn,
            valid_sets=[dtrain],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_num = X.select_dtypes(include=[np.number]).fillna(0)
        return np.maximum(self._model.predict(X_num), 0)


# ── Segment Model Trainer ─────────────────────────────────────────────────────

class SegmentModelTrainer:
    """
    Trains and selects the best (model, loss, label) per behavioral segment.

    Parameters
    ----------
    label_engineer : LabelEngineer
        Fitted LabelEngineer instance.
    feature_selector : FeatureSelectionPipeline, optional
        Fitted FeatureSelectionPipeline. If None, all features used.
    cv_folds : int
        Cross-validation folds for model selection. Default 5.
    lgb_losses : list
        LightGBM losses to try per segment.
    label_strategies : list
        Label strategies to try per segment.
    include_lstm : bool
        Include LSTM models (requires torch). Default True.
    include_tabpfn : bool
        Include TabPFN v2 (requires tabpfn). Default True.
    include_autogluon : bool
        Include AutoGluon (requires autogluon). Default False (heavy install).
    max_rows_per_segment : int
        Cap segment size for speed during search. Default 20,000.
    random_state : int
    """

    DEFAULT_LGB_LOSSES = [
        "huber_10k", "huber_20k",
        "quantile_p30", "quantile_p40", "quantile_p50",
        "log_rmse", "mape", "tweedie_p15",
    ]
    DEFAULT_LABEL_STRATEGIES = ["raw", "robust", "quantile", "shrunk_composite"]

    def __init__(
        self,
        label_engineer: LabelEngineer,
        feature_selector: Optional[FeatureSelectionPipeline] = None,
        cv_folds: int = 5,
        lgb_losses: Optional[List[str]] = None,
        label_strategies: Optional[List[str]] = None,
        include_lstm: bool = True,
        include_tabpfn: bool = True,
        include_autogluon: bool = False,
        max_rows_per_segment: int = 20_000,
        random_state: int = 42,
    ):
        self.label_engineer = label_engineer
        self.feature_selector = feature_selector
        self.cv_folds = cv_folds
        self.lgb_losses = lgb_losses or self.DEFAULT_LGB_LOSSES
        self.label_strategies = label_strategies or self.DEFAULT_LABEL_STRATEGIES
        self.include_lstm = include_lstm
        self.include_tabpfn = include_tabpfn
        self.include_autogluon = include_autogluon
        self.max_rows_per_segment = max_rows_per_segment
        self.random_state = random_state

        # Results
        self.segment_results_: Dict[str, pd.DataFrame] = {}
        self.best_per_segment_: Dict[str, dict] = {}
        self.fitted_models_: Dict[str, Any] = {}   # segment → fitted model

    def fit(
        self,
        X: pd.DataFrame,
        y_verified: pd.Series,
        segment_col: str = "segment",
    ) -> "SegmentModelTrainer":
        """
        Search best (model, loss, label) per segment then refit on full segment data.
        """
        segments = X[segment_col].unique()
        logger.info(f"SegmentModelTrainer: searching across {len(segments)} segments...")

        for seg in segments:
            mask = X[segment_col] == seg
            if mask.sum() < 200:
                logger.warning(f"Segment {seg}: only {mask.sum()} rows — skipping")
                continue

            X_seg = X[mask].reset_index(drop=True)
            y_seg = y_verified[mask].reset_index(drop=True)

            # Cap for speed
            if len(X_seg) > self.max_rows_per_segment:
                rng = np.random.RandomState(self.random_state)
                idx = rng.choice(len(X_seg), self.max_rows_per_segment, replace=False)
                X_seg = X_seg.iloc[idx].reset_index(drop=True)
                y_seg = y_seg.iloc[idx].reset_index(drop=True)

            # Feature selection
            feat_cols = (self.feature_selector.final_features_
                         if self.feature_selector else
                         X_seg.select_dtypes(include=[np.number]).columns.tolist())
            feat_cols = [c for c in feat_cols if c in X_seg.columns]

            logger.info(f"\nSegment {seg}: {len(X_seg):,} rows, {len(feat_cols)} features")

            results = self._search_segment(X_seg, y_seg, feat_cols, seg)
            self.segment_results_[seg] = pd.DataFrame(results).sort_values("cv_mae")

            best = self.segment_results_[seg].iloc[0]
            self.best_per_segment_[seg] = best.to_dict()

            logger.info(f"  Best for {seg}: model={best['model']}, "
                        f"loss={best['loss']}, label={best['label']}, "
                        f"cv_mae={best['cv_mae']:,.0f} THB")

            # Refit best on full segment
            self.fitted_models_[seg] = self._refit_best(
                X[mask], y_verified[mask], best, feat_cols, seg
            )

        return self

    def predict(
        self, X: pd.DataFrame, segment_col: str = "segment"
    ) -> pd.Series:
        """Predict income using segment-specific best model."""
        result = pd.Series(np.nan, index=X.index)
        for seg, model in self.fitted_models_.items():
            mask = X[segment_col] == seg
            if mask.any():
                result[mask] = model.predict(X[mask])
        return result.fillna(0)

    def results_summary(self) -> pd.DataFrame:
        """Return full search results across all segments."""
        rows = []
        for seg, df in self.segment_results_.items():
            df = df.copy()
            df["segment"] = seg
            rows.append(df)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # ── PRIVATE ──────────────────────────────────────────────────────────────

    def _search_segment(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feat_cols: List[str],
        segment: str,
    ) -> List[dict]:
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        results = []

        for label_strat in self.label_strategies:
            try:
                y_label = self.label_engineer.transform(
                    X, y, strategy=label_strat, segment_col="segment"
                    if "segment" in X.columns else None
                )
            except Exception as e:
                logger.debug(f"Label strategy {label_strat} failed: {e}")
                continue

            # ── LightGBM × losses ────────────────────────────────────────
            for loss_name in self.lgb_losses:
                cv_mae = self._cv_lgbm(X[feat_cols], y_label, y_true=y,
                                        loss_name=loss_name, kf=kf)
                results.append({
                    "model": "LightGBM",
                    "loss": loss_name,
                    "label": label_strat,
                    "cv_mae": cv_mae,
                })

            # ── Deep Learning (tabular features only for now) ────────────
            if self.include_tabpfn:
                try:
                    cv_mae = self._cv_tabpfn(X[feat_cols], y_label, y_true=y, kf=kf)
                    results.append({
                        "model": "TabPFN_v2",
                        "loss": "default",
                        "label": label_strat,
                        "cv_mae": cv_mae,
                    })
                except Exception as e:
                    logger.debug(f"TabPFN skipped: {e}")

        return results

    def _cv_lgbm(
        self,
        X: pd.DataFrame,
        y_label: pd.Series,
        y_true: pd.Series,
        loss_name: str,
        kf: KFold,
    ) -> float:
        obj_fn, eval_fn = LossRegistry.get(loss_name)
        maes = []
        for train_idx, val_idx in kf.split(X):
            X_tr, y_tr = X.iloc[train_idx].fillna(0), y_label.iloc[train_idx]
            X_val = X.iloc[val_idx].fillna(0)
            y_val_true = y_true.iloc[val_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr.values)
            model = lgb.train(
                {"learning_rate": 0.05, "max_depth": 5, "num_leaves": 31,
                 "verbose": -1, "n_jobs": -1},
                dtrain, num_boost_round=200, fobj=obj_fn,
                callbacks=[lgb.log_evaluation(-1)],
            )
            preds = np.maximum(model.predict(X_val), 0)
            maes.append(np.mean(np.abs(y_val_true.values - preds)))
        return float(np.mean(maes))

    def _cv_tabpfn(
        self,
        X: pd.DataFrame,
        y_label: pd.Series,
        y_true: pd.Series,
        kf: KFold,
    ) -> float:
        from .tabpfn_model import TabPFNModel
        maes = []
        for train_idx, val_idx in kf.split(X):
            model = TabPFNModel(max_train_rows=3000, log_target=False, n_estimators=4)
            model.fit(X.iloc[train_idx], y_label.iloc[train_idx])
            preds = model.predict(X.iloc[val_idx])
            maes.append(np.mean(np.abs(y_true.iloc[val_idx].values - preds)))
        return float(np.mean(maes))

    def _refit_best(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        best: pd.Series,
        feat_cols: List[str],
        segment: str,
    ) -> Any:
        """Refit the winning model on the full segment data."""
        y_label = self.label_engineer.transform(
            X, y, strategy=best["label"],
            segment_col="segment" if "segment" in X.columns else None
        )
        X_feat = X[feat_cols]

        if best["model"] == "LightGBM":
            model = LGBMIncomeModel(loss_name=best["loss"], random_state=self.random_state)
            model.fit(X_feat, y_label)
            return model

        elif best["model"] == "TabPFN_v2":
            from .deep_learning.tabpfn_model import TabPFNModel
            model = TabPFNModel(log_target=False, random_state=self.random_state)
            model.fit(X_feat, y_label)
            return model

        else:
            # Fallback
            model = LGBMIncomeModel(loss_name="huber_10k", random_state=self.random_state)
            model.fit(X_feat, y_label)
            return model

    def save(self, directory: str) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        with open(f"{directory}/segment_trainer.pkl", "wb") as f:
            pickle.dump(self, f)
        logger.info(f"SegmentModelTrainer saved to {directory}/")

    @classmethod
    def load(cls, directory: str) -> "SegmentModelTrainer":
        with open(f"{directory}/segment_trainer.pkl", "rb") as f:
            return pickle.load(f)
