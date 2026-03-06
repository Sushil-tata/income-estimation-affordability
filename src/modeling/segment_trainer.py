"""
Segment Model Trainer
──────────────────────
Trains the best model per persona (L0 / L1 / L2 / PAYROLL) by:
  1. Applying label engineering (persona-specific label strategy)
  2. Applying feature selection (persona-specific features)
  3. Training multiple models with multiple loss functions
  4. Selecting the best (model, loss, label) combination by CV MAE

THIN customers are excluded from ML training — they receive a regulatory
policy floor income (BOT minimum) applied by the downstream affordability engine.

Persona-specific search space defaults
───────────────────────────────────────
  PAYROLL : losses=[huber_10k, quantile_p50], labels=[raw, robust]
  L0      : losses=[huber_10k, quantile_p40, quantile_p50], labels=[shrunk_composite, robust]
  L1      : losses=[huber_20k, quantile_p30, quantile_p40], labels=[robust, quantile]
  L2      : losses=[quantile_p30, huber_20k],               labels=[quantile, robust]

Full grid is always evaluated when lgb_losses / label_strategies are supplied explicitly.
The defaults above narrow the default search when use_persona_defaults=True (default).

Output: per-persona (model_name, loss, label_strategy, cv_mae, model_object)
         + income_estimate_type: "ML_ESTIMATE" | "POLICY_FLOOR" | "UNSEEN_PERSONA"
"""

import numpy as np
import pandas as pd
import logging
import pickle
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from sklearn.model_selection import TimeSeriesSplit

import lightgbm as lgb

from .label_engineering import LabelEngineer
from .loss_functions import LossRegistry, evaluate_income_predictions
from feature_selection import FeatureSelectionPipeline

logger = logging.getLogger(__name__)


# ── Persona-aware search space defaults ──────────────────────────────────────

# Personas that bypass ML — income is formula-based, not ML-estimated.
# THIN  → BOT minimum policy floor.
# PT    → pass-through formula: median_credit_6m × min(retention_6m, retention_3m).
_POLICY_FLOOR_PERSONAS: set = {"THIN", "PT"}

# Narrowed loss / label candidates per persona.
# These apply when use_persona_defaults=True and the caller does not supply
# explicit lgb_losses / label_strategies.
PERSONA_DEFAULT_LOSSES: Dict[str, List[str]] = {
    "PAYROLL": ["huber_10k", "quantile_p50"],
    "L0":      ["huber_10k", "quantile_p40", "quantile_p50"],
    "L1":      ["huber_20k", "quantile_p30", "quantile_p40"],
    "L2":      ["quantile_p30", "huber_20k"],
    # Legacy segment labels — kept so old training scripts still work
    "SALARY_LIKE":      ["huber_10k", "quantile_p40"],
    "SME":              ["huber_20k", "quantile_p30"],
    "GIG_FREELANCE":    ["quantile_p30", "huber_20k"],
    "PASSIVE_INVESTOR": ["huber_10k", "quantile_p40"],
}

PERSONA_DEFAULT_LABELS: Dict[str, List[str]] = {
    "PAYROLL": ["raw", "robust"],
    "L0":      ["shrunk_composite", "robust"],
    "L1":      ["robust", "quantile"],
    "L2":      ["quantile", "robust"],
    # Legacy segment labels
    "SALARY_LIKE":      ["shrunk_composite", "robust"],
    "SME":              ["robust", "quantile"],
    "GIG_FREELANCE":    ["quantile", "robust"],
    "PASSIVE_INVESTOR": ["shrunk_composite", "robust"],
}


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

        lgb_params["objective"] = obj_fn   # LightGBM 4.x: custom obj goes in params
        self._model = lgb.train(
            lgb_params, dtrain,
            num_boost_round=self.n_estimators,
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
        persona_col: str = "persona",
        use_persona_defaults: bool = True,
    ):
        self.label_engineer = label_engineer
        self.feature_selector = feature_selector
        self.cv_folds = cv_folds
        self.lgb_losses = lgb_losses      # None → resolved per-persona if use_persona_defaults
        self.label_strategies = label_strategies  # None → resolved per-persona
        self.include_lstm = include_lstm
        self.include_tabpfn = include_tabpfn
        self.include_autogluon = include_autogluon
        self.max_rows_per_segment = max_rows_per_segment
        self.random_state = random_state
        self.persona_col = persona_col
        self.use_persona_defaults = use_persona_defaults

        # Results
        self.segment_results_: Dict[str, pd.DataFrame] = {}
        self.best_per_segment_: Dict[str, dict] = {}
        self.fitted_models_: Dict[str, Any] = {}   # persona → fitted model
        self.segment_feat_cols_: Dict[str, List[str]] = {}  # persona → feature list
        self.ml_personas_: set = set()             # personas with fitted ML models
        self.policy_floor_personas_: set = set()   # personas routed to policy floor

    def fit(
        self,
        X: pd.DataFrame,
        y_verified: pd.Series,
        segment_col: Optional[str] = None,
    ) -> "SegmentModelTrainer":
        """
        Search best (model, loss, label) per persona then refit on full persona data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix including the persona/segment column.
        y_verified : pd.Series
            Verified gross income (THB/month).
        segment_col : str, optional
            Column to group by. If None, uses self.persona_col (default "persona").
            Pass "segment" for backward compatibility with pre-Phase-2 pipelines.
        """
        pcol = segment_col or self.persona_col
        # Accept "segment" as an alias for "persona" if "persona" is missing
        if pcol not in X.columns and "segment" in X.columns:
            pcol = "segment"
        elif pcol not in X.columns and "persona" in X.columns:
            pcol = "persona"

        personas = X[pcol].unique()
        logger.info(
            f"SegmentModelTrainer: searching across {len(personas)} personas "
            f"(col='{pcol}'): {sorted(personas)}"
        )

        for seg in personas:
            # ── THIN: policy floor only — no ML model ────────────────────────
            if seg in _POLICY_FLOOR_PERSONAS:
                logger.info(
                    f"Persona {seg}: skipped (policy floor — "
                    f"regulatory minimum income applied downstream)"
                )
                self.policy_floor_personas_.add(seg)
                continue

            mask = X[pcol] == seg
            if mask.sum() < 200:
                logger.warning(f"Persona {seg}: only {mask.sum()} rows — skipping")
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

            # Per-persona search space (narrowed when use_persona_defaults=True)
            losses, label_strats = self._resolve_search_space(seg)

            logger.info(
                f"\nPersona {seg}: {len(X_seg):,} rows, {len(feat_cols)} features, "
                f"{len(losses)} losses × {len(label_strats)} label strategies"
            )
            self.segment_feat_cols_[seg] = feat_cols

            results = self._search_segment(X_seg, y_seg, feat_cols, seg,
                                           lgb_losses=losses,
                                           label_strategies=label_strats)
            self.segment_results_[seg] = pd.DataFrame(results).sort_values("cv_mae")

            best = self.segment_results_[seg].iloc[0]
            self.best_per_segment_[seg] = best.to_dict()

            logger.info(
                f"  Best for {seg}: model={best['model']}, "
                f"loss={best['loss']}, label={best['label']}, "
                f"cv_mae={best['cv_mae']:,.0f} THB"
            )

            # Refit best on full persona data
            self.fitted_models_[seg] = self._refit_best(
                X[mask], y_verified[mask], best, feat_cols, seg
            )
            self.ml_personas_.add(seg)

        logger.info(
            f"SegmentModelTrainer fitted: ML={sorted(self.ml_personas_)}  "
            f"PolicyFloor={sorted(self.policy_floor_personas_)}"
        )
        return self

    def predict(
        self, X: pd.DataFrame, segment_col: Optional[str] = None
    ) -> pd.Series:
        """
        Predict income using persona-specific best model.

        THIN customers and other policy-floor personas return NaN.
        Use predict_with_metadata() to also get income_estimate_type.
        """
        return self.predict_with_metadata(X, segment_col=segment_col)["income_estimate"]

    def predict_with_metadata(
        self, X: pd.DataFrame, segment_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Predict income and return estimate type alongside the estimate.

        Returns
        -------
        pd.DataFrame with columns:
          income_estimate     : float, THB/month (NaN for POLICY_FLOOR personas)
          income_estimate_type: str — one of:
            "ML_ESTIMATE"     : persona has a fitted ML model
            "POLICY_FLOOR"    : THIN persona — downstream applies regulatory minimum
            "UNSEEN_PERSONA"  : persona not seen during training
        """
        pcol = segment_col or self.persona_col
        if pcol not in X.columns and "segment" in X.columns:
            pcol = "segment"
        elif pcol not in X.columns and "persona" in X.columns:
            pcol = "persona"

        income = pd.Series(np.nan, index=X.index, name="income_estimate")
        est_type = pd.Series("UNSEEN_PERSONA", index=X.index, name="income_estimate_type")

        for seg, model in self.fitted_models_.items():
            mask = X[pcol] == seg
            if mask.any():
                feat_cols = self.segment_feat_cols_.get(seg)
                X_seg = X.loc[mask, feat_cols] if feat_cols else X[mask]
                income[mask] = model.predict(X_seg)
                est_type[mask] = "ML_ESTIMATE"

        # Mark policy-floor personas explicitly (income stays NaN)
        for seg in self.policy_floor_personas_:
            mask = X[pcol] == seg
            est_type[mask] = "POLICY_FLOOR"

        unseen_mask = est_type == "UNSEEN_PERSONA"
        if unseen_mask.any():
            unseen = (
                X.loc[unseen_mask, pcol].unique().tolist()
                if pcol in X.columns else []
            )
            logger.warning(
                f"predict_with_metadata(): {unseen_mask.sum()} customers have "
                f"personas not seen during training: {unseen}"
            )

        return pd.DataFrame({"income_estimate": income, "income_estimate_type": est_type})

    def results_summary(self) -> pd.DataFrame:
        """Return full search results across all segments."""
        rows = []
        for seg, df in self.segment_results_.items():
            df = df.copy()
            df["segment"] = seg
            rows.append(df)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # ── PRIVATE ──────────────────────────────────────────────────────────────

    def _resolve_search_space(
        self, persona: str
    ) -> Tuple[List[str], List[str]]:
        """
        Return (lgb_losses, label_strategies) for the given persona.

        If the caller supplied explicit lists at construction time, those are used.
        Otherwise (use_persona_defaults=True) the persona-specific defaults are used.
        Fallback: full default grid.
        """
        if self.lgb_losses is not None:
            losses = self.lgb_losses
        elif self.use_persona_defaults and persona in PERSONA_DEFAULT_LOSSES:
            losses = PERSONA_DEFAULT_LOSSES[persona]
        else:
            losses = self.DEFAULT_LGB_LOSSES

        if self.label_strategies is not None:
            labels = self.label_strategies
        elif self.use_persona_defaults and persona in PERSONA_DEFAULT_LABELS:
            labels = PERSONA_DEFAULT_LABELS[persona]
        else:
            labels = self.DEFAULT_LABEL_STRATEGIES

        return losses, labels

    def _search_segment(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feat_cols: List[str],
        segment: str,
        lgb_losses: Optional[List[str]] = None,
        label_strategies: Optional[List[str]] = None,
    ) -> List[dict]:
        # TimeSeriesSplit respects temporal ordering: validation fold is always
        # AFTER training fold. This prevents validating on the past.
        kf = TimeSeriesSplit(n_splits=self.cv_folds)
        results = []

        active_losses = lgb_losses or self.lgb_losses or self.DEFAULT_LGB_LOSSES
        active_labels = label_strategies or self.label_strategies or self.DEFAULT_LABEL_STRATEGIES

        seg_col = "persona" if "persona" in X.columns else "segment" if "segment" in X.columns else None

        for label_strat in active_labels:
            try:
                y_label = self.label_engineer.transform(
                    X, y, strategy=label_strat, segment_col=seg_col
                )
            except Exception as e:
                logger.debug(f"Label strategy {label_strat} failed: {e}")
                continue

            # ── LightGBM × losses ────────────────────────────────────────
            for loss_name in active_losses:
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
        kf: TimeSeriesSplit,
    ) -> float:
        obj_fn, eval_fn = LossRegistry.get(loss_name)
        maes = []
        for train_idx, val_idx in kf.split(X):
            X_tr, y_tr = X.iloc[train_idx].fillna(0), y_label.iloc[train_idx]
            X_val = X.iloc[val_idx].fillna(0)
            y_val_true = y_true.iloc[val_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr.values)
            cv_params = {"learning_rate": 0.05, "max_depth": 5, "num_leaves": 31,
                         "verbose": -1, "n_jobs": -1,
                         "objective": obj_fn}   # LightGBM 4.x: custom obj in params
            model = lgb.train(
                cv_params, dtrain, num_boost_round=200,
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
        kf: TimeSeriesSplit,
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
        """Refit the winning model on the full persona data."""
        seg_col = "persona" if "persona" in X.columns else "segment" if "segment" in X.columns else None
        y_label = self.label_engineer.transform(
            X, y, strategy=best["label"], segment_col=seg_col
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
