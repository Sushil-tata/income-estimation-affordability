"""
Persona Router  (Phase 3)
──────────────────────────
Two-stage supervised classifier trained on K-means pseudo-labels (Phase 2).

Why a supervised router over K-means?
  1. Speed  : O(1) LightGBM tree traversal vs O(k·d) distance computation.
  2. Calibration : GBT predict_proba is better calibrated than RBF-softmax.
  3. Richness: can use the full 80-feature matrix, not just 4 composite indices.
  4. Boundary quality: GBT learns non-convex persona boundaries; K-means
     assumes spherical clusters in index space.

Training signal
───────────────
  K-means (Phase 2) produces pseudo-labels for all non-PAYROLL customers.
  These pseudo-labels train two LightGBM models:

  Stage 1 — THIN binary classifier
    Input : sparse features available even for < 6m thin-file customers.
    Target: is_thin  (1 = THIN, 0 = not THIN)
    Purpose: replace the hard data_tier == "THIN" rule with a soft learned gate.
             Catches behaviorally-thin customers (≥ 6 months but very sparse).

  Stage 2 — L0 / L1 / L2 multiclass classifier
    Input : composite indices + full behavioral feature set.
    Target: persona label  (L0=0, L1=1, L2=2)
    Purpose: replace K-means hard assignment with better-calibrated probabilities.

Inference protocol
──────────────────
  For each non-PAYROLL customer:
    1. Stage 1 → P(THIN).  If P(THIN) ≥ thin_threshold → persona = THIN.
    2. Stage 2 (non-THIN only) → P(L0), P(L1), P(L2).  argmax → persona label.
    3. persona_confidence = max(P(L0), P(L1), P(L2)).
       If confidence < moe_threshold → blend top-2 models downstream.

PAYROLL customers bypass the router entirely (handled by SegmentationPipeline).
"""

import numpy as np
import pandas as pd
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb

logger = logging.getLogger(__name__)

# ── Feature catalogues ───────────────────────────────────────────────────────

# Stage 1: minimal features available for ALL customers (including THIN < 6m)
STAGE1_FEATURES: List[str] = [
    "months_data_available",
    "transaction_count_avg_monthly",
    "avg_monthly_credit_12m",
    "cv_monthly_credit_12m",
    "has_payroll_credit",
    "months_with_zero_credit",
    "active_month_ratio_12m",           # Phase 1
    "recurring_to_total_credit_ratio",
    "dormancy_gap_max",                  # Phase 1
]

# Stage 2: full features for non-THIN customers (composite indices must be pre-computed)
STAGE2_FEATURES: List[str] = [
    # ── Composite indices (Phase 2 — must be pre-computed by IndexComputer) ──
    "si_norm", "ci_norm", "vi_norm", "ddi_norm",
    "si", "ci", "vi", "ddi",
    # ── Stability / regularity ───────────────────────────────────────────────
    "recurring_to_total_credit_ratio",
    "cv_recurring_credit_12m",
    "recurring_credit_streak_months",
    "salary_periodicity_confidence",    # Phase 1
    "regularity_score",                 # Phase 1
    "fixed_amount_similarity",          # Phase 1
    "active_month_ratio_12m",           # Phase 1
    "dormancy_gap_max",                 # Phase 1
    # ── Concentration / source diversity ────────────────────────────────────
    "dominant_credit_source_share",
    "credit_concentration_index",
    "business_mcc_credit_share",
    # ── Volatility / shape ───────────────────────────────────────────────────
    "credit_skewness_12m",              # Phase 1
    "credit_kurtosis_12m",              # Phase 1
    "credit_p95_p50_ratio",             # Phase 1
    "credit_spike_ratio",               # Phase 1
    "mom_growth_volatility",            # Phase 1
    # ── Seasonality ──────────────────────────────────────────────────────────
    "top2_month_inflow_share",          # Phase 1
    "top3_month_inflow_share",          # Phase 1
    "seasonality_index",                # Phase 1
    "peak_trough_ratio",                # Phase 1
    "rolling_3m_vs_12m_ratio",          # Phase 1
    # ── Income level / investment ────────────────────────────────────────────
    "avg_monthly_credit_12m",
    "cv_monthly_credit_12m",
    "avg_recurring_credit_12m",
    "investment_credit_frequency",
    "months_data_available",
    # ── Cash-flow dynamics ───────────────────────────────────────────────────
    "retention_ratio_6m",               # Phase 1
    "pass_through_score",               # Phase 1
    "inflow_outflow_velocity",          # Phase 1
    "usable_income_proxy",              # Phase 1
    "liquidity_buffer_ratio",           # Phase 1
]

# Canonical class encoding for Stage 2
PERSONA_CLASSES: List[str] = ["L0", "L1", "L2"]
_PERSONA_TO_INT: Dict[str, int] = {p: i for i, p in enumerate(PERSONA_CLASSES)}
_INT_TO_PERSONA: Dict[int, str] = {i: p for p, i in _PERSONA_TO_INT.items()}


class PersonaRouter:
    """
    Two-stage supervised persona router.

    Parameters
    ----------
    thin_threshold : float
        P(THIN) threshold above which a customer is assigned THIN. Default 0.50.
    n_estimators_s1 : int
        LightGBM trees for Stage 1 (THIN binary). Default 200.
    n_estimators_s2 : int
        LightGBM trees for Stage 2 (L0/L1/L2 multiclass). Default 300.
    learning_rate : float
        LightGBM learning rate (shared). Default 0.05.
    max_depth : int
        Max tree depth. Default 5.
    random_state : int
        Seed. Default 42.
    """

    def __init__(
        self,
        thin_threshold: float = 0.50,
        n_estimators_s1: int = 200,
        n_estimators_s2: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        random_state: int = 42,
    ):
        self.thin_threshold = thin_threshold
        self.n_estimators_s1 = n_estimators_s1
        self.n_estimators_s2 = n_estimators_s2
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

        self._model_s1: Optional[lgb.LGBMClassifier] = None
        self._model_s2: Optional[lgb.LGBMClassifier] = None
        self._s1_cols: Optional[List[str]] = None   # features actually used in s1
        self._s2_cols: Optional[List[str]] = None   # features actually used in s2
        self.fitted_ = False

    # ── Fitting ─────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        persona_labels: pd.Series,
    ) -> "PersonaRouter":
        """
        Fit both stages using K-means pseudo-labels as targets.

        Parameters
        ----------
        df : pd.DataFrame
            Non-PAYROLL customers with all features + composite indices.
            Must include THIN customers (they are Stage 1 training data).
        persona_labels : pd.Series
            Pseudo-labels from PersonaClusterer + THIN gate.
            Values: {"THIN", "L0", "L1", "L2"}.
            Index must match df.index.
        """
        logger.info(
            f"PersonaRouter.fit(): {len(df):,} customers  "
            f"label dist: {persona_labels.value_counts().to_dict()}"
        )

        # ── Stage 1: THIN binary ─────────────────────────────────────────────
        y1 = (persona_labels == "THIN").astype(int)
        self._s1_cols = self._available(df, STAGE1_FEATURES)
        X1 = df[self._s1_cols].fillna(0).values

        self._model_s1 = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=self.n_estimators_s1,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=31,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=-1,
        )
        self._model_s1.fit(X1, y1)
        thin_acc = (self._model_s1.predict(X1) == y1).mean()
        logger.info(
            f"  Stage 1 THIN binary: {self._s1_cols} "
            f"| train accuracy={thin_acc:.3f}  "
            f"| THIN prevalence={y1.mean():.3f}"
        )

        # ── Stage 2: L0/L1/L2 multiclass (non-THIN only) ────────────────────
        non_thin_mask = persona_labels != "THIN"
        df2 = df[non_thin_mask]
        y2_raw = persona_labels[non_thin_mask]

        # Encode labels: L0→0, L1→1, L2→2
        y2 = y2_raw.map(_PERSONA_TO_INT)
        unknown = y2.isna()
        if unknown.any():
            logger.warning(
                f"Stage 2: {unknown.sum()} non-THIN rows have unknown label "
                f"{y2_raw[unknown].unique()} — dropped."
            )
            df2 = df2[~unknown]
            y2 = y2[~unknown].astype(int)

        self._s2_cols = self._available(df2, STAGE2_FEATURES)
        X2 = df2[self._s2_cols].fillna(0).values

        self._model_s2 = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=self.n_estimators_s2,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=31,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=-1,
        )
        self._model_s2.fit(X2, y2.values)
        s2_acc = (self._model_s2.predict(X2) == y2.values).mean()
        logger.info(
            f"  Stage 2 multiclass: {len(self._s2_cols)} features "
            f"| train accuracy={s2_acc:.3f}"
        )

        self.fitted_ = True
        return self

    # ── Prediction ──────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Assign persona labels to non-PAYROLL customers.

        Returns
        -------
        pd.Series of {"THIN", "L0", "L1", "L2"} with index matching df.
        """
        self._check_fitted()
        personas = pd.Series("THIN", index=df.index, name="persona")

        # Stage 1: thin probability
        thin_prob = self._thin_prob(df)
        non_thin_mask = thin_prob < self.thin_threshold

        if non_thin_mask.any():
            X2 = df.loc[non_thin_mask, self._s2_cols].fillna(0).values
            cls_ids = self._model_s2.predict(X2)
            personas[non_thin_mask] = [_INT_TO_PERSONA[int(c)] for c in cls_ids]

        return personas

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return persona probabilities for non-PAYROLL customers.

        Returns
        -------
        pd.DataFrame with columns L0_prob, L1_prob, L2_prob.
        Rows where Stage 1 predicts THIN have NaN in all three columns.
        """
        self._check_fitted()
        prob_cols = [f"{p}_prob" for p in PERSONA_CLASSES]
        probs = pd.DataFrame(
            np.nan,
            index=df.index,
            columns=prob_cols,
            dtype=float,
        )

        thin_prob = self._thin_prob(df)
        non_thin_mask = thin_prob < self.thin_threshold

        if non_thin_mask.any():
            X2 = df.loc[non_thin_mask, self._s2_cols].fillna(0).values
            raw_probs = self._model_s2.predict_proba(X2)   # (n, 3), cols = L0,L1,L2
            for i, persona in enumerate(PERSONA_CLASSES):
                probs.loc[non_thin_mask, f"{persona}_prob"] = raw_probs[:, i]

        return probs

    def predict_full(
        self, df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
        """
        Convenience: return (persona_labels, proba_df, thin_prob_series) in one call.

        Avoids calling Stage 1 twice.
        """
        self._check_fitted()

        thin_prob = self._thin_prob(df)
        non_thin_mask = thin_prob < self.thin_threshold

        personas = pd.Series("THIN", index=df.index, name="persona")
        prob_cols = [f"{p}_prob" for p in PERSONA_CLASSES]
        probs = pd.DataFrame(np.nan, index=df.index, columns=prob_cols, dtype=float)

        if non_thin_mask.any():
            X2 = df.loc[non_thin_mask, self._s2_cols].fillna(0).values
            cls_ids = self._model_s2.predict(X2)
            raw_probs = self._model_s2.predict_proba(X2)

            personas[non_thin_mask] = [_INT_TO_PERSONA[int(c)] for c in cls_ids]
            for i, persona in enumerate(PERSONA_CLASSES):
                probs.loc[non_thin_mask, f"{persona}_prob"] = raw_probs[:, i]

        return personas, probs, thin_prob

    # ── Feature importance ───────────────────────────────────────────────────

    def feature_importance(self, stage: int = 2) -> pd.DataFrame:
        """
        Return LightGBM feature importances for Stage 1 or Stage 2.

        Parameters
        ----------
        stage : int
            1 for THIN binary, 2 for L0/L1/L2 multiclass. Default 2.
        """
        self._check_fitted()
        if stage == 1:
            cols, model = self._s1_cols, self._model_s1
        else:
            cols, model = self._s2_cols, self._model_s2

        imp = pd.DataFrame({
            "feature": cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return imp

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"PersonaRouter saved to {path}")

    @classmethod
    def load(cls, path: str) -> "PersonaRouter":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _thin_prob(self, df: pd.DataFrame) -> np.ndarray:
        """Return P(THIN) for each row. Shape: (n,)."""
        X1 = df[self._s1_cols].fillna(0).values
        return self._model_s1.predict_proba(X1)[:, 1]   # P(class=1) = P(THIN)

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("PersonaRouter must be fitted before predict.")

    @staticmethod
    def _available(df: pd.DataFrame, feature_list: List[str]) -> List[str]:
        """Return only those features that exist in df."""
        avail = [f for f in feature_list if f in df.columns]
        missing = [f for f in feature_list if f not in df.columns]
        if missing:
            logger.debug(f"PersonaRouter: {len(missing)} features not in df (skipped): {missing}")
        return avail
