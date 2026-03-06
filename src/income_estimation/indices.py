"""
Composite Income Indices
─────────────────────────
Computes four behavioural indices from the customer feature matrix:

  SI  (Stability Index)      — regularity and predictability of income stream
  CI  (Concentration Index)  — single-source vs multi-source income structure
  VI  (Volatility Index)     — lumpiness and asymmetry of income distribution
  DDI (Data Density Index)   — richness of transaction history

Index definitions
─────────────────
  SI  = clip(1 / max(CV, cv_floor), 0, si_cap) × recurring_share
        High SI → PAYROLL-like regularity; low SI → irregular / gig income

  CI  = dominant_credit_source_share
        1 → single payer (salary); 0 → many payers (SME / portfolio)

  VI  = CV × clip(max(skewness, 0), 0, skew_cap) / skew_cap
        High VI → lumpy, spike-prone income (gig / informal economy)

  DDI = alpha × (months / months_norm) + (1-alpha) × (avg_tx / tx_norm)
        High DDI → rich history; low DDI → thin file

Normalisation
─────────────
  raw_norm = clip((raw - P5) / (P95 - P5), 0, 1)
  Scale parameters are fitted from the training population and reused at scoring.
  This makes normalised indices comparable across vintages.

Usage
─────
  ic = IndexComputer()
  ic.fit(train_features_df)
  scored = ic.transform(score_features_df)   # adds si, ci, vi, ddi, *_norm cols
"""

import numpy as np
import pandas as pd
import logging
import pickle
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

INDEX_COLS = ["si", "ci", "vi", "ddi"]
INDEX_NORM_COLS = ["si_norm", "ci_norm", "vi_norm", "ddi_norm"]


class IndexComputer:
    """
    Computes and normalises SI / CI / VI / DDI composite indices.

    Parameters
    ----------
    si_cv_floor : float
        Floor for CV in SI to prevent explosion when CV → 0.
        Default 0.05 → maximum raw SI = 20 × recurring_share.
    si_raw_cap : float
        Upper cap on raw SI before multiplying recurring_share. Default 20.
    vi_skew_cap : float
        Upper clip on skewness in VI computation. Default 5.
    ddi_months_norm : float
        Months benchmark for full DDI data component. Default 12.
    ddi_tx_norm : float
        Average monthly transactions benchmark for full density. Default 30.
    ddi_alpha : float
        Weight of months component in DDI (history breadth). Default 0.60.
    norm_lo_pct : float
        Lower training percentile for min-max normalisation. Default 5.
    norm_hi_pct : float
        Upper training percentile for min-max normalisation. Default 95.
    """

    def __init__(
        self,
        si_cv_floor: float = 0.05,
        si_raw_cap: float = 20.0,
        vi_skew_cap: float = 5.0,
        ddi_months_norm: float = 12.0,
        ddi_tx_norm: float = 30.0,
        ddi_alpha: float = 0.60,
        norm_lo_pct: float = 5.0,
        norm_hi_pct: float = 95.0,
    ):
        self.si_cv_floor = si_cv_floor
        self.si_raw_cap = si_raw_cap
        self.vi_skew_cap = vi_skew_cap
        self.ddi_months_norm = ddi_months_norm
        self.ddi_tx_norm = ddi_tx_norm
        self.ddi_alpha = ddi_alpha
        self.norm_lo_pct = norm_lo_pct
        self.norm_hi_pct = norm_hi_pct

        self._scale_lo: Optional[Dict[str, float]] = None
        self._scale_hi: Optional[Dict[str, float]] = None
        self.fitted_ = False

    # ── Raw index computation ────────────────────────────────────────────────

    def compute_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute raw (unscaled) SI, CI, VI, DDI for all customers.

        Required input columns (output of FeatureEngineer)
        --------------------------------------------------
          cv_monthly_credit_12m          : float  [0, ∞)
          recurring_to_total_credit_ratio: float  [0, 1]
          credit_skewness_12m            : float  (−∞, ∞)
          dominant_credit_source_share   : float  [0, 1]
          months_data_available          : int
          transaction_count_avg_monthly  : float

        Returns
        -------
        pd.DataFrame with columns: si, ci, vi, ddi (raw values, not normalised)
        """
        self._check_required_cols(df)
        out = pd.DataFrame(index=df.index)

        # ── SI: Stability Index ──────────────────────────────────────────
        # clip(1/max(CV, floor), 0, cap) × recurring_share
        # Floor prevents 1/CV blowing up for near-perfect payroll customers.
        cv = df["cv_monthly_credit_12m"].fillna(1.0).clip(lower=self.si_cv_floor)
        recurring_share = df["recurring_to_total_credit_ratio"].fillna(0.0).clip(0, 1)
        out["si"] = np.minimum(1.0 / cv, self.si_raw_cap) * recurring_share

        # ── CI: Concentration Index ──────────────────────────────────────
        # Single dominant-source share as HHI proxy.
        # Range: [0, 1]. High = single payer, Low = diversified.
        out["ci"] = df["dominant_credit_source_share"].fillna(0.5).clip(0, 1)

        # ── VI: Volatility Index ─────────────────────────────────────────
        # CV × normalised_skewness — captures both spread and asymmetry.
        # Only positive skewness counts (lumpy upside spikes are the risk signal).
        skewness = df["credit_skewness_12m"].fillna(0.0).clip(lower=0, upper=self.vi_skew_cap)
        out["vi"] = df["cv_monthly_credit_12m"].fillna(1.0) * (skewness / self.vi_skew_cap)

        # ── DDI: Data Density Index ──────────────────────────────────────
        # Weighted blend of history length and transaction frequency.
        months_frac = (
            df["months_data_available"].fillna(0) / self.ddi_months_norm
        ).clip(0, 1)
        tx_frac = (
            df["transaction_count_avg_monthly"].fillna(0) / self.ddi_tx_norm
        ).clip(0, 1)
        out["ddi"] = self.ddi_alpha * months_frac + (1 - self.ddi_alpha) * tx_frac

        return out

    # ── Normalisation ────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "IndexComputer":
        """
        Compute P5/P95 normalisation bounds from training data.

        Should be called once on the training population; the fitted scaler is
        then used to normalise both training and scoring populations for
        consistent index values across vintages.
        """
        raw = self.compute_raw(df)
        self._scale_lo = {
            col: float(np.percentile(raw[col].dropna(), self.norm_lo_pct))
            for col in INDEX_COLS
        }
        self._scale_hi = {
            col: float(np.percentile(raw[col].dropna(), self.norm_hi_pct))
            for col in INDEX_COLS
        }
        # Guard: ensure hi > lo to prevent division by zero
        for col in INDEX_COLS:
            if self._scale_hi[col] <= self._scale_lo[col]:
                self._scale_hi[col] = self._scale_lo[col] + 1e-6

        self.fitted_ = True

        logger.info(
            "IndexComputer fitted — normalisation bounds:\n"
            + "\n".join(
                f"  {col}: raw P{self.norm_lo_pct:.0f}={self._scale_lo[col]:.4f}  "
                f"P{self.norm_hi_pct:.0f}={self._scale_hi[col]:.4f}"
                for col in INDEX_COLS
            )
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add raw (si, ci, vi, ddi) and normalised (*_norm) index columns to df.

        Normalised values are clipped to [0, 1]; values outside the training
        P5–P95 range are floored/capped, not discarded.
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before transform().")

        raw = self.compute_raw(df)
        result = df.copy()

        for col in INDEX_COLS:
            result[col] = raw[col]
            lo, hi = self._scale_lo[col], self._scale_hi[col]
            result[f"{col}_norm"] = ((raw[col] - lo) / (hi - lo)).clip(0, 1)

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on df then return transformed df (convenience method)."""
        return self.fit(df).transform(df)

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"IndexComputer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "IndexComputer":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _check_required_cols(df: pd.DataFrame) -> None:
        required = [
            "cv_monthly_credit_12m",
            "recurring_to_total_credit_ratio",
            "credit_skewness_12m",
            "dominant_credit_source_share",
            "months_data_available",
            "transaction_count_avg_monthly",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"IndexComputer: missing required columns: {missing}")

    def summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return per-segment index summary for diagnostics.

        If df contains a 'segment' or 'persona' column, statistics are
        broken out by segment; otherwise overall statistics are returned.
        """
        transformed = self.transform(df)
        group_col = next(
            (c for c in ("persona", "segment") if c in transformed.columns), None
        )
        cols = INDEX_COLS + INDEX_NORM_COLS
        if group_col:
            return transformed.groupby(group_col)[cols].describe().round(3)
        return transformed[cols].describe().round(3)
