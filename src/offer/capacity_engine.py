"""
Layer 9A — Capacity Engine
───────────────────────────
Deterministically translate the Layer G affordability output into product-level
borrowing capacity: maximum EMI, maximum loan amount by tenor, maximum card line.

Inputs (from frozen Layer G — read-only)
─────────────────────────────────────────
  adjusted_income       : PolicyIncome (= income estimate, no haircut)
  adsc                  : Available Debt Servicing Capacity
                          = allowable_obligation − existing_obligations
  existing_obligations  : Monthly debt commitments already in place
  dscr_used             : Effective DSCR cap (min of persona_cap, bci_band_cap)

Outputs (new fields — never overwrite upstream fields)
───────────────────────────────────────────────────────
  max_obligation        : adjusted_income × dscr_used  [echoed for auditability]
  residual_capacity     : adsc  [echoed for auditability; = max_obligation − existing_obligations]
  max_emi               : residual_capacity (monthly capacity available for new obligation)
  max_loan_by_tenor     : dict  {tenor_months → max_loan_amount_thb}
  max_card_line         : float  residual_capacity × card_line_months_equivalent

Design notes
────────────
  • All arithmetic is deterministic. No model, no randomness.
  • Uses standard annuity present-value formula for loan sizing:
      PV = EMI × [(1+r)^n − 1] / [r(1+r)^n]
    where r = monthly_rate = annual_rate/12, n = tenor in months.
  • Card line proxy: residual_capacity × card_line_months_equivalent
    (configurable — typically 24–36 months of monthly capacity).
  • When adsc ≤ 0, max_emi = 0 and all loan amounts = 0.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CapacityEngine:
    """
    Layer 9A: Deterministic safe borrowing capacity computation.

    Parameters
    ----------
    product_tenors_months : list[int]
        Tenor options (months) to compute max loan amounts for.
        Default: [12, 24, 36, 48, 60]  [PROVISIONAL]
    reference_rate_annual : float
        Annual interest rate used in loan sizing.  [PROVISIONAL]
        Note: this is an internal sizing rate, not the customer offer rate.
    card_line_months_equivalent : int
        card_line = residual_capacity × this value.  [PROVISIONAL]
        Represents the number of months of EMI capacity that translates to a
        card credit limit (typically 24–36).
    """

    def __init__(
        self,
        product_tenors_months: Optional[List[int]] = None,
        reference_rate_annual: float = 0.18,
        card_line_months_equivalent: int = 30,
    ):
        self.product_tenors_months     = product_tenors_months or [12, 24, 36, 48, 60]
        self.reference_rate_annual     = reference_rate_annual
        self.card_line_months_equivalent = card_line_months_equivalent

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute(
        self,
        final_output: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute product-level borrowing capacity per customer.

        Parameters
        ----------
        final_output : pd.DataFrame
            Output from InferencePipeline.run() → final_output.
            Required columns: adjusted_income, adsc, existing_obligations, dscr_used.
            SPARSE/THIN customers are handled gracefully (capacity = 0).

        Returns
        -------
        pd.DataFrame
            Index matches final_output. New columns only — no upstream fields modified.
            Columns:
              max_obligation         : float  (echoed for audit)
              residual_capacity      : float  (echoed for audit)
              max_emi                : float  (≥ 0)
              max_card_line          : float  (≥ 0)
              max_loan_{n}m          : float  for each tenor n in product_tenors_months
        """
        out = pd.DataFrame(index=final_output.index)

        income    = self._safe_col(final_output, "adjusted_income",   0.0)
        adsc      = self._safe_col(final_output, "adsc",              0.0)
        dscr      = self._safe_col(final_output, "dscr_used",         0.40)
        existing  = self._safe_col(final_output, "existing_obligations", 0.0)

        # max_obligation: echo what Layer G computed (for audit transparency)
        out["max_obligation"]    = (income * dscr).clip(lower=0).round(0)

        # residual_capacity = adsc (echo from Layer G — do NOT recompute independently)
        # adsc was already computed as allowable_obligation − existing_obligations.
        residual = adsc.clip(lower=0)
        out["residual_capacity"] = residual.round(0)

        # max_emi: the maximum monthly obligation this customer can take on
        out["max_emi"] = residual.clip(lower=0).round(0)

        # max_loan by tenor
        monthly_rate = self.reference_rate_annual / 12.0
        for tenor in self.product_tenors_months:
            af  = self._annuity_factor(monthly_rate, tenor)
            col = f"max_loan_{tenor}m"
            out[col] = (residual.clip(lower=0) * af).round(0)

        # max_card_line: proxy for credit card limit
        out["max_card_line"] = (
            residual.clip(lower=0) * self.card_line_months_equivalent
        ).round(0)

        logger.info(
            f"CapacityEngine: {len(out):,} customers | "
            f"positive_capacity={int((out['max_emi'] > 0).sum()):,} | "
            f"zero_capacity={int((out['max_emi'] <= 0).sum()):,}"
        )
        return out

    # ── Statics ────────────────────────────────────────────────────────────────

    @staticmethod
    def _annuity_factor(monthly_rate: float, tenor_months: int) -> float:
        """
        Present-value annuity factor.

        PV = EMI × [(1+r)^n − 1] / [r × (1+r)^n]
        → annuity_factor = [(1+r)^n − 1] / [r × (1+r)^n]

        When rate = 0: annuity_factor = n (simple sum).
        """
        if monthly_rate == 0.0 or monthly_rate < 1e-9:
            return float(tenor_months)
        r = monthly_rate
        n = tenor_months
        pv_factor = ((1 + r) ** n - 1) / (r * (1 + r) ** n)
        return float(pv_factor)

    @staticmethod
    def _safe_col(df: pd.DataFrame, col: str, default: float) -> pd.Series:
        """Return column if present, else fill with default."""
        if col in df.columns:
            return df[col].fillna(default).astype(float)
        logger.warning(f"CapacityEngine: column '{col}' not found — using default {default}")
        return pd.Series(default, index=df.index, dtype=float)

    @classmethod
    def from_config(cls, cfg: dict) -> "CapacityEngine":
        """Construct from 'offer_optimization' section of config.yaml."""
        oc = cfg.get("offer_optimization", {})
        return cls(
            product_tenors_months=oc.get("product_tenors_months", [12, 24, 36, 48, 60]),
            reference_rate_annual=oc.get("reference_rate_annual", 0.18),
            card_line_months_equivalent=oc.get("card_line_months_equivalent", 30),
        )
