"""
Rule-Based Segmenter
────────────────────
Deterministic segment assignment for the single highest-confidence case:
  PAYROLL — SCB payroll credit identified (verified employer deposit).

All other customers (including historically-labelled SALARY_LIKE) pass
to PersonaClusterer, which assigns L0 / L1 / L2 via composite indices.

Phase 2 change: SALARY_LIKE rule removed.
  Rationale: the rule-layer SALARY_LIKE bucket was too coarse. Customers
  matching the old rule are now captured by L0 (highest SI centroid) which
  is richer, data-driven, and more inclusive for the Thai market.
  The SALARY_LIKE constant is kept for backward compatibility.
"""

import pandas as pd
from typing import Optional


# Persona/segment label constants
PAYROLL = "PAYROLL"
SALARY_LIKE = "SALARY_LIKE"   # kept for backward compatibility; no longer assigned
UNASSIGNED = "UNASSIGNED"


class RuleBasedSegmenter:
    """
    Applies the single deterministic PAYROLL rule.

    Customers with verified SCB payroll credits are bypassed from the
    clustering model and assigned PAYROLL directly.
    All other customers are returned as UNASSIGNED for PersonaClusterer.

    Parameters
    ----------
    payroll_flag_col : str
        Column name for the SCB payroll credit flag (0/1). Default "has_payroll_credit".
    """

    def __init__(self, payroll_flag_col: str = "has_payroll_credit", **_ignored):
        # **_ignored absorbs deprecated params (cv_threshold, salary_min_months,
        # dominant_share_threshold) from old callers so they don't crash.
        self.payroll_flag_col = payroll_flag_col

    def assign(self, df: pd.DataFrame) -> pd.Series:
        """
        Assign PAYROLL to confirmed payroll customers; UNASSIGNED to all others.

        Parameters
        ----------
        df : pd.DataFrame
            Customer-level feature dataframe.
            Required: has_payroll_credit (0/1)

        Returns
        -------
        pd.Series of {PAYROLL, UNASSIGNED}
        """
        segments = pd.Series(UNASSIGNED, index=df.index, name="segment")

        if self.payroll_flag_col in df.columns:
            payroll_mask = df[self.payroll_flag_col] == 1
            segments[payroll_mask] = PAYROLL
        else:
            import logging
            logging.getLogger(__name__).warning(
                f"RuleBasedSegmenter: '{self.payroll_flag_col}' not in df — "
                f"no PAYROLL assignments made."
            )

        return segments

    def get_segment_counts(self, segments: pd.Series) -> pd.DataFrame:
        """Return segment distribution summary."""
        counts = segments.value_counts().reset_index()
        counts.columns = ["segment", "count"]
        counts["pct"] = (counts["count"] / len(segments) * 100).round(2)
        return counts
