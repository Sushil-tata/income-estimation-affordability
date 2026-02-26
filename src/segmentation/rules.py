"""
Rule-Based Segmenter
────────────────────
Deterministic segment assignment for high-confidence cases:
  S1 — PAYROLL       : SCB payroll credit identified
  S2 — SALARY_LIKE   : Regular monthly credit, low variance, single dominant source

All other customers pass through to BehavioralClusterer.
"""

import pandas as pd
import numpy as np
from typing import Tuple


# Segment labels
PAYROLL = "PAYROLL"
SALARY_LIKE = "SALARY_LIKE"
UNASSIGNED = "UNASSIGNED"


class RuleBasedSegmenter:
    """
    Applies deterministic rules to identify PAYROLL and SALARY_LIKE customers.
    Customers not matching any rule are returned as UNASSIGNED for clustering.

    Parameters
    ----------
    payroll_flag_col : str
        Column name indicating SCB payroll credit flag (0/1).
    cv_threshold : float
        Max coefficient of variation of monthly credits to qualify as SALARY_LIKE.
    salary_min_months : int
        Min months with a recurring credit to qualify as SALARY_LIKE.
    dominant_share_threshold : float
        Min share of dominant credit source to qualify as SALARY_LIKE.
    """

    def __init__(
        self,
        payroll_flag_col: str = "has_payroll_credit",
        cv_threshold: float = 0.25,
        salary_min_months: int = 6,
        dominant_share_threshold: float = 0.60,
    ):
        self.payroll_flag_col = payroll_flag_col
        self.cv_threshold = cv_threshold
        self.salary_min_months = salary_min_months
        self.dominant_share_threshold = dominant_share_threshold

    def assign(self, df: pd.DataFrame) -> pd.Series:
        """
        Assign deterministic segments to each customer.

        Parameters
        ----------
        df : pd.DataFrame
            Customer-level monthly aggregate feature dataframe.
            Expected columns:
              - has_payroll_credit          : int (0/1)
              - cv_monthly_credit_12m       : float
              - months_with_salary_pattern  : int
              - dominant_credit_source_share: float
              - months_data_available       : int

        Returns
        -------
        pd.Series
            Segment label per customer: PAYROLL | SALARY_LIKE | UNASSIGNED
        """
        segments = pd.Series(UNASSIGNED, index=df.index, name="segment")

        # Rule 1: PAYROLL — SCB payroll flag
        payroll_mask = df[self.payroll_flag_col] == 1
        segments[payroll_mask] = PAYROLL

        # Rule 2: SALARY_LIKE — only for non-payroll customers
        non_payroll = ~payroll_mask
        salary_mask = (
            non_payroll
            & (df["cv_monthly_credit_12m"] <= self.cv_threshold)
            & (df["months_with_salary_pattern"] >= self.salary_min_months)
            & (df["dominant_credit_source_share"] >= self.dominant_share_threshold)
        )
        segments[salary_mask] = SALARY_LIKE

        return segments

    def get_segment_counts(self, segments: pd.Series) -> pd.DataFrame:
        """Return segment distribution summary."""
        counts = segments.value_counts().reset_index()
        counts.columns = ["segment", "count"]
        counts["pct"] = (counts["count"] / len(segments) * 100).round(2)
        return counts
