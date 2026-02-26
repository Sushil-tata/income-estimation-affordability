"""
Behavioral Clusterer
─────────────────────
Assigns behavioral income segments to UNASSIGNED customers using
rule-based scoring and optional clustering.

Segments:
  S3 — SME            : Business-pattern cash flows, high credit volumes
  S4 — GIG_FREELANCE  : Irregular credits, variable amounts, multiple sources
  S5 — PASSIVE        : Low activity, income from interest/FD/dividends
  S6 — THIN           : Insufficient history or transaction density
"""

import pandas as pd
import numpy as np
from typing import Optional


SME = "SME"
GIG_FREELANCE = "GIG_FREELANCE"
PASSIVE_INVESTOR = "PASSIVE_INVESTOR"
THIN = "THIN"
UNASSIGNED = "UNASSIGNED"


class BehavioralClusterer:
    """
    Rule-scored behavioral segmentation for customers not assigned by RuleBasedSegmenter.

    Scoring approach:
      Each customer receives a score for each candidate segment.
      The segment with the highest score is assigned.
      THIN takes precedence when data is insufficient.

    Parameters
    ----------
    thin_min_months : int
        Min months of data required; below this → THIN.
    thin_min_tx_monthly : float
        Min average monthly transactions; below this → THIN.
    sme_business_credit_ratio : float
        Business MCC credit share threshold for SME classification.
    """

    def __init__(
        self,
        thin_min_months: int = 6,
        thin_min_tx_monthly: float = 5.0,
        sme_business_credit_ratio: float = 0.40,
    ):
        self.thin_min_months = thin_min_months
        self.thin_min_tx_monthly = thin_min_tx_monthly
        self.sme_business_credit_ratio = sme_business_credit_ratio

    def assign(self, df: pd.DataFrame) -> pd.Series:
        """
        Assign behavioral segments to UNASSIGNED customers.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain UNASSIGNED customers only.
            Expected columns:
              - months_data_available           : int
              - transaction_count_avg_monthly   : float
              - business_mcc_credit_share       : float
              - avg_monthly_credit_12m                  : float
              - cv_monthly_credit_12m           : float
              - months_with_zero_credit         : int
              - investment_credit_frequency     : float
              - inflow_outflow_ratio            : float
              - credit_concentration_index      : float  (0=diversified, 1=concentrated)

        Returns
        -------
        pd.Series
            Behavioral segment label per customer.
        """
        segments = pd.Series(UNASSIGNED, index=df.index, name="segment")

        # --- Priority 0: THIN (data insufficient — must check first) ---
        thin_mask = (
            (df["months_data_available"] < self.thin_min_months)
            | (df["transaction_count_avg_monthly"] < self.thin_min_tx_monthly)
        )
        segments[thin_mask] = THIN

        remaining = segments == UNASSIGNED

        # --- Priority 1: SME ---
        sme_mask = remaining & (
            (df["business_mcc_credit_share"] >= self.sme_business_credit_ratio)
            | (
                (df["inflow_outflow_ratio"] > 1.5)          # High credit cycling
                & (df["avg_monthly_credit_12m"] > 50000)            # High volume (THB)
                & (df["credit_concentration_index"] < 0.5)  # Multiple payers
            )
        )
        segments[sme_mask] = SME
        remaining = segments == UNASSIGNED

        # --- Priority 2: PASSIVE_INVESTOR ---
        passive_mask = remaining & (
            (df["investment_credit_frequency"] > 0.3)        # Regular investment credits
            & (df["transaction_count_avg_monthly"] < 15)     # Low overall activity
            & (df["avg_monthly_credit_12m"] < df["avg_monthly_credit_12m"].quantile(0.5))
        )
        segments[passive_mask] = PASSIVE_INVESTOR
        remaining = segments == UNASSIGNED

        # --- Priority 3: GIG_FREELANCE (catch-all for irregular non-SME) ---
        gig_mask = remaining & (
            (df["cv_monthly_credit_12m"] > 0.25)             # High variance
            | (df["months_with_zero_credit"] > 2)            # Gaps in income
        )
        segments[gig_mask] = GIG_FREELANCE

        # Remaining UNASSIGNED → default to GIG_FREELANCE
        segments[segments == UNASSIGNED] = GIG_FREELANCE

        return segments

    def get_segment_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a scoring breakdown for interpretability/debugging.
        Useful for understanding borderline segment assignments.
        """
        scores = pd.DataFrame(index=df.index)

        scores["thin_flag"] = (
            (df["months_data_available"] < self.thin_min_months)
            | (df["transaction_count_avg_monthly"] < self.thin_min_tx_monthly)
        ).astype(int)

        scores["sme_score"] = (
            (df["business_mcc_credit_share"] >= self.sme_business_credit_ratio).astype(int) * 2
            + (df["inflow_outflow_ratio"] > 1.5).astype(int)
            + (df["credit_concentration_index"] < 0.5).astype(int)
        )

        scores["passive_score"] = (
            (df["investment_credit_frequency"] > 0.3).astype(int) * 2
            + (df["transaction_count_avg_monthly"] < 15).astype(int)
        )

        scores["gig_score"] = (
            (df["cv_monthly_credit_12m"] > 0.25).astype(int) * 2
            + (df["months_with_zero_credit"] > 2).astype(int)
        )

        return scores
