"""
Feature Engineer
────────────────
Builds the customer-level monthly aggregate feature set from raw
transaction data following the P&L framework:
  Credits   → Income signals
  Debits    → Commitments | Recurring | Lifestyle

Input  : Transaction-level dataframe (monthly aggregates preferred)
Output : Customer-level feature matrix
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Constructs the full feature set from monthly aggregate transaction data.

    Parameters
    ----------
    lookback_months : int
        Number of months of history to use.
    observation_date : str
        Reference date (YYYY-MM-DD) for lookback calculation.
    """

    def __init__(self, lookback_months: int = 12, observation_date: Optional[str] = None):
        self.lookback_months = lookback_months
        self.observation_date = pd.to_datetime(observation_date) if observation_date else None

    def build_features(self, monthly_agg: pd.DataFrame) -> pd.DataFrame:
        """
        Build full feature matrix from monthly aggregates.

        Parameters
        ----------
        monthly_agg : pd.DataFrame
            Monthly aggregated transaction data.
            Required columns:
              customer_id, year_month,
              total_credit_amount, total_debit_amount,
              recurring_credit_amount, irregular_credit_amount,
              investment_credit_amount,
              commitment_amount, recurring_expense_amount, lifestyle_amount,
              eom_balance, transaction_count,
              business_mcc_credit_share, dominant_credit_source_share,
              has_payroll_credit (0/1 per month)

        Returns
        -------
        pd.DataFrame
            Customer-level feature matrix (one row per customer).
        """
        logger.info("Building features from monthly aggregates...")

        features = {}

        grouped = monthly_agg.groupby("customer_id")

        # ── CREDITS ────────────────────────────────────────────────────────
        features.update(self._credit_features(grouped))

        # ── DEBITS / P&L CATEGORIES ────────────────────────────────────────
        features.update(self._debit_features(grouped))

        # ── BALANCE ────────────────────────────────────────────────────────
        features.update(self._balance_features(grouped))

        # ── SEGMENTATION SIGNALS ───────────────────────────────────────────
        features.update(self._segmentation_signals(grouped))

        # ── DERIVED / COMPOSITE ────────────────────────────────────────────
        feat_df = pd.DataFrame(features)
        feat_df = self._derived_features(feat_df)

        logger.info(f"Feature matrix: {feat_df.shape[0]:,} customers × {feat_df.shape[1]} features")
        return feat_df

    # ── PRIVATE METHODS ─────────────────────────────────────────────────────

    def _credit_features(self, grouped) -> dict:
        f = {}

        credits = grouped["total_credit_amount"]
        f["avg_monthly_credit_12m"] = credits.mean()
        f["max_monthly_credit_3m"] = grouped["total_credit_amount"].apply(
            lambda x: x.iloc[-3:].max() if len(x) >= 3 else x.max()
        )
        f["median_monthly_credit_12m"] = credits.median()
        f["cv_monthly_credit_12m"] = credits.apply(
            lambda x: x.std() / x.mean() if x.mean() > 0 and len(x) > 1 else np.nan
        ).fillna(1.0)  # 1.0 = worst-case (fully volatile) for customers with 0 or 1 month
        f["months_with_zero_credit"] = grouped["total_credit_amount"].apply(
            lambda x: (x == 0).sum()
        )
        f["months_data_available"] = grouped["year_month"].count()

        # Recurring vs irregular credits
        f["avg_recurring_credit_12m"] = grouped["recurring_credit_amount"].mean()
        f["avg_irregular_credit_12m"] = grouped["irregular_credit_amount"].mean()
        f["recurring_to_total_credit_ratio"] = (
            f["avg_recurring_credit_12m"] / f["avg_monthly_credit_12m"].replace(0, np.nan)
        ).fillna(0.0)
        f["cv_recurring_credit_12m"] = grouped["recurring_credit_amount"].apply(
            lambda x: x.std() / x.mean() if x.mean() > 0 and len(x) > 1 else np.nan
        ).fillna(1.0)

        # Recurring credit streak (consecutive months with recurring credit)
        f["recurring_credit_streak_months"] = grouped["recurring_credit_amount"].apply(
            lambda x: self._max_consecutive_positive(x)
        )

        # Investment credits
        f["avg_investment_credit_12m"] = grouped["investment_credit_amount"].mean()
        f["investment_credit_frequency"] = grouped["investment_credit_amount"].apply(
            lambda x: (x > 0).mean()
        )

        # Payroll flag
        f["has_payroll_credit"] = grouped["has_payroll_credit"].max()
        f["months_with_salary_pattern"] = grouped["has_payroll_credit"].sum()

        # Credit source concentration
        f["dominant_credit_source_share"] = grouped["dominant_credit_source_share"].mean()
        f["credit_concentration_index"] = grouped["dominant_credit_source_share"].mean()

        return f

    def _debit_features(self, grouped) -> dict:
        f = {}

        # ── Commitments ────────────────────────────────────────────────────
        f["avg_commitment_amount_12m"] = grouped["commitment_amount"].mean()
        f["commitment_regularity"] = grouped["commitment_amount"].apply(
            lambda x: (x > 0).mean()
        )

        # ── Recurring expenses ─────────────────────────────────────────────
        f["avg_recurring_expense_12m"] = grouped["recurring_expense_amount"].mean()
        f["recurring_expense_regularity"] = grouped["recurring_expense_amount"].apply(
            lambda x: (x > 0).mean()
        )

        # ── Lifestyle ──────────────────────────────────────────────────────
        f["avg_lifestyle_amount_12m"] = grouped["lifestyle_amount"].mean()

        # ── Total debits ───────────────────────────────────────────────────
        f["avg_total_debit_12m"] = grouped["total_debit_amount"].mean()
        f["cv_total_debit_12m"] = grouped["total_debit_amount"].apply(
            lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan
        )

        return f

    def _balance_features(self, grouped) -> dict:
        f = {}

        f["avg_eom_balance_3m"] = grouped["eom_balance"].apply(
            lambda x: x.iloc[-3:].mean() if len(x) >= 3 else x.mean()
        )
        f["avg_eom_balance_6m"] = grouped["eom_balance"].apply(
            lambda x: x.iloc[-6:].mean() if len(x) >= 6 else x.mean()
        )
        f["avg_eom_balance_12m"] = grouped["eom_balance"].mean()
        f["max_eom_balance_3m"] = grouped["eom_balance"].apply(
            lambda x: x.iloc[-3:].max() if len(x) >= 3 else x.max()
        )
        f["min_eom_balance_3m"] = grouped["eom_balance"].apply(
            lambda x: x.iloc[-3:].min() if len(x) >= 3 else x.min()
        )
        f["balance_cv_12m"] = grouped["eom_balance"].apply(
            lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan
        )
        f["months_below_1000_balance"] = grouped["eom_balance"].apply(
            lambda x: (x < 1000).sum()
        )
        f["balance_trend_slope"] = grouped["eom_balance"].apply(
            lambda x: self._linear_slope(x)
        )

        # Transaction density
        f["transaction_count_avg_monthly"] = grouped["transaction_count"].mean()

        return f

    def _segmentation_signals(self, grouped) -> dict:
        f = {}
        f["business_mcc_credit_share"] = grouped["business_mcc_credit_share"].mean()
        return f

    def _derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["net_monthly_flow_avg"] = df["avg_monthly_credit_12m"] - df["avg_total_debit_12m"]
        df["inflow_outflow_ratio"] = (
            df["avg_monthly_credit_12m"] / df["avg_total_debit_12m"].replace(0, np.nan)
        ).fillna(1.0)
        df["savings_rate_proxy"] = (
            df["net_monthly_flow_avg"] / df["avg_monthly_credit_12m"].replace(0, np.nan)
        ).fillna(0.0)
        df["commitment_ratio"] = (
            df["avg_commitment_amount_12m"] / df["avg_total_debit_12m"].replace(0, np.nan)
        ).fillna(0.0)
        df["lifestyle_ratio"] = (
            df["avg_lifestyle_amount_12m"] / df["avg_total_debit_12m"].replace(0, np.nan)
        ).fillna(0.0)
        df["balance_to_max_credit_ratio"] = (
            df["avg_eom_balance_3m"] / df["max_monthly_credit_3m"].replace(0, np.nan)
        ).fillna(0.0)
        df["income_to_obligation_ratio"] = (
            df["avg_recurring_credit_12m"] / df["avg_commitment_amount_12m"].replace(0, np.nan)
        ).fillna(0.0)
        df["financial_stress_index"] = (
            df["months_below_1000_balance"] / df["months_data_available"].replace(0, np.nan)
        ).fillna(0.0)
        return df

    @staticmethod
    def _max_consecutive_positive(series: pd.Series) -> int:
        """Count max consecutive months with positive value."""
        max_streak = streak = 0
        for val in series:
            if val > 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    @staticmethod
    def _linear_slope(series: pd.Series) -> float:
        """Compute linear trend slope over the series."""
        if len(series) < 2:
            return 0.0
        x = np.arange(len(series))
        try:
            slope = np.polyfit(x, series.values, 1)[0]
        except Exception:
            slope = 0.0
        return slope
