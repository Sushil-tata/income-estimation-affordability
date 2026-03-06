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

    # Data tier thresholds (months of history)
    TIER_12M_MIN = 12    # Full feature set, full confidence
    TIER_9M_MIN  = 9     # Most features, annualised, BCI data_richness capped at 0.75
    TIER_6M_MIN  = 6     # Reduced features, BCI data_richness capped at 0.50
    TIER_THIN    = 0     # < 6 months → THIN, no ML model, policy floor only

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

        # ── PHASE 1: Extended feature groups ───────────────────────────────
        features.update(self._recurring_structure_features(grouped))
        features.update(self._volatility_features(grouped))
        features.update(self._seasonality_features(grouped))
        features.update(self._short_window_features(grouped))
        features.update(self._regularity_features(grouped))

        # ── DERIVED / COMPOSITE ────────────────────────────────────────────
        feat_df = pd.DataFrame(features)
        feat_df = self._derived_features(feat_df)

        # ── DATA WINDOW TIER ───────────────────────────────────────────────
        # Assign data_tier based on months of history available.
        # Downstream components (BCI, clustering) use this to cap confidence
        # and to bypass full feature computation for thin-history customers.
        feat_df["data_tier"] = feat_df["months_data_available"].apply(
            self._assign_data_tier
        )

        # Reset index so customer_id becomes a column (consistent with
        # generate_sample._aggregate_to_customer_features output)
        feat_df = feat_df.reset_index()

        logger.info(f"Feature matrix: {feat_df.shape[0]:,} customers × {feat_df.shape[1]} features")
        tier_counts = feat_df["data_tier"].value_counts().to_dict()
        logger.info(f"Data tiers: {tier_counts}")
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
        # Note: credit_concentration_index is intentionally distinct from
        # dominant_credit_source_share — it will be replaced by a true HHI
        # proxy in Phase 1 feature expansion. For now use the same source
        # but keep separate columns so Phase 1 can overwrite only one.
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

        # Phase 1 extensions
        f["min_balance_to_mean_ratio"] = grouped["eom_balance"].apply(
            lambda x: x.min() / x.mean() if x.mean() > 0 else 0.0
        ).fillna(0.0)
        f["months_negative_balance"] = grouped["eom_balance"].apply(
            lambda x: (x < 0).sum()
        )
        f["balance_volatility_6m"] = grouped["eom_balance"].apply(
            lambda x: (x.iloc[-6:].std() / abs(x.iloc[-6:].mean()))
            if len(x) >= 6 and x.iloc[-6:].mean() != 0 else np.nan
        ).fillna(1.0)

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

        # ── Phase 1: Balance extended ────────────────────────────────────
        df["liquidity_buffer_ratio"] = (
            df["avg_eom_balance_6m"] / df["avg_monthly_credit_12m"].replace(0, np.nan)
        ).fillna(0.0)

        # ── Phase 1: 6-month credit-expense derived ───────────────────────
        df["retention_ratio_6m"] = (
            (df["avg_monthly_credit_6m"] - df["avg_monthly_debit_6m"])
            / df["avg_monthly_credit_6m"].replace(0, np.nan)
        ).fillna(0.0).clip(-1, 1)

        # 3-month retention ratio — used alongside 6M for PT income robustness.
        # min(retention_ratio_6m, retention_ratio_3m) ensures the more
        # conservative window governs the PT usable income estimate.
        df["retention_ratio_3m"] = (
            (df["median_credit_3m"] - df["avg_monthly_debit_3m"])
            / df["median_credit_3m"].replace(0, np.nan)
        ).fillna(0.0).clip(-1, 1)

        df["pass_through_score"] = (1 - df["retention_ratio_6m"]).clip(0, 1)

        df["end_balance_ratio_6m"] = (
            df["avg_eom_balance_6m"] / df["avg_monthly_credit_12m"].replace(0, np.nan)
        ).fillna(0.0)

        df["recurring_debit_share"] = (
            (df["avg_commitment_amount_12m"] + df["avg_recurring_expense_12m"])
            / df["avg_total_debit_12m"].replace(0, np.nan)
        ).fillna(0.0).clip(0, 1)

        df["usable_income_proxy"] = (
            df["avg_monthly_credit_12m"]
            * df["recurring_to_total_credit_ratio"].clip(0, 1)
            * (1 - df["recurring_debit_share"])
        )

        df["churn_intensity"] = (
            (df["avg_monthly_credit_6m"] - df["avg_monthly_debit_6m"]).clip(lower=0)
            / df["avg_monthly_credit_12m"].replace(0, np.nan)
        ).fillna(0.0)

        df["inflow_outflow_velocity"] = (
            (df["avg_monthly_credit_6m"] - df["avg_monthly_debit_6m"])
            / df["avg_monthly_credit_6m"].replace(0, np.nan)
        ).fillna(0.0)

        df["debit_credit_ratio_6m"] = (
            df["median_debit_6m"] / df["median_credit_6m"].replace(0, np.nan)
        ).fillna(1.0)

        # ── Phase 1: L0 regularity ────────────────────────────────────────
        df["regularity_score"] = (1 - df["cv_monthly_credit_12m"]).clip(0, 1)

        return df

    # ── PHASE 1: New feature group methods ─────────────────────────────────

    def _recurring_structure_features(self, grouped) -> dict:
        """Recurring income stream structure — 5 features + 2 6m helper columns."""
        f = {}
        f["median_recurring_credit_12m"] = grouped["recurring_credit_amount"].median()

        f["recurring_stream_survival_ratio"] = grouped["recurring_credit_amount"].apply(
            lambda x: (x > x.median() * 0.5).mean() if x.median() > 0 else 0.0
        )

        f["recurring_credit_deviation_mom"] = grouped["recurring_credit_amount"].apply(
            lambda x: x.diff().abs().median() / x.median()
            if len(x) > 1 and x.median() > 0 else np.nan
        ).fillna(1.0)

        # Fraction of months where recurring credit is within 10% of its own median
        # — approximation of fixed-period payroll periodicity
        f["salary_periodicity_confidence"] = grouped["recurring_credit_amount"].apply(
            lambda x: (np.abs(x - x.median()) <= x.median() * 0.10).mean()
            if x.median() > 0 else 0.0
        )

        f["active_month_ratio_12m"] = grouped["total_credit_amount"].apply(
            lambda x: (x > 0).mean()
        )

        # 6-month window helpers used by multiple derived features
        f["avg_monthly_credit_6m"] = grouped["total_credit_amount"].apply(
            lambda x: x.iloc[-6:].mean() if len(x) >= 6 else x.mean()
        )
        f["avg_monthly_debit_6m"] = grouped["total_debit_amount"].apply(
            lambda x: x.iloc[-6:].mean() if len(x) >= 6 else x.mean()
        )
        return f

    def _volatility_features(self, grouped) -> dict:
        """Income volatility shape features — 6 features."""
        f = {}
        f["credit_skewness_12m"] = grouped["total_credit_amount"].apply(
            lambda x: float(x.skew()) if len(x) >= 4 else 0.0
        ).fillna(0.0)

        f["credit_kurtosis_12m"] = grouped["total_credit_amount"].apply(
            lambda x: float(x.kurt()) if len(x) >= 4 else 0.0
        ).fillna(0.0)

        f["credit_p95_p50_ratio"] = grouped["total_credit_amount"].apply(
            lambda x: np.percentile(x, 95) / np.median(x) if np.median(x) > 0 else 1.0
        ).fillna(1.0)

        # Fraction of months where credit exceeds 2× its own median
        f["credit_spike_ratio"] = grouped["total_credit_amount"].apply(
            lambda x: (x > x.median() * 2).mean() if x.median() > 0 else 0.0
        )

        f["mom_growth_volatility"] = grouped["total_credit_amount"].apply(
            lambda x: x.pct_change().dropna().std() if len(x) > 2 else np.nan
        ).fillna(1.0).clip(0, 5)

        # Worst single-month credit drop (as pct of prior month) in last 6m
        f["max_month_drop_6m"] = grouped["total_credit_amount"].apply(
            lambda x: (-(x.iloc[-6:].pct_change().dropna())).clip(lower=0).max()
            if len(x) >= 3 else 0.0
        ).fillna(0.0)
        return f

    def _seasonality_features(self, grouped) -> dict:
        """Credit seasonality and temporal concentration — 5 features."""
        f = {}
        # Max/min ratio of positive months — captures seasonal business swings
        f["seasonality_index"] = grouped["total_credit_amount"].apply(
            lambda x: x.max() / x[x > 0].min()
            if (x > 0).any() and x[x > 0].min() > 0 else 1.0
        ).fillna(1.0).clip(1, 20)

        f["peak_trough_ratio"] = grouped["total_credit_amount"].apply(
            lambda x: x.max() / x.median() if x.median() > 0 else 1.0
        ).fillna(1.0)

        # Temporal income concentration: fraction of annual total in top months
        f["top2_month_inflow_share"] = grouped["total_credit_amount"].apply(
            lambda x: x.nlargest(2).sum() / x.sum() if x.sum() > 0 else np.nan
        ).fillna(1.0)

        f["top3_month_inflow_share"] = grouped["total_credit_amount"].apply(
            lambda x: x.nlargest(3).sum() / x.sum() if x.sum() > 0 else np.nan
        ).fillna(1.0)

        # Recent 3m vs full 12m — detects ramp-up or income deterioration
        f["rolling_3m_vs_12m_ratio"] = grouped["total_credit_amount"].apply(
            lambda x: x.iloc[-3:].median() / x.median()
            if len(x) >= 3 and x.median() > 0 else 1.0
        ).fillna(1.0)
        return f

    def _short_window_features(self, grouped) -> dict:
        """Short-window (3m / 6m) credit and debit statistics — 5 features."""
        f = {}
        f["median_credit_3m"] = grouped["total_credit_amount"].apply(
            lambda x: x.iloc[-3:].median() if len(x) >= 3 else x.median()
        )
        f["median_credit_6m"] = grouped["total_credit_amount"].apply(
            lambda x: x.iloc[-6:].median() if len(x) >= 6 else x.median()
        )
        f["credit_cv_6m"] = grouped["total_credit_amount"].apply(
            lambda x: x.iloc[-6:].std() / x.iloc[-6:].mean()
            if len(x) >= 6 and x.iloc[-6:].mean() > 0 else np.nan
        ).fillna(1.0)
        f["credit_std_6m"] = grouped["total_credit_amount"].apply(
            lambda x: x.iloc[-6:].std() if len(x) >= 6 else x.std()
        ).fillna(0.0)
        f["median_debit_6m"] = grouped["total_debit_amount"].apply(
            lambda x: x.iloc[-6:].median() if len(x) >= 6 else x.median()
        )
        f["avg_monthly_debit_3m"] = grouped["total_debit_amount"].apply(
            lambda x: x.iloc[-3:].mean() if len(x) >= 3 else x.mean()
        )
        f["avg_recurring_credit_6m"] = grouped["recurring_credit_amount"].apply(
            lambda x: x.iloc[-6:].mean() if len(x) >= 6 else x.mean()
        )
        return f

    def _regularity_features(self, grouped) -> dict:
        """Temporal regularity of income stream — 3 features."""
        f = {}
        # Fraction of months where total credit is within 10% of its median
        # (fixed-income proxy, distinct from salary_periodicity_confidence
        #  which uses recurring_credit_amount)
        f["fixed_amount_similarity"] = grouped["total_credit_amount"].apply(
            lambda x: (np.abs(x - x.median()) <= x.median() * 0.10).mean()
            if x.median() > 0 else 0.0
        )
        f["income_slope_6m"] = grouped["total_credit_amount"].apply(
            lambda x: self._linear_slope(x.iloc[-6:]) if len(x) >= 3 else 0.0
        )
        f["dormancy_gap_max"] = grouped["total_credit_amount"].apply(
            lambda x: self._max_gap_months(x)
        )
        return f

    @staticmethod
    def _assign_data_tier(months: int) -> str:
        """
        Classify a customer into a data confidence tier based on history length.

        Tiers
        -----
        12M  : ≥ 12 months — full feature set, full model confidence
        9M   : 9–11 months — features computed, annualised where needed,
               BCI data_richness score capped at 0.75
        6M   : 6–8 months  — limited features, BCI data_richness capped at 0.50
        THIN : < 6 months  — insufficient history; policy floor applied,
               no ML income model, BCI data_richness capped at 0.20
        """
        if months >= 12:
            return "12M"
        elif months >= 9:
            return "9M"
        elif months >= 6:
            return "6M"
        else:
            return "THIN"

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

    @staticmethod
    def _max_gap_months(series: pd.Series) -> int:
        """Max consecutive months with zero/near-zero credit (dormancy gap)."""
        max_gap = gap = 0
        for v in series:
            if v <= 0:
                gap += 1
                max_gap = max(max_gap, gap)
            else:
                gap = 0
        return max_gap
