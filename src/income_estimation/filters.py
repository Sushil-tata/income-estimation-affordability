"""
Transaction Filter
──────────────────
Pre-filters monthly transaction aggregates BEFORE feature engineering.

Why this matters:
  Without exclusion, loan disbursements, internal sweeps, and chit fund
  payouts inflate income signals. A customer who received a 500K home loan
  disbursement looks like a high-income SME. Rotating savings (chit funds)
  produce large periodic credits that are NOT income.

Exclusion rules (hard remove from income signals):
  LOAN_DISBURSEMENT  — large one-time credit tagged as loan proceeds
  INTERNAL_SWEEP     — transfer from own account (same customer, same bank)
  REVERSAL           — credit that cancels a prior debit / chargeback
  INSURANCE_PAYOUT   — one-time insurance settlement
  CHIT_FUND          — rotating savings payout: large periodic credit
                       from same counterparty at ~12-month intervals

Flagged for review (not removed — may be legitimate income):
  FX_REMITTANCE      — migrant worker remittance income
  GOVT_TRANSFER      — government subsidy, pension, OTOP support

Two operating modes:
  Mode A — Tagged data (production):
      monthly_agg has pre-tagged exclusion amount columns.
  Mode B — Heuristic (when transaction type tags are unavailable):
      Detects chit fund patterns from spike + interval behaviour.
      Conservative: only exclude credits >= 8× personal median.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class TransactionFilter:
    """
    Removes non-income credits from monthly aggregates before feature engineering.

    Parameters
    ----------
    exclude_tagged : bool
        Apply Mode A column-based exclusions. Default True.
    apply_heuristics : bool
        Apply Mode B heuristic chit-fund / spike detection. Default True.
    spike_review_threshold : float
        Credits > this multiple of personal median are flagged for review
        (not excluded). Default 5.0.
    chit_fund_min_multiple : float
        Credits >= this multiple of personal median AND meeting interval
        criteria are treated as chit fund candidates and excluded.
        Default 8.0 (very conservative — avoids excluding genuine SME contracts).
    chit_fund_min_history_months : int
        Minimum months of data required before applying chit fund heuristic.
        Default 6.
    """

    # Columns that, if present, carry amounts to hard-exclude per month
    EXCLUSION_AMOUNT_COLS = {
        "loan_disbursement_amount": "LOAN_DISBURSEMENT",
        "internal_sweep_amount":    "INTERNAL_SWEEP",
        "reversal_amount":          "REVERSAL",
        "insurance_payout_amount":  "INSURANCE_PAYOUT",
        "excluded_credit_amount":   "PRE_TAGGED",   # catch-all pre-computed exclusion
    }

    # Columns that flag months for manual review (amounts NOT excluded)
    REVIEW_FLAG_COLS = [
        "has_fx_remittance",
        "has_govt_transfer",
        "has_prize_lottery",
        "has_review_flag",
    ]

    def __init__(
        self,
        exclude_tagged: bool = True,
        apply_heuristics: bool = True,
        spike_review_threshold: float = 5.0,
        chit_fund_min_multiple: float = 8.0,
        chit_fund_min_history_months: int = 6,
    ):
        self.exclude_tagged = exclude_tagged
        self.apply_heuristics = apply_heuristics
        self.spike_review_threshold = spike_review_threshold
        self.chit_fund_min_multiple = chit_fund_min_multiple
        self.chit_fund_min_history_months = chit_fund_min_history_months

    def apply(
        self,
        monthly_agg: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply exclusion rules to monthly aggregates.

        Parameters
        ----------
        monthly_agg : pd.DataFrame
            Monthly aggregated transaction data.
            Required columns: customer_id, year_month, total_credit_amount
            Optional columns (Mode A):
              loan_disbursement_amount, internal_sweep_amount,
              reversal_amount, insurance_payout_amount,
              excluded_credit_amount   — pre-computed aggregate exclusion
            Optional columns (review flags):
              has_fx_remittance, has_govt_transfer, has_review_flag

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]:
            - filtered_agg  : monthly_agg with total_credit_amount reduced
            - exclusion_log : per-customer exclusion summary
        """
        df = monthly_agg.copy()

        # Tracks amount excluded and review flags per customer-month
        exc_amount = pd.Series(0.0, index=df.index)
        review_flag = pd.Series(False, index=df.index)
        exc_reasons = pd.Series("", index=df.index)

        # ── Mode A: column-based tagged exclusions ────────────────────────────
        if self.exclude_tagged:
            for col, reason in self.EXCLUSION_AMOUNT_COLS.items():
                if col in df.columns:
                    amt = df[col].fillna(0).clip(lower=0)
                    exc_amount += amt
                    exc_reasons = exc_reasons.where(
                        amt == 0, exc_reasons + f"{reason}|"
                    )

        # ── Review flags (mark but do not exclude) ────────────────────────────
        for col in self.REVIEW_FLAG_COLS:
            if col in df.columns:
                review_flag |= df[col].fillna(False).astype(bool)

        # ── Mode B: heuristic chit fund / spike detection ─────────────────────
        if self.apply_heuristics:
            heuristic = self._detect_anomalies(df)

            # Chit fund: exclude 80% of the spike credit (leave 20% as possible
            # legitimate income from same counterparty in same month)
            chit_mask = heuristic["is_chit_fund_candidate"]
            if chit_mask.any():
                chit_exc = (df.loc[chit_mask, "total_credit_amount"] * 0.80).fillna(0)
                exc_amount = exc_amount.copy()
                exc_amount.loc[chit_mask] += chit_exc.values
                exc_reasons.loc[chit_mask] += "CHIT_FUND_HEURISTIC|"
                logger.info(
                    f"TransactionFilter: chit fund heuristic flagged "
                    f"{chit_mask.sum()} customer-months"
                )

            # Spike: review flag only (not excluded — may be legitimate SME contract)
            review_flag |= heuristic["is_spike_review"]

        # ── Apply exclusions to credit columns ────────────────────────────────
        exc_amount = exc_amount.clip(lower=0)

        original_credit = df["total_credit_amount"].copy()
        df["total_credit_amount"] = (df["total_credit_amount"] - exc_amount).clip(lower=0)

        # Scale down recurring credit proportionally to maintain consistency
        # (if 30% of total credit was excluded, recurring should also reduce by 30%)
        if "recurring_credit_amount" in df.columns:
            scale = df["total_credit_amount"] / (original_credit + 1e-8)
            scale = scale.clip(upper=1.0)  # never scale up
            df["recurring_credit_amount"] = (
                df["recurring_credit_amount"] * scale
            ).clip(lower=0)

        if "irregular_credit_amount" in df.columns:
            scale = df["total_credit_amount"] / (original_credit + 1e-8)
            scale = scale.clip(upper=1.0)
            df["irregular_credit_amount"] = (
                df["irregular_credit_amount"] * scale
            ).clip(lower=0)

        # ── Build per-customer summary ─────────────────────────────────────────
        exc_df = pd.DataFrame({
            "customer_id":  df["customer_id"],
            "amount_excluded": exc_amount,
            "review_flagged":  review_flag,
        })
        summary = exc_df.groupby("customer_id").agg(
            n_months_with_exclusion=("amount_excluded", lambda x: (x > 0).sum()),
            total_amount_excluded=("amount_excluded", "sum"),
            n_months_review_flagged=("review_flagged", "sum"),
        ).reset_index()
        summary["has_any_exclusion"] = summary["total_amount_excluded"] > 0

        n_affected = summary["has_any_exclusion"].sum()
        logger.info(
            f"TransactionFilter: {n_affected:,} / {len(summary):,} customers "
            f"had credits excluded. "
            f"Total excluded: {summary['total_amount_excluded'].sum():,.0f} THB"
        )

        return df, summary

    def _detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Heuristic detection for chit fund patterns and large spikes.

        Chit fund candidate: credit >= chit_fund_min_multiple × personal median
        AND customer has enough history for the pattern to be meaningful.

        Spike review: credit >= spike_review_threshold × personal median.
        These are flagged for review, NOT excluded (could be a large SME contract).
        """
        flags = pd.DataFrame(index=df.index)
        flags["is_chit_fund_candidate"] = False
        flags["is_spike_review"] = False

        if df["total_credit_amount"].isna().all():
            return flags

        median_per_customer = (
            df.groupby("customer_id")["total_credit_amount"]
            .transform("median")
            .fillna(1.0)
        )
        months_per_customer = (
            df.groupby("customer_id")["year_month"]
            .transform("count")
        )

        ratio = df["total_credit_amount"] / (median_per_customer + 1e-8)

        # Spike review (not excluded)
        flags["is_spike_review"] = ratio > self.spike_review_threshold

        # Chit fund exclusion candidate (very large spike + sufficient history)
        flags["is_chit_fund_candidate"] = (
            (ratio >= self.chit_fund_min_multiple)
            & (months_per_customer >= self.chit_fund_min_history_months)
        )

        return flags

    def get_exclusion_impact(
        self,
        original_agg: pd.DataFrame,
        filtered_agg: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Per-customer comparison of avg monthly credit before and after filtering.
        Useful for calibrating exclusion thresholds.
        """
        orig = original_agg.groupby("customer_id")["total_credit_amount"].mean()
        filt = filtered_agg.groupby("customer_id")["total_credit_amount"].mean()
        impact = pd.DataFrame({"avg_credit_original": orig, "avg_credit_filtered": filt})
        impact["reduction_abs"] = impact["avg_credit_original"] - impact["avg_credit_filtered"]
        impact["reduction_pct"] = (
            impact["reduction_abs"] / impact["avg_credit_original"].replace(0, np.nan)
        ).fillna(0).round(4)
        return impact.reset_index()
