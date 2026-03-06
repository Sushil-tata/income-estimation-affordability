"""
Credit Card Feature Engineer (Optional Module)
───────────────────────────────────────────────
Extracts 8 credit-card-specific features from monthly CC aggregate data.

This module is OPTIONAL — it is only called when the customer has an active
credit card at the bank. When CC data is unavailable, the features are simply
absent from the feature matrix and downstream models fall back to the full
feature set computed by FeatureEngineer.

CC features enrich three downstream components:
  1. Composite indices: SI enrichment via cc_utilisation, CI enrichment via
     cc_merchant_category_diversity (B2B MCCs as additional concentration proxy)
  2. BCI behavioral_consistency component: cc_payment_behaviour_score
  3. Affordability engine: cc_min_payment_amount added to existing obligations;
     cc_income_floor_breach check in AffordabilityEngine.compute()

Input  : Monthly CC aggregate data (one row per customer_id × year_month)
Output : Customer-level CC feature DataFrame (one row per customer_id)

Required input columns
----------------------
  customer_id, year_month
  cc_credit_limit          : Monthly credit limit (THB)
  cc_outstanding_balance   : End-of-month outstanding balance (THB)
  cc_spend_amount          : Total CC spend in month (THB)
  cc_payment_amount        : Payment made toward CC balance (THB)
  cc_min_payment_amount    : Minimum payment due (THB)
  cc_merchant_category_hhi : HHI of spend across merchant categories (0–1)
                             (1 = spend entirely in one category)
  cc_months_active         : Months since first CC transaction (can be precomputed)
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CreditCardFeatureEngineer:
    """
    Computes customer-level CC features from monthly CC aggregate data.

    Parameters
    ----------
    min_months : int
        Minimum months of CC history to compute features.
        Customers below this threshold receive NaN for all CC features.
    """

    def __init__(self, min_months: int = 3):
        self.min_months = min_months

    def build_features(self, cc_monthly: pd.DataFrame) -> pd.DataFrame:
        """
        Build CC feature matrix from monthly CC aggregates.

        Parameters
        ----------
        cc_monthly : pd.DataFrame
            Monthly CC aggregate data. See module docstring for required columns.

        Returns
        -------
        pd.DataFrame
            Customer-level CC feature matrix (one row per customer_id).
            Customers with fewer than min_months of CC history are included
            but all feature columns are NaN — callers should left-join this
            onto the main feature matrix.
        """
        logger.info(f"Building CC features for {cc_monthly['customer_id'].nunique():,} customers...")

        g = cc_monthly.groupby("customer_id")
        feat = pd.DataFrame(index=cc_monthly["customer_id"].unique())
        feat.index.name = "customer_id"

        cc_months = g["year_month"].count()
        feat["cc_months_active"] = cc_months

        # ── 1. Average utilisation (outstanding / limit) ──────────────────────
        # High utilisation → financial stress or high spending power (context-dependent).
        # Values clipped at 1.5 to handle temporary over-limit situations.
        feat["cc_avg_utilisation"] = g.apply(
            lambda df: (df["cc_outstanding_balance"] / df["cc_credit_limit"].replace(0, np.nan))
            .clip(0, 1.5).mean()
        ).fillna(np.nan)

        # ── 2. Monthly CC spend (absolute and relative to credit) ─────────────
        feat["cc_avg_monthly_spend"] = g["cc_spend_amount"].mean()

        # CC spend as fraction of bank credit (proxy for CC-reliance)
        # Merged with main features downstream; left as standalone here.
        feat["cc_spend_6m"] = g["cc_spend_amount"].apply(
            lambda x: x.iloc[-6:].sum() if len(x) >= 6 else x.sum()
        )

        # ── 3. Payment behaviour score (0–1) ─────────────────────────────────
        # 1.0 = always pays full balance; 0.5 = always pays minimum; 0.0 = never pays
        # score = mean(payment / outstanding), clipped to [0, 1]
        feat["cc_payment_behaviour_score"] = g.apply(
            lambda df: (df["cc_payment_amount"] / df["cc_outstanding_balance"].replace(0, np.nan))
            .clip(0, 1).mean()
        ).fillna(0.0)

        # ── 4. Months at minimum payment (financial stress signal) ────────────
        # Count months where payment ≈ minimum payment (within 5% tolerance)
        feat["cc_months_at_min_payment"] = g.apply(
            lambda df: (
                df["cc_payment_amount"] <= df["cc_min_payment_amount"] * 1.05
            ).sum()
        )

        # ── 5. Average minimum payment (used as existing obligation) ──────────
        # AffordabilityEngine adds this to avg_commitment_amount_12m
        feat["cc_min_payment_amount"] = g["cc_min_payment_amount"].mean()

        # ── 6. Merchant category diversity (inverse HHI) ──────────────────────
        # Low HHI = spend diversified across many categories (lifestyle/personal)
        # High HHI = concentrated spend (may indicate B2B / business card use)
        # We invert HHI so higher score = more diverse (more personal use)
        feat["cc_merchant_category_diversity"] = g["cc_merchant_category_hhi"].apply(
            lambda x: 1 - x.mean() if x.notna().any() else np.nan
        )

        # ── 7. Credit limit utilisation trend (linear slope over time) ──────
        # Rising utilisation trend → potential financial stress
        feat["cc_utilisation_trend"] = g.apply(
            self._utilisation_slope
        ).fillna(0.0)

        # ── 8. CC-to-bank income ratio (spend proxy for income floor check) ───
        # Computed downstream in AffordabilityEngine using cc_spend_6m /
        # adjusted_income. Stored here for reference.
        feat["cc_avg_credit_limit"] = g["cc_credit_limit"].mean()

        # ── Mask customers with insufficient CC history ───────────────────────
        thin_cc = cc_months < self.min_months
        cc_feature_cols = [c for c in feat.columns if c != "cc_months_active"]
        feat.loc[thin_cc, cc_feature_cols] = np.nan
        if thin_cc.any():
            logger.info(
                f"  {thin_cc.sum():,} customers have < {self.min_months} months of CC history "
                f"— CC features set to NaN (will use bank-only features)"
            )

        logger.info(
            f"CC feature matrix: {feat.shape[0]:,} customers × {feat.shape[1]} CC features"
        )
        return feat.reset_index()

    @staticmethod
    def _utilisation_slope(df: pd.DataFrame) -> float:
        """Linear slope of monthly CC utilisation for one customer group."""
        util = (
            df["cc_outstanding_balance"]
            / df["cc_credit_limit"].replace(0, np.nan)
        ).clip(0, 1.5).values
        if len(util) < 3 or np.isnan(util).all():
            return 0.0
        util = np.where(np.isnan(util), 0.0, util)
        try:
            return float(np.polyfit(np.arange(len(util)), util, 1)[0])
        except Exception:
            return 0.0

    @classmethod
    def merge_with_features(
        cls,
        features: pd.DataFrame,
        cc_features: pd.DataFrame,
        on: str = "customer_id",
    ) -> pd.DataFrame:
        """
        Left-join CC features onto the main feature matrix.

        Customers without CC data retain all bank features; CC columns are NaN.
        Downstream models (LightGBM, TabPFN) handle NaN natively.

        Parameters
        ----------
        features : pd.DataFrame
            Output of FeatureEngineer.build_features() — bank transaction features.
        cc_features : pd.DataFrame
            Output of CreditCardFeatureEngineer.build_features().

        Returns
        -------
        pd.DataFrame
            Merged feature matrix with CC features added as optional columns.
        """
        merged = features.merge(cc_features, on=on, how="left")
        cc_cols = [c for c in cc_features.columns if c != on]
        n_with_cc = merged[cc_cols[0]].notna().sum() if cc_cols else 0
        logger.info(
            f"CC merge: {n_with_cc:,} / {len(merged):,} customers have CC features "
            f"({n_with_cc / max(len(merged), 1):.1%})"
        )
        return merged
