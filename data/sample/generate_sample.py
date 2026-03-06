"""
Sample Data Generator
──────────────────────
Generates synthetic monthly transaction aggregate data for development
and testing of the Income Estimation & Affordability pipeline.

All data is synthetic — it statistically mimics real transaction patterns
per behavioral segment but contains no real customer information.

Segments and their statistical fingerprints:
  PAYROLL         : Regular monthly credits, payroll flag, single dominant source
  SALARY_LIKE     : Regular credits, low CV, no payroll flag
  SME             : High volume, business MCC credits, multiple payers
  GIG_FREELANCE   : Irregular credits, gaps in income, multiple small sources
  PASSIVE_INVESTOR: Low transaction activity, investment/interest credits
  THIN            : Short history, low transaction density

Output formats:
  1. Monthly transaction aggregates  → input to FeatureEngineer
  2. Customer-level features         → input to Segmentation + Income pipelines
  3. Training data (features + labels) → 160K verified income records
  4. Scoring population              → unlabeled, all segments

Usage:
  gen = SampleDataGenerator(seed=42)
  monthly_df  = gen.monthly_transactions(n_customers=5000)
  features_df = gen.customer_features(n_customers=5000)
  train_X, train_y = gen.training_data(n_customers=10000)
  score_df    = gen.scoring_population(n_customers=50000)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


# ── Segment mix for scoring population ──────────────────────────────────────
SEGMENT_MIX = {
    "PAYROLL":          0.15,
    "SALARY_LIKE":      0.25,
    "SME":              0.20,
    "GIG_FREELANCE":    0.20,
    "PASSIVE_INVESTOR": 0.10,
    "THIN":             0.10,
}

# ── Occupation split for training labels ─────────────────────────────────────
# Salaried: PAYROLL + SALARY_LIKE + some GIG
# Self-Employed: SME + GIG + PASSIVE
OCCUPATION_BY_SEGMENT = {
    "PAYROLL":          "SA",
    "SALARY_LIKE":      "SA",
    "SME":              "SE",
    "GIG_FREELANCE":    "SE",
    "PASSIVE_INVESTOR": "SE",
    "THIN":             "SA",   # Unknown — default to SA
}

# ── Income distributions per segment (THB/month, log-normal params) ──────────
# log-normal: mean and std of log(income)
INCOME_PARAMS = {
    "PAYROLL":          {"dist": "lognormal", "mu": 10.6, "sigma": 0.45},   # ~40K median
    "SALARY_LIKE":      {"dist": "lognormal", "mu": 10.3, "sigma": 0.50},   # ~30K median
    "SME":              {"dist": "lognormal", "mu": 11.2, "sigma": 0.80},   # ~73K median, high var
    "GIG_FREELANCE":    {"dist": "lognormal", "mu": 10.1, "sigma": 0.65},   # ~24K median
    "PASSIVE_INVESTOR": {"dist": "lognormal", "mu": 10.8, "sigma": 0.70},   # ~49K median
    "THIN":             {"dist": "lognormal", "mu": 10.0, "sigma": 0.55},   # ~22K median
}

# Clip to realistic bounds (THB/month)
INCOME_CLIP = {"min": 8_000, "max": 800_000}


class SampleDataGenerator:
    """
    Synthetic data generator for the Income Estimation & Affordability framework.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    observation_date : str
        Reference date (YYYY-MM-DD). Features are computed as of this date.
        Defaults to 2024-07-01 (end of dev window).
    n_months : int
        Months of history to generate per customer. Default 12.
    """

    def __init__(
        self,
        seed: int = 42,
        observation_date: str = "2024-07-01",
        n_months: int = 12,
    ):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.observation_date = pd.to_datetime(observation_date)
        self.n_months = n_months

        # Pre-compute month labels (newest last)
        self.month_labels = [
            (self.observation_date - relativedelta(months=n_months - 1 - i))
            .strftime("%Y-%m")
            for i in range(n_months)
        ]

    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════════

    def monthly_transactions(
        self,
        n_customers: int = 5_000,
        segment_mix: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Generate monthly transaction aggregate data.

        Returns one row per (customer_id, year_month) — long format.
        This is the direct input to FeatureEngineer.build_features().

        Columns:
          customer_id, year_month, segment (true label for validation),
          total_credit_amount, total_debit_amount,
          recurring_credit_amount, irregular_credit_amount, investment_credit_amount,
          commitment_amount, recurring_expense_amount, lifestyle_amount,
          eom_balance, transaction_count,
          business_mcc_credit_share, dominant_credit_source_share, has_payroll_credit
        """
        mix = segment_mix or SEGMENT_MIX
        segments, n_per_seg = self._allocate_segments(n_customers, mix)

        all_rows = []
        cust_id = 1

        for seg, n in zip(segments, n_per_seg):
            for _ in range(n):
                rows = self._generate_customer_months(cust_id, seg)
                all_rows.extend(rows)
                cust_id += 1

        df = pd.DataFrame(all_rows)
        df["customer_id"] = df["customer_id"].astype(str).str.zfill(8)
        df = df.sort_values(["customer_id", "year_month"]).reset_index(drop=True)
        return df

    def customer_features(
        self,
        n_customers: int = 5_000,
        segment_mix: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Generate customer-level feature matrix directly.

        Bypasses monthly transaction generation + FeatureEngineer for speed.
        Output is equivalent to FeatureEngineer.build_features() output.
        Includes true segment label for validation.
        """
        monthly = self.monthly_transactions(n_customers, segment_mix)
        features = self._aggregate_to_customer_features(monthly)
        return features

    def training_data(
        self,
        n_customers: int = 10_000,
        sa_share: float = 0.60,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate training dataset with verified income labels.

        Mimics the 160K verified lending records from CardX.
        Only includes customers whose income is verified (no THIN).

        Parameters
        ----------
        n_customers : int
            Number of training records to generate.
        sa_share : float
            Share of salaried customers (default 60%).

        Returns
        -------
        X : pd.DataFrame
            Customer-level feature matrix with segment column.
        y : pd.Series
            Verified gross monthly income (THB). Index matches X.
        """
        # Training mix: no THIN (unverified), heavier PAYROLL + SALARY_LIKE
        train_mix = {
            "PAYROLL":          sa_share * 0.35,
            "SALARY_LIKE":      sa_share * 0.65,
            "SME":              (1 - sa_share) * 0.45,
            "GIG_FREELANCE":    (1 - sa_share) * 0.40,
            "PASSIVE_INVESTOR": (1 - sa_share) * 0.15,
        }

        features = self.customer_features(n_customers, segment_mix=train_mix)

        # Generate verified income — correlated with credit behavior
        y = self._generate_verified_income(features)

        return features, y

    def scoring_population(
        self,
        n_customers: int = 50_000,
        segment_mix: Optional[dict] = None,
        include_payroll_income: bool = True,
    ) -> pd.DataFrame:
        """
        Generate full scoring population (no income labels).

        Represents the ETB base to be scored by the pipeline.
        Includes payroll_income column for PAYROLL customers only.
        """
        features = self.customer_features(n_customers, segment_mix)

        # Add payroll income for PAYROLL customers (known from SCB deposits)
        if include_payroll_income:
            features["payroll_income"] = np.nan
            payroll_mask = features["segment"] == "PAYROLL"
            n_payroll = payroll_mask.sum()
            if n_payroll > 0:
                features.loc[payroll_mask, "payroll_income"] = self._sample_income(
                    "PAYROLL", n_payroll
                )

        return features

    # ═══════════════════════════════════════════════════════════════════════════
    # MONTHLY TRANSACTION GENERATORS (per segment)
    # ═══════════════════════════════════════════════════════════════════════════

    def _generate_customer_months(self, cust_id: int, segment: str) -> list:
        """Generate n_months of monthly aggregate rows for one customer."""
        gen = getattr(self, f"_gen_{segment.lower()}")
        params = gen()

        # Determine actual months available (THIN gets fewer)
        if segment == "THIN":
            n_active = self.rng.randint(2, 6)
            active_months = self.month_labels[-n_active:]
        else:
            # Occasional missing months for realism
            drop_prob = {"PAYROLL": 0.0, "SALARY_LIKE": 0.02,
                         "SME": 0.05, "GIG_FREELANCE": 0.10,
                         "PASSIVE_INVESTOR": 0.08, "THIN": 0.0}
            p_drop = drop_prob.get(segment, 0.03)
            active_months = [m for m in self.month_labels
                             if self.rng.random() > p_drop]
            if not active_months:
                active_months = self.month_labels[-3:]

        rows = []
        balance = params["initial_balance"]

        for i, ym in enumerate(active_months):
            credit, debit, row_extra = self._monthly_cashflow(params, segment, i, len(active_months))

            # EOM balance: random walk with mean reversion
            balance = max(0, balance + credit - debit + self.rng.normal(0, balance * 0.05))

            rows.append({
                "customer_id": cust_id,
                "year_month": ym,
                "segment": segment,
                "total_credit_amount": round(credit, 2),
                "total_debit_amount": round(debit, 2),
                "recurring_credit_amount": round(row_extra["recurring_credit"], 2),
                "irregular_credit_amount": round(row_extra["irregular_credit"], 2),
                "investment_credit_amount": round(row_extra["investment_credit"], 2),
                "commitment_amount": round(row_extra["commitment"], 2),
                "recurring_expense_amount": round(row_extra["recurring_expense"], 2),
                "lifestyle_amount": round(row_extra["lifestyle"], 2),
                "eom_balance": round(balance, 2),
                "transaction_count": int(row_extra["tx_count"]),
                "business_mcc_credit_share": round(row_extra["biz_mcc_share"], 4),
                "dominant_credit_source_share": round(row_extra["dominant_share"], 4),
                "has_payroll_credit": int(row_extra["has_payroll"]),
            })

        return rows

    # ── Segment parameter generators ─────────────────────────────────────────

    def _gen_payroll(self) -> dict:
        base_income = self._sample_income("PAYROLL", 1)[0]
        return {
            "base_credit": base_income,
            "credit_cv": self.rng.uniform(0.02, 0.08),     # Very stable
            "debit_ratio": self.rng.uniform(0.55, 0.85),
            "commitment_ratio": self.rng.uniform(0.15, 0.35),
            "recurring_ratio": self.rng.uniform(0.25, 0.45),
            "lifestyle_ratio": self.rng.uniform(0.10, 0.25),
            "dominant_share": self.rng.uniform(0.88, 1.00),
            "biz_mcc_share": self.rng.uniform(0.00, 0.03),
            "has_payroll": True,
            "investment_freq": 0.0,
            "tx_count_base": self.rng.randint(15, 40),
            "initial_balance": base_income * self.rng.uniform(0.5, 3.0),
        }

    def _gen_salary_like(self) -> dict:
        base_income = self._sample_income("SALARY_LIKE", 1)[0]
        return {
            "base_credit": base_income,
            "credit_cv": self.rng.uniform(0.05, 0.22),     # Low but not as tight as payroll
            "debit_ratio": self.rng.uniform(0.60, 0.90),
            "commitment_ratio": self.rng.uniform(0.15, 0.40),
            "recurring_ratio": self.rng.uniform(0.25, 0.45),
            "lifestyle_ratio": self.rng.uniform(0.10, 0.25),
            "dominant_share": self.rng.uniform(0.62, 0.92),
            "biz_mcc_share": self.rng.uniform(0.00, 0.05),
            "has_payroll": False,
            "investment_freq": self.rng.uniform(0.0, 0.05),
            "tx_count_base": self.rng.randint(12, 35),
            "initial_balance": base_income * self.rng.uniform(0.3, 2.5),
        }

    def _gen_sme(self) -> dict:
        base_income = self._sample_income("SME", 1)[0]
        # SME: high credit volume, business MCC, multiple payers
        credit_multiplier = self.rng.uniform(1.5, 4.0)  # Revenue > income (gross sales)
        return {
            "base_credit": base_income * credit_multiplier,
            "credit_cv": self.rng.uniform(0.20, 0.60),     # High variance
            "debit_ratio": self.rng.uniform(0.70, 1.10),   # Can be >1 (business cycling)
            "commitment_ratio": self.rng.uniform(0.10, 0.30),
            "recurring_ratio": self.rng.uniform(0.15, 0.35),
            "lifestyle_ratio": self.rng.uniform(0.05, 0.20),
            "dominant_share": self.rng.uniform(0.15, 0.50),  # Diversified payers
            "biz_mcc_share": self.rng.uniform(0.40, 0.85),
            "has_payroll": False,
            "investment_freq": self.rng.uniform(0.0, 0.15),
            "tx_count_base": self.rng.randint(25, 80),
            "initial_balance": base_income * self.rng.uniform(0.5, 5.0),
            "true_income": base_income,  # Track actual income vs gross revenue
        }

    def _gen_gig_freelance(self) -> dict:
        base_income = self._sample_income("GIG_FREELANCE", 1)[0]
        return {
            "base_credit": base_income,
            "credit_cv": self.rng.uniform(0.30, 0.80),     # High irregular variance
            "debit_ratio": self.rng.uniform(0.55, 0.95),
            "commitment_ratio": self.rng.uniform(0.05, 0.25),
            "recurring_ratio": self.rng.uniform(0.20, 0.45),
            "lifestyle_ratio": self.rng.uniform(0.10, 0.30),
            "dominant_share": self.rng.uniform(0.20, 0.55),
            "biz_mcc_share": self.rng.uniform(0.00, 0.15),
            "has_payroll": False,
            "investment_freq": 0.0,
            "zero_income_months": self.rng.randint(1, 4),  # Gaps
            "tx_count_base": self.rng.randint(8, 25),
            "initial_balance": base_income * self.rng.uniform(0.1, 1.5),
        }

    def _gen_passive_investor(self) -> dict:
        base_income = self._sample_income("PASSIVE_INVESTOR", 1)[0]
        return {
            "base_credit": base_income * self.rng.uniform(0.3, 0.8),  # Low regular credits
            "credit_cv": self.rng.uniform(0.15, 0.50),
            "debit_ratio": self.rng.uniform(0.30, 0.65),   # Low spend
            "commitment_ratio": self.rng.uniform(0.05, 0.20),
            "recurring_ratio": self.rng.uniform(0.15, 0.35),
            "lifestyle_ratio": self.rng.uniform(0.05, 0.20),
            "dominant_share": self.rng.uniform(0.30, 0.70),
            "biz_mcc_share": self.rng.uniform(0.00, 0.05),
            "has_payroll": False,
            "investment_freq": self.rng.uniform(0.30, 0.80),  # Key signal
            "investment_credit_base": base_income * self.rng.uniform(0.3, 1.2),
            "tx_count_base": self.rng.randint(5, 18),
            "initial_balance": base_income * self.rng.uniform(2.0, 10.0),  # High savings
        }

    def _gen_thin(self) -> dict:
        base_income = self._sample_income("THIN", 1)[0]
        return {
            "base_credit": base_income,
            "credit_cv": self.rng.uniform(0.10, 0.50),
            "debit_ratio": self.rng.uniform(0.50, 0.90),
            "commitment_ratio": self.rng.uniform(0.05, 0.20),
            "recurring_ratio": self.rng.uniform(0.20, 0.40),
            "lifestyle_ratio": self.rng.uniform(0.05, 0.20),
            "dominant_share": self.rng.uniform(0.40, 0.90),
            "biz_mcc_share": self.rng.uniform(0.00, 0.10),
            "has_payroll": False,
            "investment_freq": 0.0,
            "tx_count_base": self.rng.randint(2, 8),       # Very few transactions
            "initial_balance": base_income * self.rng.uniform(0.1, 1.0),
        }

    # ── Monthly cashflow generator ───────────────────────────────────────────

    def _monthly_cashflow(self, params: dict, segment: str, month_idx: int, n_months: int):
        """Generate credit/debit amounts for a single month."""
        base = params["base_credit"]
        cv = params["credit_cv"]

        # Zero-income months for GIG
        if segment == "GIG_FREELANCE":
            zero_months = params.get("zero_income_months", 0)
            zero_prob = zero_months / max(n_months, 1)
            if self.rng.random() < zero_prob:
                total_credit = self.rng.uniform(0, base * 0.1)  # Near-zero
            else:
                total_credit = max(0, self.rng.normal(base, base * cv))
        elif segment == "SME":
            # SME: seasonal variation
            seasonal_factor = 1 + 0.20 * np.sin(2 * np.pi * month_idx / 12)
            total_credit = max(0, self.rng.normal(base * seasonal_factor, base * cv))
        else:
            total_credit = max(0, self.rng.normal(base, base * cv))

        # Credit breakdown
        if segment == "PAYROLL":
            recurring_credit = total_credit * self.rng.uniform(0.90, 1.00)
            investment_credit = 0.0
        elif segment == "PASSIVE_INVESTOR":
            inv_base = params.get("investment_credit_base", base * 0.5)
            investment_credit = (inv_base * self.rng.uniform(0.7, 1.3)
                                 if self.rng.random() < params["investment_freq"] else 0.0)
            recurring_credit = total_credit * self.rng.uniform(0.10, 0.40)
            total_credit = recurring_credit + investment_credit + self.rng.uniform(0, base * 0.1)
        else:
            recurring_credit = total_credit * self.rng.uniform(0.30, 0.80)
            investment_credit = 0.0

        irregular_credit = max(0, total_credit - recurring_credit - investment_credit)

        # Debit amounts
        debit_ratio = min(params["debit_ratio"], 1.5)   # Cap to avoid extreme values
        total_debit = total_credit * debit_ratio * self.rng.uniform(0.85, 1.15)
        total_debit = max(0, total_debit)

        commitment = total_debit * params["commitment_ratio"] * self.rng.uniform(0.85, 1.15)
        recurring_expense = total_debit * params["recurring_ratio"] * self.rng.uniform(0.85, 1.15)
        lifestyle = total_debit * params["lifestyle_ratio"] * self.rng.uniform(0.70, 1.30)

        # Normalise debits to not exceed total
        debit_parts = commitment + recurring_expense + lifestyle
        if debit_parts > total_debit * 0.95:
            scale = (total_debit * 0.95) / debit_parts
            commitment *= scale
            recurring_expense *= scale
            lifestyle *= scale

        # Business MCC share and dominant source
        biz_mcc = params["biz_mcc_share"] * self.rng.uniform(0.80, 1.20)
        dominant = params["dominant_share"] * self.rng.uniform(0.90, 1.10)
        biz_mcc = np.clip(biz_mcc, 0, 1)
        dominant = np.clip(dominant, 0, 1)

        tx_count = max(1, int(self.rng.normal(params["tx_count_base"],
                                               params["tx_count_base"] * 0.20)))

        extra = {
            "recurring_credit": recurring_credit,
            "irregular_credit": irregular_credit,
            "investment_credit": investment_credit,
            "commitment": commitment,
            "recurring_expense": recurring_expense,
            "lifestyle": lifestyle,
            "biz_mcc_share": biz_mcc,
            "dominant_share": dominant,
            "has_payroll": params["has_payroll"],
            "tx_count": tx_count,
        }

        return total_credit, total_debit, extra

    # ═══════════════════════════════════════════════════════════════════════════
    # FEATURE AGGREGATION
    # ═══════════════════════════════════════════════════════════════════════════

    def _aggregate_to_customer_features(self, monthly: pd.DataFrame) -> pd.DataFrame:
        """Aggregate monthly data to customer-level features matching FeatureEngineer output."""
        g = monthly.groupby("customer_id")

        feat = pd.DataFrame(index=monthly["customer_id"].unique())
        feat.index.name = "customer_id"

        # ── Segment (true label) ──────────────────────────────────────────────
        feat["segment"] = g["segment"].first()

        # ── Credits ──────────────────────────────────────────────────────────
        feat["avg_monthly_credit_12m"] = g["total_credit_amount"].mean()
        feat["median_monthly_credit_12m"] = g["total_credit_amount"].median()
        feat["max_monthly_credit_3m"] = g["total_credit_amount"].apply(
            lambda x: x.sort_index().iloc[-3:].max() if len(x) >= 3 else x.max()
        )
        feat["cv_monthly_credit_12m"] = g["total_credit_amount"].apply(
            lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan
        ).fillna(1.0)

        feat["months_with_zero_credit"] = g["total_credit_amount"].apply(
            lambda x: (x < 500).sum()
        )
        feat["months_data_available"] = g["year_month"].count()

        feat["avg_recurring_credit_12m"] = g["recurring_credit_amount"].mean()
        feat["avg_irregular_credit_12m"] = g["irregular_credit_amount"].mean()
        feat["avg_investment_credit_12m"] = g["investment_credit_amount"].mean()

        feat["cv_recurring_credit_12m"] = g["recurring_credit_amount"].apply(
            lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan
        ).fillna(1.0)

        feat["recurring_to_total_credit_ratio"] = (
            feat["avg_recurring_credit_12m"] / feat["avg_monthly_credit_12m"].replace(0, np.nan)
        ).fillna(0)

        feat["investment_credit_frequency"] = g["investment_credit_amount"].apply(
            lambda x: (x > 100).mean()
        )

        feat["recurring_credit_streak_months"] = g["recurring_credit_amount"].apply(
            lambda x: _max_consecutive_positive(x.values)
        )

        feat["has_payroll_credit"] = g["has_payroll_credit"].max()
        feat["months_with_salary_pattern"] = g["has_payroll_credit"].sum()
        feat["dominant_credit_source_share"] = g["dominant_credit_source_share"].mean()
        feat["credit_concentration_index"] = g["dominant_credit_source_share"].mean()
        feat["business_mcc_credit_share"] = g["business_mcc_credit_share"].mean()

        # ── Debits ────────────────────────────────────────────────────────────
        feat["avg_commitment_amount_12m"] = g["commitment_amount"].mean()
        feat["commitment_regularity"] = g["commitment_amount"].apply(
            lambda x: (x > 0).mean()
        )
        feat["avg_recurring_expense_12m"] = g["recurring_expense_amount"].mean()
        feat["recurring_expense_regularity"] = g["recurring_expense_amount"].apply(
            lambda x: (x > 0).mean()
        )
        feat["avg_lifestyle_amount_12m"] = g["lifestyle_amount"].mean()
        feat["avg_total_debit_12m"] = g["total_debit_amount"].mean()
        feat["cv_total_debit_12m"] = g["total_debit_amount"].apply(
            lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan
        ).fillna(1.0)

        # ── Balance ───────────────────────────────────────────────────────────
        feat["avg_eom_balance_3m"] = g["eom_balance"].apply(
            lambda x: x.sort_index().iloc[-3:].mean() if len(x) >= 3 else x.mean()
        )
        feat["avg_eom_balance_6m"] = g["eom_balance"].apply(
            lambda x: x.sort_index().iloc[-6:].mean() if len(x) >= 6 else x.mean()
        )
        feat["avg_eom_balance_12m"] = g["eom_balance"].mean()
        feat["max_eom_balance_3m"] = g["eom_balance"].apply(
            lambda x: x.sort_index().iloc[-3:].max() if len(x) >= 3 else x.max()
        )
        feat["min_eom_balance_3m"] = g["eom_balance"].apply(
            lambda x: x.sort_index().iloc[-3:].min() if len(x) >= 3 else x.min()
        )
        feat["balance_cv_12m"] = g["eom_balance"].apply(
            lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan
        ).fillna(1.0)
        feat["months_below_1000_balance"] = g["eom_balance"].apply(
            lambda x: (x < 1_000).sum()
        )
        feat["balance_trend_slope"] = g["eom_balance"].apply(
            lambda x: _linear_slope(x.values)
        )
        feat["transaction_count_avg_monthly"] = g["transaction_count"].mean()

        # ── Derived ───────────────────────────────────────────────────────────
        feat["net_monthly_flow_avg"] = feat["avg_monthly_credit_12m"] - feat["avg_total_debit_12m"]
        feat["inflow_outflow_ratio"] = (
            feat["avg_monthly_credit_12m"] / feat["avg_total_debit_12m"].replace(0, np.nan)
        ).fillna(1.0)
        feat["savings_rate_proxy"] = (
            feat["net_monthly_flow_avg"] / feat["avg_monthly_credit_12m"].replace(0, np.nan)
        ).fillna(0)
        feat["commitment_ratio"] = (
            feat["avg_commitment_amount_12m"] / feat["avg_total_debit_12m"].replace(0, np.nan)
        ).fillna(0)
        feat["lifestyle_ratio"] = (
            feat["avg_lifestyle_amount_12m"] / feat["avg_total_debit_12m"].replace(0, np.nan)
        ).fillna(0)
        feat["balance_to_max_credit_ratio"] = (
            feat["avg_eom_balance_3m"] / feat["max_monthly_credit_3m"].replace(0, np.nan)
        ).fillna(0)
        feat["income_to_obligation_ratio"] = (
            feat["avg_recurring_credit_12m"] / feat["avg_commitment_amount_12m"].replace(0, np.nan)
        ).fillna(0)
        feat["financial_stress_index"] = (
            feat["months_below_1000_balance"] / feat["months_data_available"].replace(0, 1)
        )

        # ── Phase 1: Recurring structure ──────────────────────────────────────
        feat["median_recurring_credit_12m"] = g["recurring_credit_amount"].median()

        feat["recurring_stream_survival_ratio"] = g["recurring_credit_amount"].apply(
            lambda x: (x > x.median() * 0.5).mean() if x.median() > 0 else 0.0
        )
        feat["recurring_credit_deviation_mom"] = g["recurring_credit_amount"].apply(
            lambda x: x.diff().abs().median() / x.median()
            if len(x) > 1 and x.median() > 0 else 1.0
        ).fillna(1.0)
        feat["salary_periodicity_confidence"] = g["recurring_credit_amount"].apply(
            lambda x: (np.abs(x - x.median()) <= x.median() * 0.10).mean()
            if x.median() > 0 else 0.0
        )
        feat["active_month_ratio_12m"] = g["total_credit_amount"].apply(
            lambda x: (x > 0).mean()
        )
        feat["avg_monthly_credit_6m"] = g["total_credit_amount"].apply(
            lambda x: x.sort_index().iloc[-6:].mean() if len(x) >= 6 else x.mean()
        )
        feat["avg_monthly_debit_6m"] = g["total_debit_amount"].apply(
            lambda x: x.sort_index().iloc[-6:].mean() if len(x) >= 6 else x.mean()
        )

        # ── Phase 1: Volatility ───────────────────────────────────────────────
        feat["credit_skewness_12m"] = g["total_credit_amount"].apply(
            lambda x: float(x.skew()) if len(x) >= 4 else 0.0
        ).fillna(0.0)
        feat["credit_kurtosis_12m"] = g["total_credit_amount"].apply(
            lambda x: float(x.kurt()) if len(x) >= 4 else 0.0
        ).fillna(0.0)
        feat["credit_p95_p50_ratio"] = g["total_credit_amount"].apply(
            lambda x: np.percentile(x, 95) / np.median(x) if np.median(x) > 0 else 1.0
        ).fillna(1.0)
        feat["credit_spike_ratio"] = g["total_credit_amount"].apply(
            lambda x: (x > x.median() * 2).mean() if x.median() > 0 else 0.0
        )
        feat["mom_growth_volatility"] = g["total_credit_amount"].apply(
            lambda x: x.pct_change().dropna().std() if len(x) > 2 else 1.0
        ).fillna(1.0).clip(0, 5)
        feat["max_month_drop_6m"] = g["total_credit_amount"].apply(
            lambda x: (-(x.sort_index().iloc[-6:].pct_change().dropna())).clip(lower=0).max()
            if len(x) >= 3 else 0.0
        ).fillna(0.0)

        # ── Phase 1: Seasonality ──────────────────────────────────────────────
        feat["seasonality_index"] = g["total_credit_amount"].apply(
            lambda x: x.max() / x[x > 0].min()
            if (x > 0).any() and x[x > 0].min() > 0 else 1.0
        ).fillna(1.0).clip(1, 20)
        feat["peak_trough_ratio"] = g["total_credit_amount"].apply(
            lambda x: x.max() / x.median() if x.median() > 0 else 1.0
        ).fillna(1.0)
        feat["top2_month_inflow_share"] = g["total_credit_amount"].apply(
            lambda x: x.nlargest(2).sum() / x.sum() if x.sum() > 0 else 1.0
        ).fillna(1.0)
        feat["top3_month_inflow_share"] = g["total_credit_amount"].apply(
            lambda x: x.nlargest(3).sum() / x.sum() if x.sum() > 0 else 1.0
        ).fillna(1.0)
        feat["rolling_3m_vs_12m_ratio"] = g["total_credit_amount"].apply(
            lambda x: x.sort_index().iloc[-3:].median() / x.median()
            if len(x) >= 3 and x.median() > 0 else 1.0
        ).fillna(1.0)

        # ── Phase 1: Short-window ─────────────────────────────────────────────
        feat["median_credit_3m"] = g["total_credit_amount"].apply(
            lambda x: x.sort_index().iloc[-3:].median() if len(x) >= 3 else x.median()
        )
        feat["median_credit_6m"] = g["total_credit_amount"].apply(
            lambda x: x.sort_index().iloc[-6:].median() if len(x) >= 6 else x.median()
        )
        feat["credit_cv_6m"] = g["total_credit_amount"].apply(
            lambda x: x.sort_index().iloc[-6:].std() / x.sort_index().iloc[-6:].mean()
            if len(x) >= 6 and x.sort_index().iloc[-6:].mean() > 0 else 1.0
        ).fillna(1.0)
        feat["credit_std_6m"] = g["total_credit_amount"].apply(
            lambda x: x.sort_index().iloc[-6:].std() if len(x) >= 6 else x.std()
        ).fillna(0.0)
        feat["median_debit_6m"] = g["total_debit_amount"].apply(
            lambda x: x.sort_index().iloc[-6:].median() if len(x) >= 6 else x.median()
        )

        # ── Phase 1: Regularity ───────────────────────────────────────────────
        feat["fixed_amount_similarity"] = g["total_credit_amount"].apply(
            lambda x: (np.abs(x - x.median()) <= x.median() * 0.10).mean()
            if x.median() > 0 else 0.0
        )
        feat["income_slope_6m"] = g["total_credit_amount"].apply(
            lambda x: _linear_slope(x.sort_index().iloc[-6:].values) if len(x) >= 3 else 0.0
        )
        feat["dormancy_gap_max"] = g["total_credit_amount"].apply(
            lambda x: _max_gap_months(x.values)
        )

        # ── Phase 1: Balance extended ─────────────────────────────────────────
        feat["min_balance_to_mean_ratio"] = g["eom_balance"].apply(
            lambda x: x.min() / x.mean() if x.mean() > 0 else 0.0
        ).fillna(0.0)
        feat["months_negative_balance"] = g["eom_balance"].apply(
            lambda x: (x < 0).sum()
        )
        feat["balance_volatility_6m"] = g["eom_balance"].apply(
            lambda x: (x.sort_index().iloc[-6:].std() / abs(x.sort_index().iloc[-6:].mean()))
            if len(x) >= 6 and x.sort_index().iloc[-6:].mean() != 0 else 1.0
        ).fillna(1.0)

        # ── Phase 1: Derived extensions ───────────────────────────────────────
        feat["liquidity_buffer_ratio"] = (
            feat["avg_eom_balance_6m"] / feat["avg_monthly_credit_12m"].replace(0, np.nan)
        ).fillna(0.0)

        feat["retention_ratio_6m"] = (
            (feat["avg_monthly_credit_6m"] - feat["avg_monthly_debit_6m"])
            / feat["avg_monthly_credit_6m"].replace(0, np.nan)
        ).fillna(0.0).clip(-1, 1)

        feat["pass_through_score"] = (1 - feat["retention_ratio_6m"]).clip(0, 1)

        feat["end_balance_ratio_6m"] = (
            feat["avg_eom_balance_6m"] / feat["avg_monthly_credit_12m"].replace(0, np.nan)
        ).fillna(0.0)

        feat["recurring_debit_share"] = (
            (feat["avg_commitment_amount_12m"] + feat["avg_recurring_expense_12m"])
            / feat["avg_total_debit_12m"].replace(0, np.nan)
        ).fillna(0.0).clip(0, 1)

        feat["usable_income_proxy"] = (
            feat["avg_monthly_credit_12m"]
            * feat["recurring_to_total_credit_ratio"].clip(0, 1)
            * (1 - feat["recurring_debit_share"])
        )

        feat["churn_intensity"] = (
            (feat["avg_monthly_credit_6m"] - feat["avg_monthly_debit_6m"]).clip(lower=0)
            / feat["avg_monthly_credit_12m"].replace(0, np.nan)
        ).fillna(0.0)

        feat["inflow_outflow_velocity"] = (
            (feat["avg_monthly_credit_6m"] - feat["avg_monthly_debit_6m"])
            / feat["avg_monthly_credit_6m"].replace(0, np.nan)
        ).fillna(0.0)

        feat["debit_credit_ratio_6m"] = (
            feat["median_debit_6m"] / feat["median_credit_6m"].replace(0, np.nan)
        ).fillna(1.0)

        feat["regularity_score"] = (1 - feat["cv_monthly_credit_12m"]).clip(0, 1)

        # ── Data tier ─────────────────────────────────────────────────────────
        feat["data_tier"] = feat["months_data_available"].apply(_assign_data_tier)

        return feat.reset_index()

    # ═══════════════════════════════════════════════════════════════════════════
    # INCOME LABEL GENERATION
    # ═══════════════════════════════════════════════════════════════════════════

    def _generate_verified_income(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate verified income labels correlated with transaction features.

        The income estimate should correlate with avg_monthly_credit for salaried
        customers (income ≈ credit) but diverge for SME (credit >> income due to
        business revenue cycling).
        """
        incomes = []

        for _, row in features.iterrows():
            seg = row["segment"]
            avg_credit = row["avg_monthly_credit_12m"]

            if seg in ("PAYROLL", "SALARY_LIKE"):
                # Income ~ recurring credits with small noise
                income = avg_credit * self.rng.uniform(0.88, 1.08)

            elif seg == "SME":
                # True income is a fraction of gross credit (business revenue)
                # Profit margin varies by business type
                income = avg_credit * self.rng.uniform(0.20, 0.55)

            elif seg == "GIG_FREELANCE":
                # Average credits underestimate income in good months
                income = avg_credit * self.rng.uniform(0.70, 1.20)

            elif seg == "PASSIVE_INVESTOR":
                # Investment income + some regular income
                income = (row["avg_investment_credit_12m"] * 12 / 12 +
                          row["avg_recurring_credit_12m"]) * self.rng.uniform(0.85, 1.10)

            else:
                income = avg_credit * self.rng.uniform(0.75, 1.10)

            # Clip to realistic range
            income = np.clip(income, INCOME_CLIP["min"], INCOME_CLIP["max"])
            incomes.append(round(income, 0))

        return pd.Series(incomes, index=features.index, name="verified_income", dtype=float)

    # ═══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _allocate_segments(self, n: int, mix: dict) -> Tuple[list, list]:
        """Allocate customers to segments based on mix proportions."""
        segs = list(mix.keys())
        proportions = np.array(list(mix.values()))
        proportions = proportions / proportions.sum()  # Normalise

        counts = (proportions * n).astype(int)
        # Fix rounding: assign remainder to largest segment
        remainder = n - counts.sum()
        counts[counts.argmax()] += remainder

        return segs, counts.tolist()

    def _sample_income(self, segment: str, n: int) -> np.ndarray:
        """Sample income from segment-specific log-normal distribution."""
        params = INCOME_PARAMS[segment]
        samples = self.rng.lognormal(mean=params["mu"], sigma=params["sigma"], size=n)
        return np.clip(samples, INCOME_CLIP["min"], INCOME_CLIP["max"])


# ── Module-level helper functions (used in aggregation lambdas) ──────────────

def _max_consecutive_positive(arr: np.ndarray) -> int:
    """Count max consecutive positive values."""
    max_streak = streak = 0
    for v in arr:
        if v > 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def _linear_slope(arr: np.ndarray) -> float:
    """Compute linear trend slope."""
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=float)
    try:
        return float(np.polyfit(x, arr, 1)[0])
    except Exception:
        return 0.0


def _max_gap_months(arr: np.ndarray) -> int:
    """Max consecutive months with zero/near-zero credit (dormancy gap)."""
    max_gap = gap = 0
    for v in arr:
        if v <= 0:
            gap += 1
            max_gap = max(max_gap, gap)
        else:
            gap = 0
    return max_gap


def _assign_data_tier(months: int) -> str:
    """Map months of history to data confidence tier label."""
    if months >= 12:
        return "12M"
    elif months >= 9:
        return "9M"
    elif months >= 6:
        return "6M"
    return "THIN"


# ── Convenience functions (used in notebooks) ────────────────────────────────

def generate_sample_features(n: int = 5_000, seed: int = 42) -> pd.DataFrame:
    """Quick-start: generate customer features with true segment labels."""
    return SampleDataGenerator(seed=seed).customer_features(n)


def generate_sample_training_data(
    n: int = 10_000, seed: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Quick-start: generate training features and verified income labels."""
    return SampleDataGenerator(seed=seed).training_data(n)


def generate_scoring_population(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    """Quick-start: generate full scoring population."""
    return SampleDataGenerator(seed=seed).scoring_population(n)


# ── CLI usage ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate sample data for development")
    parser.add_argument("--n", type=int, default=10_000, help="Number of customers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/sample/", help="Output directory")
    parser.add_argument(
        "--type",
        choices=["monthly", "features", "training", "scoring", "all"],
        default="all",
    )
    args = parser.parse_args()

    import os
    os.makedirs(args.output, exist_ok=True)

    gen = SampleDataGenerator(seed=args.seed)

    if args.type in ("monthly", "all"):
        df = gen.monthly_transactions(args.n)
        df.to_parquet(f"{args.output}/monthly_transactions.parquet", index=False)
        print(f"Monthly transactions: {df.shape}  →  {args.output}/monthly_transactions.parquet")

    if args.type in ("features", "all"):
        df = gen.customer_features(args.n)
        df.to_parquet(f"{args.output}/customer_features.parquet", index=False)
        print(f"Customer features:    {df.shape}  →  {args.output}/customer_features.parquet")

    if args.type in ("training", "all"):
        X, y = gen.training_data(args.n)
        X.to_parquet(f"{args.output}/train_features.parquet", index=False)
        y.to_frame().to_parquet(f"{args.output}/train_labels.parquet", index=False)
        print(f"Training data:        {X.shape}  →  {args.output}/train_*.parquet")

    if args.type in ("scoring", "all"):
        df = gen.scoring_population(args.n)
        df.to_parquet(f"{args.output}/scoring_population.parquet", index=False)
        print(f"Scoring population:   {df.shape}  →  {args.output}/scoring_population.parquet")
