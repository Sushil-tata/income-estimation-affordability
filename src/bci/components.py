"""
BCI Component Scorers
──────────────────────
Individual scoring functions for each BCI dimension.

BCI = w1×Stability + w2×SegmentClarity + w3×DataRichness
        + w4×ModelConfidence + w5×BehavioralConsistency

All components return a score in [0, 1].
"""

import pandas as pd
import numpy as np
from typing import Optional


class BCIComponents:
    """
    Computes all five BCI component scores at the customer level.

    Each component is normalized to [0, 1] before weighting.
    """

    # ── COMPONENT 1: INCOME STABILITY ──────────────────────────────────────
    @staticmethod
    def income_stability(
        cv_monthly_credit: pd.Series,
        months_with_zero_credit: pd.Series,
        months_data_available: pd.Series,
        income_interval_width: pd.Series,
        income_estimate: pd.Series,
    ) -> pd.Series:
        """
        Measures how regular and predictable income inflows are.

        Higher stability → lower CV, fewer zero-credit months,
        narrower prediction interval relative to income.

        Returns score in [0, 1].
        """
        # Sub-score 1: CV of monthly credits (lower = more stable)
        cv_score = 1 - cv_monthly_credit.clip(0, 1)

        # Sub-score 2: Consistency — fraction of months WITH credit
        credit_month_ratio = 1 - (
            months_with_zero_credit / months_data_available.replace(0, 1)
        )
        credit_month_ratio = credit_month_ratio.clip(0, 1)

        # Sub-score 3: Prediction interval tightness
        relative_interval = income_interval_width / income_estimate.replace(0, np.nan)
        interval_score = 1 - relative_interval.clip(0, 1).fillna(0)

        # Weighted average of sub-scores
        stability_score = 0.40 * cv_score + 0.35 * credit_month_ratio + 0.25 * interval_score
        return stability_score.clip(0, 1)

    # ── COMPONENT 2: SEGMENT CLARITY ───────────────────────────────────────
    @staticmethod
    def segment_clarity(
        segment: pd.Series,
        dominant_credit_source_share: pd.Series,
        cv_monthly_credit: pd.Series,
        months_with_salary_pattern: pd.Series,
        months_data_available: pd.Series,
    ) -> pd.Series:
        """
        Measures how cleanly a customer fits their assigned persona/segment.

        PAYROLL → highest clarity (payroll verified).
        L0 (stable structured) → high clarity.
        L1 (structured irregular) → medium clarity.
        L2 (volatile/informal) → lower clarity.
        THIN → very low clarity (insufficient data).
        """
        # Persona base confidence
        # Phase 2+ labels take priority; legacy labels kept for backward compat.
        segment_base = {
            # Phase 2+ persona labels
            "PAYROLL": 1.00,
            "L0":      0.85,   # Stable structured — income pattern is clear
            "L1":      0.65,   # Structured irregular — moderate confidence
            "L2":      0.45,   # Volatile/informal — lower confidence
            "THIN":    0.20,   # Thin file — very little signal
            # Legacy segment labels (backward compatibility)
            "SALARY_LIKE":      0.85,
            "SME":              0.60,
            "GIG_FREELANCE":    0.50,
            "PASSIVE_INVESTOR": 0.65,
        }
        base_clarity = segment.map(segment_base).fillna(0.50)

        # Modifier: dominant credit source share (concentration)
        concentration_modifier = (dominant_credit_source_share - 0.5).clip(-0.5, 0.5) * 0.3

        # Modifier: salary pattern strength
        salary_pattern_ratio = months_with_salary_pattern / months_data_available.replace(0, 1)
        pattern_modifier = salary_pattern_ratio * 0.15

        # Penalty: very high CV reduces clarity
        cv_penalty = (cv_monthly_credit - 0.5).clip(0, 1) * 0.20

        clarity_score = (base_clarity + concentration_modifier + pattern_modifier - cv_penalty)
        return clarity_score.clip(0, 1)

    # ── COMPONENT 3: DATA RICHNESS ──────────────────────────────────────────
    @staticmethod
    def data_richness(
        months_data_available: pd.Series,
        transaction_count_avg_monthly: pd.Series,
        max_months: int = 12,
        max_tx_monthly: float = 50.0,
    ) -> pd.Series:
        """
        Measures the quality and depth of available data.

        More months + higher transaction density → richer data → higher BCI.
        """
        # History depth score
        history_score = (months_data_available / max_months).clip(0, 1)

        # Transaction density score
        tx_density_score = (transaction_count_avg_monthly / max_tx_monthly).clip(0, 1)

        richness_score = 0.60 * history_score + 0.40 * tx_density_score
        return richness_score.clip(0, 1)

    # ── COMPONENT 4: MODEL CONFIDENCE ──────────────────────────────────────
    @staticmethod
    def model_confidence(
        band_model_confidence: pd.Series,
        income_source: pd.Series,
    ) -> pd.Series:
        """
        Reflects how certain the income band classifier was.

        PAYROLL → full confidence (1.0).
        ESTIMATED → use classifier's margin (top_prob - 2nd_prob).
        """
        confidence = band_model_confidence.clip(0, 1)

        # Payroll customers have fully certain income → override to 1.0
        payroll_mask = income_source == "PAYROLL"
        confidence[payroll_mask] = 1.0

        return confidence

    # ── COMPONENT 5: BEHAVIORAL CONSISTENCY ────────────────────────────────
    @staticmethod
    def behavioral_consistency(
        income_estimate: pd.Series,
        avg_total_debit_12m: pd.Series,
        avg_eom_balance_3m: pd.Series,
        savings_rate_proxy: pd.Series,
        segment: pd.Series,
    ) -> pd.Series:
        """
        Checks if spending and balance patterns are consistent with
        the estimated income level.

        A customer earning 50K THB/month spending 200K/month is inconsistent.
        A customer with 0 balance on 80K income is slightly inconsistent.
        """
        # Spend-to-income ratio (should be < 1.0 for most customers)
        spend_to_income = avg_total_debit_12m / income_estimate.replace(0, np.nan)

        # Consistency score: ratio close to 0.4–0.9 is normal
        # Above 1.2 → overspend → penalize
        # Below 0.1 → passive → slight penalty (may underestimate income)
        consistency = pd.Series(1.0, index=income_estimate.index)

        overspend_mask = spend_to_income > 1.2
        underspend_mask = spend_to_income < 0.10
        normal_mask = (spend_to_income >= 0.10) & (spend_to_income <= 1.2)

        consistency[overspend_mask] = (
            1 - (spend_to_income[overspend_mask] - 1.2).clip(0, 1) * 0.8
        )
        consistency[underspend_mask] = 0.70
        consistency[normal_mask] = 1.0

        # Balance plausibility: balance should bear some relationship to income
        balance_to_income = avg_eom_balance_3m / income_estimate.replace(0, np.nan)
        balance_plausibility = balance_to_income.clip(0, 3) / 3  # Max 1 at 3x income in balance

        final_consistency = 0.70 * consistency + 0.30 * balance_plausibility
        return final_consistency.fillna(0.5).clip(0, 1)
