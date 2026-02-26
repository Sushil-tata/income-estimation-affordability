"""
BCI Scorer
────────────
Aggregates the five BCI components into a final score [0–100]
and assigns a BCI band with associated policy and haircut.

BCI → Business Confidence Index:
  Answers: "How much should we trust this income estimate?"

BCI Bands:
  80–100 : HIGH       → STP, income at face value
  60–79  : MEDIUM     → STP, 80% haircut
  40–59  : LOW        → Manual review, 60% haircut
  0–39   : VERY LOW   → Decline / Refer
"""

import pandas as pd
import numpy as np
import yaml
import logging
from typing import Optional

from .components import BCIComponents

logger = logging.getLogger(__name__)


class BCIScorer:
    """
    Computes final BCI score and band for each customer.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.weights = config["bci"]["weights"]
        self.bands_config = config["bci"]["bands"]
        self.segment_caps = config["bci"]["segment_bci_caps"]
        self.components = BCIComponents()

    def compute(
        self,
        features: pd.DataFrame,
        income_results: pd.DataFrame,
        segment_col: str = "segment",
    ) -> pd.DataFrame:
        """
        Compute BCI for all customers.

        Parameters
        ----------
        features : pd.DataFrame
            Customer-level feature matrix (from FeatureEngineer).
        income_results : pd.DataFrame
            Output from IncomeEstimationPipeline.predict().
            Required columns:
              income_estimate, income_q25, income_q75,
              income_interval_width, model_confidence, income_source
        segment_col : str
            Column name for segment in features dataframe.

        Returns
        -------
        pd.DataFrame with columns:
          bci_stability, bci_segment_clarity, bci_data_richness,
          bci_model_confidence, bci_behavioral_consistency,
          bci_raw_score, bci_score (segment-capped),
          bci_band, bci_policy, income_haircut, adjusted_income
        """
        result = pd.DataFrame(index=features.index)

        # ── Component 1: Income Stability ──────────────────────────────────
        result["bci_stability"] = BCIComponents.income_stability(
            cv_monthly_credit=features["cv_monthly_credit_12m"],
            months_with_zero_credit=features["months_with_zero_credit"],
            months_data_available=features["months_data_available"],
            income_interval_width=income_results["income_interval_width"],
            income_estimate=income_results["income_estimate"],
        )

        # ── Component 2: Segment Clarity ───────────────────────────────────
        result["bci_segment_clarity"] = BCIComponents.segment_clarity(
            segment=features[segment_col],
            dominant_credit_source_share=features["dominant_credit_source_share"],
            cv_monthly_credit=features["cv_monthly_credit_12m"],
            months_with_salary_pattern=features["months_with_salary_pattern"],
            months_data_available=features["months_data_available"],
        )

        # ── Component 3: Data Richness ──────────────────────────────────────
        result["bci_data_richness"] = BCIComponents.data_richness(
            months_data_available=features["months_data_available"],
            transaction_count_avg_monthly=features["transaction_count_avg_monthly"],
        )

        # ── Component 4: Model Confidence ──────────────────────────────────
        result["bci_model_confidence"] = BCIComponents.model_confidence(
            band_model_confidence=income_results["model_confidence"],
            income_source=income_results["income_source"],
        )

        # ── Component 5: Behavioral Consistency ────────────────────────────
        result["bci_behavioral_consistency"] = BCIComponents.behavioral_consistency(
            income_estimate=income_results["income_estimate"],
            avg_total_debit_12m=features["avg_total_debit_12m"],
            avg_eom_balance_3m=features["avg_eom_balance_3m"],
            savings_rate_proxy=features["savings_rate_proxy"],
            segment=features[segment_col],
        )

        # ── Weighted Aggregation → Raw BCI Score [0–100] ───────────────────
        w = self.weights
        result["bci_raw_score"] = 100 * (
            w["income_stability"] * result["bci_stability"]
            + w["segment_clarity"] * result["bci_segment_clarity"]
            + w["data_richness"] * result["bci_data_richness"]
            + w["model_confidence"] * result["bci_model_confidence"]
            + w["behavioral_consistency"] * result["bci_behavioral_consistency"]
        )

        # ── Segment Cap ────────────────────────────────────────────────────
        result["segment"] = features[segment_col]
        result["segment_bci_cap"] = result["segment"].map(self.segment_caps).fillna(70)
        result["bci_score"] = result[["bci_raw_score", "segment_bci_cap"]].min(axis=1).round(1)

        # ── BCI Band Assignment ────────────────────────────────────────────
        band_assignments = result["bci_score"].apply(
            lambda s: pd.Series(self._assign_band(s),
                                index=["bci_band", "bci_policy", "income_haircut"])
        )
        result["bci_band"] = band_assignments["bci_band"]
        result["bci_policy"] = band_assignments["bci_policy"]
        result["income_haircut"] = band_assignments["income_haircut"]

        # ── Adjusted Income ────────────────────────────────────────────────
        result["adjusted_income"] = (
            income_results["income_estimate"] * result["income_haircut"]
        ).round(0)

        logger.info(f"BCI computed for {len(result):,} customers")
        logger.info(f"BCI band distribution:\n{result['bci_band'].value_counts()}")

        return result

    def _assign_band(self, score: float):
        """Assign BCI band, policy, and haircut based on score."""
        for band_name, cfg in self.bands_config.items():
            if cfg["min"] <= score <= cfg["max"]:
                return band_name.upper(), cfg["policy"], cfg["haircut"]
        return "VERY_LOW", "DECLINE_OR_REFER", 0.0

    def get_summary(self, bci_result: pd.DataFrame) -> pd.DataFrame:
        """Return BCI band distribution and average scores per band."""
        summary = bci_result.groupby("bci_band").agg(
            count=("bci_score", "count"),
            avg_bci=("bci_score", "mean"),
            avg_stability=("bci_stability", "mean"),
            avg_data_richness=("bci_data_richness", "mean"),
            avg_model_confidence=("bci_model_confidence", "mean"),
        ).reset_index()
        summary["pct"] = (summary["count"] / len(bci_result) * 100).round(2)
        return summary

    def get_component_breakdown(self, bci_result: pd.DataFrame, customer_id) -> dict:
        """Return full BCI component breakdown for a single customer (for explainability)."""
        row = bci_result.loc[customer_id]
        return {
            "bci_score": row["bci_score"],
            "bci_band": row["bci_band"],
            "bci_policy": row["bci_policy"],
            "income_haircut": row["income_haircut"],
            "adjusted_income": row["adjusted_income"],
            "components": {
                "stability": {"score": row["bci_stability"], "weight": self.weights["income_stability"]},
                "segment_clarity": {"score": row["bci_segment_clarity"], "weight": self.weights["segment_clarity"]},
                "data_richness": {"score": row["bci_data_richness"], "weight": self.weights["data_richness"]},
                "model_confidence": {"score": row["bci_model_confidence"], "weight": self.weights["model_confidence"]},
                "behavioral_consistency": {"score": row["bci_behavioral_consistency"], "weight": self.weights["behavioral_consistency"]},
            },
            "segment_cap_applied": row["bci_raw_score"] > row["segment_bci_cap"],
        }
