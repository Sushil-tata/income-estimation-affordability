"""
Policy Engine
──────────────
Translates BCI + Affordability outputs into a final customer decision.

Decision outcomes:
  STP_APPROVE      : Straight-through approve — high confidence, affordable
  STP_DECLINE      : Straight-through decline — not affordable or below BOT floor
  MANUAL_REVIEW    : Refer for additional verification (low BCI, borderline)
  REFER_INCOME_VERIFY : Adequate ADSC but BCI too low to trust income estimate
"""

import pandas as pd
import numpy as np
import yaml
import logging

logger = logging.getLogger(__name__)

# Decision labels
STP_APPROVE = "STP_APPROVE"
STP_DECLINE = "STP_DECLINE"
MANUAL_REVIEW = "MANUAL_REVIEW"
REFER_INCOME_VERIFY = "REFER_INCOME_VERIFY"


class PolicyEngine:
    """
    Final policy decision engine combining BCI and Affordability outputs.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.bci_bands = config["bci"]["bands"]
        self.min_income = config["bot_norms"]["minimum_income_thb"]

    def decide(
        self,
        bci_results: pd.DataFrame,
        affordability_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply policy rules to produce final decision per customer.

        Parameters
        ----------
        bci_results : pd.DataFrame
            Output from BCIScorer.compute().
            Required: bci_band, bci_score, bci_policy, adjusted_income
        affordability_results : pd.DataFrame
            Output from AffordabilityEngine.compute().
            Required: is_affordable, adsc, affordability_reason

        Returns
        -------
        pd.DataFrame with columns:
          final_decision, decision_reason, bci_score, bci_band,
          adjusted_income, adsc, is_affordable
        """
        result = pd.DataFrame(index=bci_results.index)
        result["bci_score"] = bci_results["bci_score"]
        result["bci_band"] = bci_results["bci_band"]
        result["bci_policy"] = bci_results["bci_policy"]
        result["adjusted_income"] = bci_results["adjusted_income"]
        result["adsc"] = affordability_results["adsc"]
        result["is_affordable"] = affordability_results["is_affordable"]
        result["affordability_reason"] = affordability_results["affordability_reason"]

        # Apply decision matrix
        decisions = result.apply(self._decision_rule, axis=1)
        result["final_decision"] = decisions.apply(lambda x: x[0])
        result["decision_reason"] = decisions.apply(lambda x: x[1])

        logger.info("Policy decisions applied:")
        for dec, cnt in result["final_decision"].value_counts().items():
            logger.info(f"  {dec}: {cnt:,} ({cnt/len(result):.1%})")

        return result

    def _decision_rule(self, row: pd.Series):
        """
        Decision matrix:

        BCI Band  | Affordable | Decision
        ----------|------------|-------------------
        HIGH      | YES        | STP_APPROVE
        HIGH      | NO         | STP_DECLINE
        MEDIUM    | YES        | STP_APPROVE
        MEDIUM    | NO         | STP_DECLINE
        LOW       | YES        | REFER_INCOME_VERIFY
        LOW       | NO         | STP_DECLINE
        VERY_LOW  | *          | STP_DECLINE / REFER
        """
        bci_band = str(row["bci_band"]).upper()
        is_affordable = row["is_affordable"]
        bci_score = row["bci_score"]

        # VERY LOW BCI — hard decline or refer regardless of affordability
        if bci_band == "VERY_LOW" or bci_score < 40:
            return (STP_DECLINE, "BCI_TOO_LOW_TO_TRUST_INCOME")

        # Not affordable — decline
        if not is_affordable:
            return (STP_DECLINE, f"NOT_AFFORDABLE | {row['affordability_reason']}")

        # Affordable + HIGH/MEDIUM BCI → STP approve
        if bci_band in ("HIGH", "MEDIUM"):
            return (STP_APPROVE, f"AFFORDABLE | BCI_{bci_band}")

        # Affordable + LOW BCI → refer for income verification
        if bci_band == "LOW":
            return (REFER_INCOME_VERIFY, "AFFORDABLE_BUT_LOW_BCI_INCOME_UNVERIFIED")

        # Default
        return (MANUAL_REVIEW, "UNCLASSIFIED_CASE")

    def get_decision_summary(self, result: pd.DataFrame) -> pd.DataFrame:
        """Return decision distribution summary."""
        summary = result["final_decision"].value_counts().reset_index()
        summary.columns = ["decision", "count"]
        summary["pct"] = (summary["count"] / len(result) * 100).round(2)

        # Add average BCI and income per decision
        avg_stats = result.groupby("final_decision").agg(
            avg_bci=("bci_score", "mean"),
            avg_income=("adjusted_income", "mean"),
            avg_adsc=("adsc", "mean"),
        ).reset_index()
        avg_stats.columns = ["decision", "avg_bci", "avg_income", "avg_adsc"]

        return summary.merge(avg_stats, on="decision").sort_values("count", ascending=False)

    def get_full_output(
        self,
        bci_results: pd.DataFrame,
        affordability_results: pd.DataFrame,
        income_results: pd.DataFrame,
        features: pd.DataFrame,
        segment_col: str = "segment",
    ) -> pd.DataFrame:
        """
        Produce final consolidated output for all customers.

        Combines all pipeline outputs into a single customer-level record.
        """
        decisions = self.decide(bci_results, affordability_results)

        output = pd.DataFrame(index=features.index)
        output["segment"] = features[segment_col]

        # Income — some columns only exist when using IncomeEstimationPipeline;
        # fall back gracefully when using SegmentModelTrainer or manual paths.
        output["income_source"] = income_results["income_source"]
        output["income_estimate_raw"] = income_results["income_estimate"]
        output["income_band"] = (
            income_results["income_band"]
            if "income_band" in income_results.columns
            else pd.Series("UNKNOWN", index=income_results.index)
        )
        output["income_q25"] = (
            income_results["income_q25"]
            if "income_q25" in income_results.columns
            else income_results["income_estimate"] * 0.85
        )
        output["income_q75"] = (
            income_results["income_q75"]
            if "income_q75" in income_results.columns
            else income_results["income_estimate"] * 1.15
        )

        # BCI
        output["bci_score"] = bci_results["bci_score"]
        output["bci_band"] = bci_results["bci_band"]
        output["adjusted_income"] = bci_results["adjusted_income"]
        output["income_haircut"] = bci_results["income_haircut"]

        # Affordability
        output["existing_obligations"] = affordability_results["existing_obligations"]
        output["allowable_obligation"] = affordability_results["allowable_obligation"]
        output["adsc"] = affordability_results["adsc"]
        output["dscr_actual"] = affordability_results["dscr_actual"]
        output["is_affordable"] = affordability_results["is_affordable"]

        # Final decision
        output["final_decision"] = decisions["final_decision"]
        output["decision_reason"] = decisions["decision_reason"]

        return output
