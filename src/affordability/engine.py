"""
Affordability Engine
─────────────────────
Computes Available Debt Servicing Capacity (ADSC) per customer.

Formula:
  Adjusted Income      = income_estimate × BCI haircut
  Allowable Obligation = Adjusted Income × DSCR cap (BOT norm ~40%)
  Existing Obligations = From transaction commitment category
  ADSC                 = Allowable Obligation − Existing Obligations

ADSC > 0  AND  Adjusted Income ≥ 15,000 THB  →  AFFORDABLE
"""

import pandas as pd
import numpy as np
import yaml
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AffordabilityEngine:
    """
    Computes affordability for each customer using BOT norms.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.bot_norms = config["bot_norms"]
        self.min_income = self.bot_norms["minimum_income_thb"]
        self.dscr_cap = self.bot_norms["dscr_cap"]

    def compute(
        self,
        bci_results: pd.DataFrame,
        features: pd.DataFrame,
        existing_obligations_col: str = "avg_commitment_amount_12m",
        dscr_override: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Compute affordability for all customers.

        Parameters
        ----------
        bci_results : pd.DataFrame
            Output from BCIScorer.compute().
            Required: adjusted_income, bci_band, bci_policy
        features : pd.DataFrame
            Customer-level feature matrix.
            Required: avg_commitment_amount_12m (existing debt obligations)
        existing_obligations_col : str
            Column in features containing monthly existing obligations (THB).
        dscr_override : float, optional
            Override the BOT DSCR cap (for scenario analysis).

        Returns
        -------
        pd.DataFrame with columns:
          adjusted_income, existing_obligations, allowable_obligation,
          adsc, dscr_used, dscr_actual, is_affordable,
          affordability_status, affordability_reason
        """
        dscr = dscr_override or self.dscr_cap

        result = pd.DataFrame(index=bci_results.index)

        # Adjusted income after BCI haircut (from BCI output)
        result["adjusted_income"] = bci_results["adjusted_income"]

        # Existing monthly obligations from transaction features
        result["existing_obligations"] = features[existing_obligations_col].fillna(0)

        # Allowable total monthly obligation under BOT DSCR
        result["allowable_obligation"] = (result["adjusted_income"] * dscr).round(0)

        # Available debt servicing capacity
        result["adsc"] = (result["allowable_obligation"] - result["existing_obligations"]).round(0)

        # DSCR actually consumed by existing obligations
        result["dscr_used"] = self.dscr_cap
        result["dscr_actual"] = (
            result["existing_obligations"] / result["adjusted_income"].replace(0, np.nan)
        ).round(4)

        # Affordability flags
        result["meets_income_floor"] = result["adjusted_income"] >= self.min_income
        result["meets_dscr"] = result["adsc"] > 0

        result["is_affordable"] = result["meets_income_floor"] & result["meets_dscr"]

        # Human-readable status and reason
        result["affordability_status"] = result["is_affordable"].map(
            {True: "AFFORDABLE", False: "NOT_AFFORDABLE"}
        )
        result["affordability_reason"] = result.apply(self._reason, axis=1)

        logger.info(
            f"Affordability computed: "
            f"{result['is_affordable'].sum():,} affordable / {len(result):,} total "
            f"({result['is_affordable'].mean():.1%})"
        )

        return result

    def _reason(self, row: pd.Series) -> str:
        if row["is_affordable"]:
            return "PASS"
        reasons = []
        if not row["meets_income_floor"]:
            reasons.append(f"INCOME_BELOW_FLOOR({self.min_income:,} THB)")
        if not row["meets_dscr"]:
            reasons.append(f"DSCR_EXCEEDED(actual={row['dscr_actual']:.0%})")
        return " | ".join(reasons)

    def stress_test(
        self,
        affordability_result: pd.DataFrame,
        dscr_scenarios: list = None,
    ) -> pd.DataFrame:
        """
        Run affordability under different DSCR cap scenarios.
        Useful for policy calibration.

        Parameters
        ----------
        dscr_scenarios : list
            List of DSCR cap values to test (e.g., [0.30, 0.35, 0.40, 0.45]).

        Returns
        -------
        pd.DataFrame with approval rate per scenario.
        """
        scenarios = dscr_scenarios or [0.30, 0.35, 0.40, 0.45, 0.50]
        results = []

        for dscr in scenarios:
            allowable = affordability_result["adjusted_income"] * dscr
            adsc = allowable - affordability_result["existing_obligations"]
            meets_dscr = adsc > 0
            meets_floor = affordability_result["adjusted_income"] >= self.min_income
            affordable_rate = (meets_dscr & meets_floor).mean()

            results.append({
                "dscr_cap": dscr,
                "approval_rate": round(affordable_rate * 100, 2),
                "approved_count": int((meets_dscr & meets_floor).sum()),
            })

        return pd.DataFrame(results)

    def get_summary(self, result: pd.DataFrame) -> pd.DataFrame:
        """Affordability summary by BCI band."""
        merged = result.copy()
        summary = merged.groupby("affordability_status").agg(
            count=("is_affordable", "count"),
            avg_adjusted_income=("adjusted_income", "mean"),
            avg_adsc=("adsc", "mean"),
            avg_dscr_actual=("dscr_actual", "mean"),
        ).reset_index()
        summary["pct"] = (summary["count"] / len(result) * 100).round(2)
        return summary
