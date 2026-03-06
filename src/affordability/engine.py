"""
Affordability Engine
─────────────────────
Computes Available Debt Servicing Capacity (ADSC) per customer.

Formula:
  Adjusted Income      = PolicyIncome (no BCI haircut — BCI acts on DSCR only)
  DSCR cap             = min(persona_dscr_cap, bci_band_dscr_cap)
  Allowable Obligation = Adjusted Income × DSCR cap
  Existing Obligations = From transaction commitment category
  ADSC                 = Allowable Obligation − Existing Obligations

ADSC > 0  AND  Adjusted Income ≥ 15,000 THB  →  AFFORDABLE

BCI band DSCR caps (behavioural confidence gate):
  HIGH      → 0.45   (best behaviour — can service higher share of income)
  MEDIUM    → 0.40
  LOW       → 0.35   (uncertain — tighter DSCR, refer for review)
  VERY_LOW  → 0.30   (hard decline likely via PolicyEngine)
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

    # Default persona DSCR caps — overridden by config when present.
    # L2 is intentionally HIGHER (0.45) because its income estimate is already a
    # P30 lower bound. Applying a tight DSCR on top of an already-conservative
    # estimate would double-penalise unstructured income customers.
    DEFAULT_PERSONA_DSCR = {
        "PAYROLL": 0.40,
        "L0":      0.40,
        "L1":      0.38,   # slight conservatism — retained inflow proxy label
        "L2":      0.45,   # relaxed — income estimate is already P30 lower bound
        "THIN":    0.35,
        "PT":      0.35,   # pass-through: formula-based income, moderate conservatism
        # Legacy segment names (before new persona system is live)
        "SALARY_LIKE":      0.40,
        "SME":              0.38,
        "GIG_FREELANCE":    0.45,
        "PASSIVE_INVESTOR": 0.40,
    }

    # BCI band DSCR caps — tighten when behavioural confidence is low.
    # final dscr_used = min(persona_dscr, bci_band_dscr)
    DEFAULT_BCI_BAND_DSCR = {
        "HIGH":     0.45,
        "MEDIUM":   0.40,
        "LOW":      0.35,
        "VERY_LOW": 0.30,
    }

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.bot_norms = config["bot_norms"]
        self.min_income = self.bot_norms["minimum_income_thb"]
        self.dscr_cap = self.bot_norms["dscr_cap"]

        # Persona-aware DSCR caps: read from config if present, else use defaults
        self.persona_dscr = {
            **self.DEFAULT_PERSONA_DSCR,
            **config.get("bot_norms", {}).get("dscr_cap_by_persona", {}),
        }

        # BCI band DSCR caps: read from config if present, else use defaults
        self.bci_band_dscr = {
            **self.DEFAULT_BCI_BAND_DSCR,
            **config.get("bci", {}).get("band_dscr_caps", {}),
        }

    def compute(
        self,
        bci_results: pd.DataFrame,
        features: pd.DataFrame,
        existing_obligations_col: str = "avg_commitment_amount_12m",
        dscr_override: Optional[float] = None,
        persona_col: str = "segment",
    ) -> pd.DataFrame:
        """
        Compute affordability for all customers.

        Parameters
        ----------
        bci_results : pd.DataFrame
            Output from BCIScorer.compute().
            Required: adjusted_income, bci_band, bci_policy
            Optional: segment (used for persona-aware DSCR)
        features : pd.DataFrame
            Customer-level feature matrix.
            Required: avg_commitment_amount_12m (existing debt obligations)
            Optional: cc_min_payment_amount (CC minimum payment obligations)
        existing_obligations_col : str
            Column in features containing monthly existing obligations (THB).
        dscr_override : float, optional
            Override the BOT DSCR cap for all customers (for scenario analysis).
        persona_col : str
            Column in bci_results carrying persona / segment label.
            Used to look up persona-specific DSCR cap.

        Returns
        -------
        pd.DataFrame with columns:
          adjusted_income, existing_obligations, allowable_obligation,
          adsc, dscr_used, dscr_actual, is_affordable,
          affordability_status, affordability_reason
        """
        result = pd.DataFrame(index=bci_results.index)

        # Adjusted income after BCI haircut (from BCI output)
        result["adjusted_income"] = bci_results["adjusted_income"]

        # ── Persona-aware DSCR ────────────────────────────────────────────────
        if dscr_override is not None:
            # Explicit override (scenario analysis / stress test)
            result["dscr_used"] = dscr_override
        elif persona_col in bci_results.columns:
            result["dscr_used"] = bci_results[persona_col].map(
                self.persona_dscr
            ).fillna(self.dscr_cap)
        else:
            result["dscr_used"] = self.dscr_cap

        # ── BCI band DSCR cap ─────────────────────────────────────────────────
        # BCI band tightens the DSCR: final = min(persona_dscr, bci_band_dscr).
        # This replaces the old income haircut — behavioural confidence now gates
        # how much of income can be committed to debt, not the income itself.
        if dscr_override is None and "bci_band" in bci_results.columns:
            bci_band_dscr = (
                bci_results["bci_band"].str.upper()
                .map(self.bci_band_dscr)
                .fillna(self.dscr_cap)
            )
            result["dscr_used"] = result["dscr_used"].combine(bci_band_dscr, min)

        # ── Existing obligations ──────────────────────────────────────────────
        # Base: loan/EMI commitments from transaction features
        result["existing_obligations"] = features[existing_obligations_col].fillna(0)

        # Add CC minimum payment when available (CC obligations are real debt)
        if "cc_min_payment_amount" in features.columns:
            result["existing_obligations"] = (
                result["existing_obligations"]
                + features["cc_min_payment_amount"].fillna(0)
            )

        # ── ADSC calculation ──────────────────────────────────────────────────
        result["allowable_obligation"] = (
            result["adjusted_income"] * result["dscr_used"]
        ).round(0)

        result["adsc"] = (
            result["allowable_obligation"] - result["existing_obligations"]
        ).round(0)

        result["dscr_actual"] = (
            result["existing_obligations"] / result["adjusted_income"].replace(0, np.nan)
        ).round(4)

        # Affordability flags
        result["meets_income_floor"] = result["adjusted_income"] >= self.min_income
        result["meets_dscr"] = result["adsc"] > 0

        result["is_affordable"] = result["meets_income_floor"] & result["meets_dscr"]

        # ── CC income floor sanity check ──────────────────────────────────────
        # If observed CC spend exceeds 80% of income estimate, the estimate is
        # likely too low (customer is spending more than we think they earn).
        # Flag for review — do NOT auto-decline, but route to MANUAL_REVIEW.
        result["cc_income_floor_breach"] = False
        if "cc_spend_6m" in features.columns:
            cc_monthly_spend = features["cc_spend_6m"].fillna(0) / 6.0
            breach = cc_monthly_spend > result["adjusted_income"] * 0.80
            result["cc_income_floor_breach"] = breach
            if breach.any():
                logger.warning(
                    f"AffordabilityEngine: {breach.sum():,} customers have "
                    f"CC spend > 80% of income estimate — flagged for review"
                )

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
        if row["is_affordable"] and not row.get("cc_income_floor_breach", False):
            return "PASS"
        reasons = []
        if not row["meets_income_floor"]:
            reasons.append(f"INCOME_BELOW_FLOOR({self.min_income:,} THB)")
        if not row["meets_dscr"]:
            reasons.append(f"DSCR_EXCEEDED(actual={row.get('dscr_actual', 0):.0%})")
        if row.get("cc_income_floor_breach", False):
            reasons.append("CC_SPEND_EXCEEDS_80PCT_OF_INCOME")
        return " | ".join(reasons) if reasons else "PASS"

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
