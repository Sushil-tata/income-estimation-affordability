"""
Layer 9C — Offer Selector
──────────────────────────
Generate ranked candidate offers and select the optimal executable action.

Design
──────
  1. Generate candidate set: for each eligible tenor, compute an adjusted
     offer amount = max_loan_{n}m × confidence_multiplier, then filter to
     candidates that satisfy:
       • offer_amount ≥ min_loan_amount_thb
       • tenor ∈ eligible tenors for the persona
       • dscr-adjusted EMI ≤ max_emi (capacity feasibility re-check)

  2. Score each candidate with an explicit, auditable formula:
       score = offer_amount × tenor_preference_weight[n]
     where tenor_preference_weight is a calibration parameter [PROVISIONAL].
     No latent model — every score derivation is fully reproducible.

  3. Select the highest-scoring feasible candidate.
     Ties broken by shorter tenor (lower risk).

  4. If no candidate is feasible, action_code = REFER.

  5. Persona routing (hard rules, no model):
       PAYROLL / L0 / L1    → STP_APPROVE if max_emi > stp_min_emi_thb
       L2 / PT              → MANUAL_REVIEW (income uncertainty too high)
       THIN / SPARSE        → REFER (policy decision — no override)
       Any negative capacity → REFER

  6. Card line: included when persona ∈ {PAYROLL, L0, L1} and
     max_card_line ≥ min_card_line_thb.

Output schema (new fields — never overwrites upstream)
──────────────────────────────────────────────────────
  action_code                 : ActionCode enum string
  verification_intensity      : VerificationIntensity enum string
  offer_amount_recommended    : float  (adjusted offer = max_loan × confidence_mult)
  recommended_tenor_months    : int    (winning tenor)
  max_offerable_amount        : float  (max_loan_{best_tenor}m — pre-multiplier)
  include_card_line           : bool
  card_line_recommended       : float  (≥ 0)
  stp_flag                    : bool   (True → no human review required)
  optimization_reason_codes   : list[str]  (audit trail)
"""

import logging
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ActionCode(str, Enum):
    STP_APPROVE    = "STP_APPROVE"     # Straight-through approval
    MANUAL_REVIEW  = "MANUAL_REVIEW"   # Approve subject to review
    REFER          = "REFER"           # Refer — insufficient capacity or data


class VerificationIntensity(str, Enum):
    NONE     = "NONE"      # STP — no verification
    LIGHT    = "LIGHT"     # Payslip check only
    STANDARD = "STANDARD"  # Standard income verification
    FULL     = "FULL"      # Full document + manual review


# Persona → (action eligibility, verification intensity)
_PERSONA_POLICY: Dict[str, Dict] = {
    "PAYROLL": {"eligible": True,  "base_action": ActionCode.STP_APPROVE,   "verification": VerificationIntensity.NONE},
    "L0":      {"eligible": True,  "base_action": ActionCode.STP_APPROVE,   "verification": VerificationIntensity.LIGHT},
    "L1":      {"eligible": True,  "base_action": ActionCode.STP_APPROVE,   "verification": VerificationIntensity.STANDARD},
    "L2":      {"eligible": True,  "base_action": ActionCode.MANUAL_REVIEW, "verification": VerificationIntensity.STANDARD},
    "PT":      {"eligible": True,  "base_action": ActionCode.MANUAL_REVIEW, "verification": VerificationIntensity.FULL},
    "THIN":    {"eligible": False, "base_action": ActionCode.REFER,         "verification": VerificationIntensity.FULL},
    "SPARSE":  {"eligible": False, "base_action": ActionCode.REFER,         "verification": VerificationIntensity.FULL},
}
_DEFAULT_PERSONA_POLICY = {
    "eligible": False, "base_action": ActionCode.REFER, "verification": VerificationIntensity.FULL,
}

# Personas allowed STP card line inclusion
_CARD_LINE_ELIGIBLE_PERSONAS = {"PAYROLL", "L0", "L1"}


class OfferSelector:
    """
    Layer 9C: Candidate offer generation and ranked selection.

    Parameters
    ----------
    product_tenors_months : list[int]
        Tenors to generate candidates for.
    tenor_preference_weights : dict[int, float], optional
        Scoring weight per tenor [PROVISIONAL].
        Higher weight → preferred at equal offer amount.
        Default: uniform (1.0 for all tenors).
    min_loan_amount_thb : float
        Minimum loan amount to include a candidate.  [PROVISIONAL]
    stp_min_emi_thb : float
        Minimum max_emi to qualify for STP (rather than MANUAL_REVIEW).  [PROVISIONAL]
    min_card_line_thb : float
        Minimum card line to include card offer.  [PROVISIONAL]
    reference_rate_annual : float
        Rate used to back-check EMI feasibility (must match CapacityEngine).
    """

    def __init__(
        self,
        product_tenors_months: Optional[List[int]] = None,
        tenor_preference_weights: Optional[Dict[int, float]] = None,
        min_loan_amount_thb: float = 10_000.0,     # [PROVISIONAL]
        stp_min_emi_thb: float = 1_000.0,          # [PROVISIONAL]
        min_card_line_thb: float = 5_000.0,        # [PROVISIONAL]
        reference_rate_annual: float = 0.18,
    ):
        self.product_tenors_months   = product_tenors_months or [12, 24, 36, 48, 60]
        self.tenor_preference_weights = tenor_preference_weights or {
            t: 1.0 for t in self.product_tenors_months
        }
        # Fill in any missing tenors with 1.0
        for t in self.product_tenors_months:
            self.tenor_preference_weights.setdefault(t, 1.0)
        self.min_loan_amount_thb   = min_loan_amount_thb
        self.stp_min_emi_thb       = stp_min_emi_thb
        self.min_card_line_thb     = min_card_line_thb
        self.reference_rate_annual = reference_rate_annual

    # ── Public API ─────────────────────────────────────────────────────────────

    def select(
        self,
        capacity_output: pd.DataFrame,
        confidence_output: pd.DataFrame,
        final_output: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Select optimal offer per customer.

        Parameters
        ----------
        capacity_output : pd.DataFrame
            Output from CapacityEngine.compute(). Contains max_loan_{n}m, max_emi, max_card_line.
        confidence_output : pd.DataFrame
            Output from ConfidenceEngine.compute(). Contains confidence_multiplier.
        final_output : pd.DataFrame
            Upstream frozen output. Contains persona.

        Returns
        -------
        pd.DataFrame
            One row per customer. New columns only.
        """
        idx = final_output.index
        records = []

        persona_series     = self._safe_str_col(final_output,    "persona",               "UNKNOWN")
        max_emi_series     = self._safe_col(capacity_output,     "max_emi",               0.0)
        card_line_series   = self._safe_col(capacity_output,     "max_card_line",         0.0)
        conf_mult_series   = self._safe_col(confidence_output,   "confidence_multiplier", 0.0)

        for cid in idx:
            persona   = str(persona_series.loc[cid])
            max_emi   = float(max_emi_series.loc[cid])
            card_line = float(card_line_series.loc[cid])
            conf_mult = float(conf_mult_series.loc[cid])

            rec = self._select_single(cid, persona, max_emi, card_line, conf_mult, capacity_output)
            records.append(rec)

        out = pd.DataFrame(records, index=idx)
        logger.info(
            f"OfferSelector: {len(out):,} customers | "
            f"STP={int((out['action_code'] == ActionCode.STP_APPROVE.value).sum()):,} | "
            f"MANUAL={(out['action_code'] == ActionCode.MANUAL_REVIEW.value).sum():,} | "
            f"REFER={(out['action_code'] == ActionCode.REFER.value).sum():,}"
        )
        return out

    # ── Internal ───────────────────────────────────────────────────────────────

    def _select_single(
        self,
        cid: str,
        persona: str,
        max_emi: float,
        card_line: float,
        conf_mult: float,
        capacity_output: pd.DataFrame,
    ) -> dict:
        """Return offer record for one customer."""
        policy = _PERSONA_POLICY.get(persona, _DEFAULT_PERSONA_POLICY)
        reason_codes: List[str] = []

        # ── Hard REFER paths ──────────────────────────────────────────────────
        if not policy["eligible"]:
            reason_codes.append(f"PERSONA_{persona}_INELIGIBLE")
            return self._refer_record(policy, card_line, persona, reason_codes)

        if max_emi <= 0:
            reason_codes.append("ZERO_CAPACITY")
            return self._refer_record(policy, card_line, persona, reason_codes)

        if conf_mult <= 0:
            reason_codes.append("ZERO_CONFIDENCE_MULTIPLIER")
            return self._refer_record(policy, card_line, persona, reason_codes)

        # ── Generate candidates ───────────────────────────────────────────────
        candidates = []
        for tenor in self.product_tenors_months:
            col = f"max_loan_{tenor}m"
            if col not in capacity_output.columns:
                continue
            max_loan = float(capacity_output.loc[cid, col])
            offer_amt = round(max_loan * conf_mult, 0)

            if offer_amt < self.min_loan_amount_thb:
                continue

            # EMI feasibility re-check using back-calculated EMI
            emi_check = self._back_calculate_emi(offer_amt, tenor)
            if emi_check > max_emi * 1.001:   # 0.1% tolerance for float rounding
                reason_codes.append(f"TENOR_{tenor}M_EMI_INFEASIBLE")
                continue

            score = offer_amt * self.tenor_preference_weights.get(tenor, 1.0)
            candidates.append({
                "tenor": tenor,
                "offer_amount": offer_amt,
                "max_offerable": max_loan,
                "emi_check": emi_check,
                "score": score,
            })

        if not candidates:
            reason_codes.append("NO_FEASIBLE_CANDIDATE")
            return self._refer_record(policy, card_line, persona, reason_codes)

        # ── Rank and select ───────────────────────────────────────────────────
        # Primary: highest score; tie-break: shorter tenor
        candidates.sort(key=lambda c: (-c["score"], c["tenor"]))
        best = candidates[0]
        reason_codes.append(f"BEST_CANDIDATE_TENOR_{best['tenor']}M")

        # ── Determine action code ─────────────────────────────────────────────
        action = policy["base_action"]
        if action == ActionCode.STP_APPROVE and max_emi < self.stp_min_emi_thb:
            action = ActionCode.MANUAL_REVIEW
            reason_codes.append("BELOW_STP_EMI_FLOOR")

        stp_flag = (action == ActionCode.STP_APPROVE)
        verification = policy["verification"]

        # ── Card line ─────────────────────────────────────────────────────────
        adj_card_line = round(card_line * conf_mult, 0)
        include_card  = (
            persona in _CARD_LINE_ELIGIBLE_PERSONAS
            and adj_card_line >= self.min_card_line_thb
        )
        if include_card:
            reason_codes.append("CARD_LINE_INCLUDED")

        return {
            "action_code":              action.value,
            "verification_intensity":   verification.value,
            "offer_amount_recommended": best["offer_amount"],
            "recommended_tenor_months": best["tenor"],
            "max_offerable_amount":     best["max_offerable"],
            "include_card_line":        include_card,
            "card_line_recommended":    adj_card_line if include_card else 0.0,
            "stp_flag":                 stp_flag,
            "optimization_reason_codes": reason_codes,
        }

    def _refer_record(self, policy: dict, card_line: float, persona: str, reason_codes: List[str]) -> dict:
        return {
            "action_code":              ActionCode.REFER.value,
            "verification_intensity":   policy["verification"].value,
            "offer_amount_recommended": 0.0,
            "recommended_tenor_months": 0,
            "max_offerable_amount":     0.0,
            "include_card_line":        False,
            "card_line_recommended":    0.0,
            "stp_flag":                 False,
            "optimization_reason_codes": reason_codes,
        }

    def _back_calculate_emi(self, loan_amount: float, tenor_months: int) -> float:
        """Compute implied monthly EMI for a given loan amount and tenor."""
        r = self.reference_rate_annual / 12.0
        n = tenor_months
        if r < 1e-9:
            return loan_amount / n
        return loan_amount * r * (1 + r) ** n / ((1 + r) ** n - 1)

    @staticmethod
    def _safe_col(df: pd.DataFrame, col: str, default: float) -> pd.Series:
        if col in df.columns:
            return df[col].fillna(default).astype(float)
        logger.warning(f"OfferSelector: '{col}' not found — using {default}")
        return pd.Series(default, index=df.index, dtype=float)

    @staticmethod
    def _safe_str_col(df: pd.DataFrame, col: str, default: str) -> pd.Series:
        if col in df.columns:
            return df[col].fillna(default).astype(str)
        logger.warning(f"OfferSelector: '{col}' not found — using '{default}'")
        return pd.Series(default, index=df.index, dtype=str)

    @classmethod
    def from_config(cls, cfg: dict) -> "OfferSelector":
        """Construct from 'offer_optimization' section of config.yaml."""
        oc = cfg.get("offer_optimization", {})
        tpw_raw = oc.get("tenor_preference_weights", {})
        tpw = {int(k): float(v) for k, v in tpw_raw.items()} if tpw_raw else None
        return cls(
            product_tenors_months    = oc.get("product_tenors_months",  [12, 24, 36, 48, 60]),
            tenor_preference_weights = tpw,
            min_loan_amount_thb      = oc.get("min_loan_amount_thb",    10_000.0),
            stp_min_emi_thb          = oc.get("stp_min_emi_thb",        1_000.0),
            min_card_line_thb        = oc.get("min_card_line_thb",      5_000.0),
            reference_rate_annual    = oc.get("reference_rate_annual",  0.18),
        )
