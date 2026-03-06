"""
Layer 9 — Action Optimizer
───────────────────────────
Orchestrator that chains:
  9A  CapacityEngine       → max_emi, max_loan_{n}m, max_card_line
  9B  ConfidenceEngine     → confidence_multiplier
  9C  OfferSelector        → action_code, offer_amount_recommended, …

The orchestrator:
  • Reads frozen v5.0 final_output.
  • NEVER writes back to any upstream field.
  • Assembles ActionOptimizationResult containing:
      - capacity   : DataFrame (9A outputs)
      - confidence : DataFrame (9B outputs)
      - offers     : DataFrame (9C outputs)
      - combined   : DataFrame (all columns merged, index = customer_id)

Governance
──────────
  • All provisional parameters are tagged [PROVISIONAL] in component docstrings.
  • confidence_multiplier is an offer-sizing parameter — it does NOT modify
    PolicyIncome or dscr_used. This distinction is logged per run.
  • THIN / SPARSE always result in REFER and zero offer (enforced in 9B + 9C).
  • Full audit trail is preserved in optimization_reason_codes per customer.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from offer.capacity_engine import CapacityEngine
from offer.confidence_engine import ConfidenceEngine
from offer.offer_selector import OfferSelector, ActionCode, VerificationIntensity

logger = logging.getLogger(__name__)


@dataclass
class ActionOptimizationResult:
    """
    Container for Layer 9 outputs.

    Attributes
    ----------
    capacity   : DataFrame  — 9A outputs (max_emi, max_loan_*m, max_card_line, …)
    confidence : DataFrame  — 9B outputs (reliability_score, confidence_multiplier, …)
    offers     : DataFrame  — 9C outputs (action_code, offer_amount_recommended, …)
    combined   : DataFrame  — all three merged on customer_id index
    n_customers : int
    n_stp       : int
    n_manual    : int
    n_refer     : int
    """
    capacity:    pd.DataFrame = field(default_factory=pd.DataFrame)
    confidence:  pd.DataFrame = field(default_factory=pd.DataFrame)
    offers:      pd.DataFrame = field(default_factory=pd.DataFrame)
    combined:    pd.DataFrame = field(default_factory=pd.DataFrame)
    n_customers: int = 0
    n_stp:       int = 0
    n_manual:    int = 0
    n_refer:     int = 0


class ActionOptimizer:
    """
    Layer 9 orchestrator: 9A → 9B → 9C.

    Parameters
    ----------
    capacity_engine    : CapacityEngine
    confidence_engine  : ConfidenceEngine
    offer_selector     : OfferSelector
    """

    def __init__(
        self,
        capacity_engine:   Optional[CapacityEngine]   = None,
        confidence_engine: Optional[ConfidenceEngine] = None,
        offer_selector:    Optional[OfferSelector]    = None,
    ):
        self.capacity_engine   = capacity_engine   or CapacityEngine()
        self.confidence_engine = confidence_engine or ConfidenceEngine()
        self.offer_selector    = offer_selector    or OfferSelector()

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, final_output: pd.DataFrame) -> ActionOptimizationResult:
        """
        Execute full Layer 9 pipeline.

        Parameters
        ----------
        final_output : pd.DataFrame
            Frozen output from InferencePipeline.run().
            Required upstream columns:
              adjusted_income, adsc, dscr_used, existing_obligations, persona.
            Optional (used by 9B):
              p_reliable10, p_over10, model_confidence,
              income_interval_width, income_estimate_raw.

        Returns
        -------
        ActionOptimizationResult
        """
        logger.info(
            "ActionOptimizer.run(): starting Layer 9 — "
            "NOTE: confidence_multiplier is an offer-sizing parameter; "
            "PolicyIncome and dscr_used are NOT modified."
        )

        # ── 9A: Capacity ───────────────────────────────────────────────────────
        capacity = self.capacity_engine.compute(final_output)

        # ── 9B: Confidence ────────────────────────────────────────────────────
        confidence = self.confidence_engine.compute(final_output)

        # ── 9C: Offer selection ───────────────────────────────────────────────
        offers = self.offer_selector.select(capacity, confidence, final_output)

        # ── Merge ─────────────────────────────────────────────────────────────
        combined = pd.concat([capacity, confidence, offers], axis=1)

        # Summary stats
        n_stp    = int((offers["action_code"] == ActionCode.STP_APPROVE.value).sum())
        n_manual = int((offers["action_code"] == ActionCode.MANUAL_REVIEW.value).sum())
        n_refer  = int((offers["action_code"] == ActionCode.REFER.value).sum())

        logger.info(
            f"ActionOptimizer: complete | n={len(combined):,} | "
            f"STP={n_stp:,} | MANUAL={n_manual:,} | REFER={n_refer:,}"
        )

        return ActionOptimizationResult(
            capacity    = capacity,
            confidence  = confidence,
            offers      = offers,
            combined    = combined,
            n_customers = len(combined),
            n_stp       = n_stp,
            n_manual    = n_manual,
            n_refer     = n_refer,
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "ActionOptimizer":
        """Construct all sub-engines from config dict."""
        return cls(
            capacity_engine   = CapacityEngine.from_config(cfg),
            confidence_engine = ConfidenceEngine.from_config(cfg),
            offer_selector    = OfferSelector.from_config(cfg),
        )
