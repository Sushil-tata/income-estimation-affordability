"""
Layer 9 — Offer / Action Optimization
───────────────────────────────────────
Additive layer that converts the frozen v5.0 income engine output into
the optimal executable lending action.

Components
──────────
  9A  CapacityEngine         — deterministic: max EMI / loan / card from adsc
  9B  ConfidenceEngine       — monotone multiplier from reliability signals
  9C  OfferSelector          — candidate generation + ranked selection
  ActionOptimizer            — orchestrator: chains 9A → 9B → 9C

Governance
──────────
  • Layers A–G (v5.0) are frozen and unchanged.
  • Layer 9 only READS from upstream outputs — it never modifies them.
  • policy_income, dscr_used, adsc, bci_score, decision matrix are untouched.
  • All thresholds are calibration parameters marked [PROVISIONAL].
  • confidence_multiplier is an offer-sizing parameter, NOT an income haircut.
    PolicyIncome is never modified. The distinction must appear in audit logs.
"""

from offer.capacity_engine import CapacityEngine
from offer.confidence_engine import ConfidenceEngine
from offer.offer_selector import OfferSelector, ActionCode, VerificationIntensity
from offer.action_optimizer import ActionOptimizer, ActionOptimizationResult

__all__ = [
    "CapacityEngine",
    "ConfidenceEngine",
    "OfferSelector",
    "ActionOptimizer",
    "ActionOptimizationResult",
    "ActionCode",
    "VerificationIntensity",
]
