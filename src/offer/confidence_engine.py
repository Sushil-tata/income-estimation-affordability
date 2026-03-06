"""
Layer 9B — Confidence Engine
─────────────────────────────
Compute a monotone confidence_multiplier that scales offer amounts
based on income estimation reliability.

Design constraints
──────────────────
  • BCI is NOT applied here — it was already applied in Layer G (dscr_used).
    Re-applying BCI would double-penalise and is prohibited.
  • THIN / SPARSE personas always get multiplier = 0.0.
  • All other multipliers are in [min_multiplier, 1.0].
  • Monotonicity guarantee: higher uncertainty → lower multiplier.
    Proven by construction (see _reliability_score docstring).
  • All parameters are [PROVISIONAL] pending 6-month vintage calibration.

Inputs (read from final_output + capacity output)
──────────────────────────────────────────────────
  persona            : str   (PAYROLL, L0, L1, L2, PT, THIN, SPARSE)
  p_reliable10       : float [0,1]  P(income within ±10%) — optional
  p_over10           : float [0,1]  P(income over-estimated by >10%) — optional
  model_confidence   : float [0,1]  fallback if p_reliable10/p_over10 missing
  income_interval_width : float     fallback signal for interval width

Outputs
───────
  confidence_multiplier   : float ∈ [0, 1.0]  — always ≥ 0
  reliability_score       : float ∈ [0, 1.0]  — pre-multiplier diagnostic
  confidence_reason_code  : str   — human-readable audit label
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Default persona base multipliers [PROVISIONAL] ────────────────────────────
# Reflect a-priori estimation reliability by persona.
# PAYROLL: payslip-backed, highest confidence.
# L0/L1/L2: progressively less formal income evidence.
# PT: part-time / gig — volatile income.
# THIN: policy floor used — no reliable estimate, offer = 0.
# SPARSE: data-sufficiency failed — offer = 0.
_DEFAULT_PERSONA_BASE: Dict[str, float] = {
    "PAYROLL": 1.00,   # [PROVISIONAL]
    "L0":      0.95,   # [PROVISIONAL]
    "L1":      0.85,   # [PROVISIONAL]
    "L2":      0.70,   # [PROVISIONAL]
    "PT":      0.65,   # [PROVISIONAL]
    "THIN":    0.00,   # Hard floor — policy decision, not calibration
    "SPARSE":  0.00,   # Hard floor — data sufficiency failed
}

_DEFAULT_UNKNOWN_BASE = 0.60   # [PROVISIONAL] — fallback for unseen persona codes


class ConfidenceEngine:
    """
    Layer 9B: Monotone confidence multiplier for offer sizing.

    Parameters
    ----------
    persona_base_multipliers : dict[str, float], optional
        Per-persona base multipliers. Defaults to _DEFAULT_PERSONA_BASE.
        Must have THIN=0.0 and SPARSE=0.0 (enforced in __init__).
    reliability_weight_p_reliable : float
        Weight on p_reliable10 in reliability score.  [PROVISIONAL]
    reliability_weight_p_over : float
        Weight on (1 - p_over10) in reliability score.  [PROVISIONAL]
    reliability_alpha : float
        Exponent applied to reliability_score before multiplying by persona_base.
        > 1 → concave (more penalisation for low reliability).  [PROVISIONAL]
    min_multiplier : float
        Hard floor for non-THIN/non-SPARSE personas.  [PROVISIONAL]
    fallback_confidence_proxy_weight : float
        Weight on model_confidence when p_reliable10 is absent.  [PROVISIONAL]
    interval_width_penalty_scale : float
        Scales how much a wide income interval reduces reliability.  [PROVISIONAL]
    """

    def __init__(
        self,
        persona_base_multipliers: Optional[Dict[str, float]] = None,
        reliability_weight_p_reliable: float = 0.60,   # [PROVISIONAL]
        reliability_weight_p_over: float = 0.40,        # [PROVISIONAL]
        reliability_alpha: float = 1.0,                 # [PROVISIONAL]
        min_multiplier: float = 0.30,                   # [PROVISIONAL]
        fallback_confidence_proxy_weight: float = 0.70, # [PROVISIONAL]
        interval_width_penalty_scale: float = 0.50,     # [PROVISIONAL]
    ):
        bases = dict(_DEFAULT_PERSONA_BASE)
        if persona_base_multipliers:
            bases.update(persona_base_multipliers)
        # Safety guard — THIN and SPARSE must never offer
        bases["THIN"]   = 0.0
        bases["SPARSE"] = 0.0
        self.persona_base_multipliers          = bases
        self.reliability_weight_p_reliable     = reliability_weight_p_reliable
        self.reliability_weight_p_over         = reliability_weight_p_over
        self.reliability_alpha                 = reliability_alpha
        self.min_multiplier                    = min_multiplier
        self.fallback_confidence_proxy_weight  = fallback_confidence_proxy_weight
        self.interval_width_penalty_scale      = interval_width_penalty_scale

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute(self, final_output: pd.DataFrame) -> pd.DataFrame:
        """
        Compute confidence multiplier per customer.

        Parameters
        ----------
        final_output : pd.DataFrame
            Must contain 'persona'.  Optional: p_reliable10, p_over10,
            model_confidence, income_interval_width, income_estimate_raw.

        Returns
        -------
        pd.DataFrame
            Index matches final_output. Three new columns:
              reliability_score       : float [0,1]
              confidence_multiplier   : float [0,1]
              confidence_reason_code  : str
        """
        out = pd.DataFrame(index=final_output.index)

        persona   = self._safe_str_col(final_output, "persona", "UNKNOWN")
        p_rel     = self._safe_col(final_output, "p_reliable10",         np.nan)
        p_over    = self._safe_col(final_output, "p_over10",             np.nan)
        conf      = self._safe_col(final_output, "model_confidence",      np.nan)
        width     = self._safe_col(final_output, "income_interval_width", np.nan)
        income_raw= self._safe_col(final_output, "income_estimate_raw",   np.nan)

        # ── Reliability score ─────────────────────────────────────────────────
        rel_score, reason_code = self._reliability_score(
            persona, p_rel, p_over, conf, width, income_raw
        )
        out["reliability_score"] = rel_score.clip(0.0, 1.0).round(4)

        # ── Persona base multiplier ───────────────────────────────────────────
        persona_base = persona.map(self.persona_base_multipliers).fillna(_DEFAULT_UNKNOWN_BASE)

        # ── Final multiplier  ─────────────────────────────────────────────────
        # multiplier = persona_base × reliability_score^alpha
        # THIN/SPARSE → persona_base=0.0 → multiplier=0.0 (hard floor enforced)
        raw_mult = persona_base * (rel_score.clip(0.0, 1.0) ** self.reliability_alpha)

        # For THIN/SPARSE keep at 0; for others clip to [min_multiplier, persona_base]
        is_zero_persona = persona.isin(["THIN", "SPARSE"])
        final_mult = np.where(
            is_zero_persona,
            0.0,
            raw_mult.clip(lower=self.min_multiplier, upper=persona_base),
        )

        out["confidence_multiplier"]  = pd.Series(final_mult, index=final_output.index).round(4)
        out["confidence_reason_code"] = reason_code

        logger.info(
            f"ConfidenceEngine: {len(out):,} customers | "
            f"zero_multiplier={int((out['confidence_multiplier'] == 0).sum()):,} | "
            f"mean_multiplier={out['confidence_multiplier'].mean():.3f}"
        )
        return out

    # ── Internal ───────────────────────────────────────────────────────────────

    def _reliability_score(
        self,
        persona: pd.Series,
        p_rel:   pd.Series,
        p_over:  pd.Series,
        conf:    pd.Series,
        width:   pd.Series,
        income_raw: pd.Series,
    ):
        """
        Monotone reliability score ∈ [0, 1].

        Priority:
          1. If p_reliable10 and p_over10 are both present → use primary formula.
          2. Else fallback to proxy from model_confidence and interval width.
          3. For THIN/SPARSE → score = 0.0 (persona_base handles the floor,
             but we set score=0 here for auditability).

        Primary formula (monotone by construction):
          w1 = reliability_weight_p_reliable
          w2 = reliability_weight_p_over
          score = (w1 × p_reliable10 + w2 × (1 − p_over10)) / (w1 + w2)

          Monotonicity proof:
            ∂score/∂p_reliable10  =  w1/(w1+w2)  > 0  ✓ (higher reliability → higher score)
            ∂score/∂p_over10      = −w2/(w1+w2)  < 0  ✓ (higher overestimation → lower score)

        Fallback formula:
          normalised_width = clip(width / max(income_raw, 1), 0, 1)
          score = w_conf × model_confidence + (1 − w_conf) × (1 − interval_width_penalty)
          where interval_width_penalty = interval_width_penalty_scale × normalised_width
        """
        w1, w2 = self.reliability_weight_p_reliable, self.reliability_weight_p_over

        has_primary = p_rel.notna() & p_over.notna()

        # Primary path
        primary_score = (w1 * p_rel.fillna(0.5) + w2 * (1.0 - p_over.fillna(0.5))) / (w1 + w2)

        # Fallback path
        conf_filled = conf.fillna(0.5)
        norm_width  = (width / income_raw.replace(0, np.nan)).fillna(0.5).clip(0, 1)
        width_penalty = self.interval_width_penalty_scale * norm_width
        fallback_score = (
            self.fallback_confidence_proxy_weight * conf_filled
            + (1.0 - self.fallback_confidence_proxy_weight) * (1.0 - width_penalty)
        )

        score = pd.Series(
            np.where(has_primary, primary_score, fallback_score),
            index=persona.index,
        )

        # THIN / SPARSE always 0
        score = score.where(~persona.isin(["THIN", "SPARSE"]), other=0.0)

        # Reason code
        reason = pd.Series("PRIMARY_RELIABILITY", index=persona.index)
        reason = reason.where(has_primary, other="FALLBACK_PROXY")
        reason = reason.where(~persona.isin(["THIN", "SPARSE"]), other="ZERO_PERSONA_BASE")

        return score.clip(0.0, 1.0), reason

    @staticmethod
    def _safe_col(df: pd.DataFrame, col: str, default) -> pd.Series:
        if col in df.columns:
            return df[col].astype(float)
        logger.debug(f"ConfidenceEngine: '{col}' not found — using default {default}")
        return pd.Series(default, index=df.index, dtype=float)

    @staticmethod
    def _safe_str_col(df: pd.DataFrame, col: str, default: str) -> pd.Series:
        if col in df.columns:
            return df[col].fillna(default).astype(str)
        logger.warning(f"ConfidenceEngine: '{col}' not found — using default '{default}'")
        return pd.Series(default, index=df.index, dtype=str)

    @classmethod
    def from_config(cls, cfg: dict) -> "ConfidenceEngine":
        """Construct from 'offer_optimization' section of config.yaml."""
        oc = cfg.get("offer_optimization", {})
        pb = oc.get("persona_base_multipliers", {})
        return cls(
            persona_base_multipliers          = pb if pb else None,
            reliability_weight_p_reliable     = oc.get("reliability_weight_p_reliable", 0.60),
            reliability_weight_p_over         = oc.get("reliability_weight_p_over",     0.40),
            reliability_alpha                 = oc.get("reliability_alpha",             1.0),
            min_multiplier                    = oc.get("min_multiplier",               0.30),
            fallback_confidence_proxy_weight  = oc.get("fallback_confidence_proxy_weight", 0.70),
            interval_width_penalty_scale      = oc.get("interval_width_penalty_scale", 0.50),
        )
