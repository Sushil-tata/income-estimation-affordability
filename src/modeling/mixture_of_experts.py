"""
Mixture-of-Experts (MoE) Income Blender
─────────────────────────────────────────
Blends the income estimates from the top-2 persona models when the
SegmentationPipeline signals low confidence (use_moe_blend == True).

When to blend
─────────────
  PersonaRouter assigns each non-PAYROLL, non-THIN customer a primary persona
  (L0 / L1 / L2) and an associated probability vector [P(L0), P(L1), P(L2)].

  If max(P) < moe_confidence_threshold (default 0.70):
    → The customer sits near a cluster boundary.
    → A single persona model may be over-confident.
    → Blend the top-2 persona model predictions proportionally.

  If max(P) ≥ 0.70:
    → Use the primary persona model directly (no blending overhead).

Blending formula
────────────────
  Let p1, p2 = top-2 persona probabilities (renormalised to sum to 1).
  income_blended = p1 * income(persona_1) + p2 * income(persona_2)

  This is a soft, probability-weighted average — not a hard switch.

PAYROLL and THIN customers are never blended:
  PAYROLL → income comes from payroll_income column directly.
  THIN    → policy floor applied by AffordabilityEngine.

Usage
─────
  blender = MixtureOfExperts(fitted_trainer)
  income  = blender.predict(features_df, segmentation_result_df)

  Returns a pd.Series of income estimates (THB/month) with the same
  index as features_df.  The series also carries a metadata DataFrame
  accessible via blender.last_metadata_.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# Persona label order (must match SegmentationPipeline / PersonaRouter)
_ML_PERSONAS: List[str] = ["L0", "L1", "L2"]
_PROB_COLS: List[str] = ["L0_prob", "L1_prob", "L2_prob"]


class MixtureOfExperts:
    """
    Blends top-2 persona model predictions for boundary customers.

    Parameters
    ----------
    trainer : SegmentModelTrainer
        Fitted SegmentModelTrainer with per-persona models.
    moe_confidence_threshold : float
        Customers with max(L0/L1/L2_prob) below this get blended.
        Should match the value used in SegmentationPipeline (default 0.70).
    persona_col : str
        Column in segmentation_result holding the assigned persona.
        Default "persona".
    """

    def __init__(
        self,
        trainer,
        moe_confidence_threshold: float = 0.70,
        persona_col: str = "persona",
    ):
        self.trainer = trainer
        self.moe_confidence_threshold = moe_confidence_threshold
        self.persona_col = persona_col
        self.last_metadata_: Optional[pd.DataFrame] = None

    def predict(
        self,
        features: pd.DataFrame,
        segmentation_result: pd.DataFrame,
    ) -> pd.Series:
        """
        Return income estimates, blending where persona confidence is low.

        Parameters
        ----------
        features : pd.DataFrame
            Customer feature matrix (FeatureEngineer output).
        segmentation_result : pd.DataFrame
            Output of SegmentationPipeline.run() — must contain:
              persona, use_moe_blend, L0_prob, L1_prob, L2_prob,
              persona_confidence.

        Returns
        -------
        pd.Series (float) — income estimate in THB/month.
            NaN for THIN customers (policy floor applied downstream).
        """
        idx = features.index
        income = pd.Series(np.nan, index=idx, name="income_estimate")
        blend_flag = pd.Series(False, index=idx, name="moe_blended")
        top1_persona = segmentation_result[self.persona_col].reindex(idx)

        # ── Pre-compute per-persona model predictions for ALL ML customers ──
        # We compute predictions for all three personas upfront to avoid
        # repeated slicing inside the blending loop.
        persona_predictions: Dict[str, pd.Series] = {}
        for persona in _ML_PERSONAS:
            if persona in self.trainer.fitted_models_:
                X_persona = features.copy()
                X_persona[self.persona_col] = persona
                if "segment" in X_persona.columns:
                    X_persona["segment"] = persona
                preds = self.trainer.predict_with_metadata(X_persona,
                                                           segment_col=self.persona_col)
                persona_predictions[persona] = preds["income_estimate"]
            else:
                persona_predictions[persona] = pd.Series(np.nan, index=idx)

        # ── Single-persona customers (high confidence or PAYROLL) ────────────
        no_blend_mask = ~segmentation_result["use_moe_blend"].reindex(idx).fillna(False)

        # PAYROLL: trainer may or may not have a PAYROLL model; use it if available
        payroll_mask = top1_persona == "PAYROLL"
        if payroll_mask.any() and "PAYROLL" in self.trainer.fitted_models_:
            X_payroll = features.copy()
            X_payroll[self.persona_col] = "PAYROLL"
            if "segment" in X_payroll.columns:
                X_payroll["segment"] = "PAYROLL"
            pr = self.trainer.predict_with_metadata(X_payroll, segment_col=self.persona_col)
            income[payroll_mask] = pr.loc[payroll_mask, "income_estimate"].values

        # ML personas, no blend needed
        for persona in _ML_PERSONAS:
            mask = (top1_persona == persona) & no_blend_mask
            if mask.any() and persona in persona_predictions:
                income[mask] = persona_predictions[persona][mask]

        # ── MoE blend customers ──────────────────────────────────────────────
        blend_mask = segmentation_result["use_moe_blend"].reindex(idx).fillna(False)
        n_blend = blend_mask.sum()

        if n_blend > 0:
            logger.info(f"MoE: blending {n_blend} boundary customers")

            prob_df = segmentation_result[_PROB_COLS].reindex(idx)

            for row_idx in blend_mask[blend_mask].index:
                probs = prob_df.loc[row_idx, _PROB_COLS].values.astype(float)

                # Top-2 personas by probability
                ranked = np.argsort(probs)[::-1]  # descending
                top1_idx, top2_idx = ranked[0], ranked[1]
                p1, p2 = probs[top1_idx], probs[top2_idx]
                total = p1 + p2
                if total == 0:
                    total = 1.0

                persona1 = _ML_PERSONAS[top1_idx]
                persona2 = _ML_PERSONAS[top2_idx]

                est1 = persona_predictions[persona1].get(row_idx, np.nan)
                est2 = persona_predictions[persona2].get(row_idx, np.nan)

                # Fall back to single-model if one estimate is missing
                if np.isnan(est1) and np.isnan(est2):
                    blended = np.nan
                elif np.isnan(est1):
                    blended = est2
                elif np.isnan(est2):
                    blended = est1
                else:
                    blended = (p1 / total) * est1 + (p2 / total) * est2

                income[row_idx] = blended
                blend_flag[row_idx] = True

        # ── Metadata ─────────────────────────────────────────────────────────
        self.last_metadata_ = pd.DataFrame({
            "persona":        top1_persona,
            "income_estimate": income,
            "moe_blended":    blend_flag,
            "persona_confidence": segmentation_result["persona_confidence"].reindex(idx),
        })

        n_ml    = (~top1_persona.isin(["THIN"]) & ~blend_flag & income.notna()).sum()
        n_thin  = (top1_persona == "THIN").sum()
        logger.info(
            f"MoE.predict(): ML_single={n_ml}  MoE_blended={blend_flag.sum()}  "
            f"THIN(policy)={n_thin}"
        )
        return income
