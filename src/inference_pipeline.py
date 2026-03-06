"""
End-to-End Inference Pipeline
──────────────────────────────
Single entry point that chains all framework components from raw monthly
aggregate data through to a final credit decision.

Component chain
───────────────
  FeatureEngineer                     → behavioral features per customer
      ↓                                 INPUT: up to 6 months of monthly aggregates (2–6M),
                                        eligibility gated by OQS
  SegmentationPipeline                → persona assignment (PAYROLL/L0/L1/L2/THIN/PT)
      ↓
  PersonaStabilitySmoother            → suppress persona churn across runs
      ↓
  SegmentModelTrainer + MixtureOfExperts → income estimate
      ↓                                   ML_ESTIMATE (L0/L1/L2) | POLICY_FLOOR (THIN)
                                          PT_FORMULA (PT: credit × retention_robust)
  BCIScorer                           → BCI score + band (drives DSCR cap, NOT income haircut)
      ↓                                 AdjustedIncome = PolicyIncome (no haircut applied)
  AffordabilityEngine                 → DSCR = min(persona_cap, bci_band_cap)
      ↓                                 ADSC = AdjustedIncome × DSCR − obligations
  PolicyEngine                        → final decision (STP_APPROVE / STP_DECLINE / ...)

Run-state (for sequential scoring)
───────────────────────────────────
  The pipeline returns a `run_state` dict after each run.
  Pass it back as `prev_run_state` on the next monthly run so the
  PersonaStabilitySmoother can suppress churn.

  run_state = {
      "personas":        pd.Series,   stable persona labels
      "smoothed_probs":  pd.DataFrame, L0/L1/L2 smoothed probs
      "run_date":        str,
  }

Usage
─────
  # Training
  pipeline = InferencePipeline(config_path="config/config.yaml")
  pipeline.fit(monthly_agg_train, y_verified_income)
  pipeline.save("artifacts/models/")

  # Scoring (first run)
  result = pipeline.run(monthly_agg_score)
  state  = result.run_state

  # Scoring (subsequent monthly run — pass previous state)
  result = pipeline.run(monthly_agg_next_month, prev_run_state=state)

  # Access outputs
  result.final_output           # consolidated customer-level DataFrame
  result.policy_result          # final_decision, decision_reason
  result.decision_summary()     # approval rates by decision type
"""

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import yaml

from income_estimation.features import FeatureEngineer
from segmentation.pipeline import SegmentationPipeline
from modeling.label_engineering import LabelEngineer
from modeling.segment_trainer import SegmentModelTrainer
from modeling.mixture_of_experts import MixtureOfExperts
from modeling.persona_stability import PersonaStabilitySmoother
from bci.scorer import BCIScorer
from affordability.engine import AffordabilityEngine
from affordability.policy import PolicyEngine

logger = logging.getLogger(__name__)

_PROB_COLS = ["L0_prob", "L1_prob", "L2_prob"]


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class InferencePipelineResult:
    """
    Structured result returned by InferencePipeline.run().

    Attributes
    ----------
    final_output       : Consolidated customer-level DataFrame (the main output).
    features           : Feature matrix (FeatureEngineer output).
    segmentation_result: Raw segmentation pipeline output (personas + indices).
    income_result      : Income estimates with income_estimate_type column.
    bci_result         : BCIScorer output.
    affordability_result: AffordabilityEngine output.
    policy_result      : PolicyEngine.decide() output.
    run_state          : State dict to pass to the next monthly run.
    """
    final_output:           pd.DataFrame
    features:               pd.DataFrame
    segmentation_result:    pd.DataFrame
    income_result:          pd.DataFrame
    bci_result:             pd.DataFrame
    affordability_result:   pd.DataFrame
    policy_result:          pd.DataFrame
    run_state:              Dict[str, Any] = field(default_factory=dict)

    def decision_summary(self) -> pd.DataFrame:
        """Approval rates and average metrics per decision type."""
        n = len(self.policy_result)
        summary = self.policy_result["final_decision"].value_counts().reset_index()
        summary.columns = ["decision", "count"]
        summary["pct"] = (summary["count"] / n * 100).round(2)

        avg = self.final_output.groupby("final_decision").agg(
            avg_bci=("bci_score", "mean"),
            avg_income=("income_estimate_raw", "mean"),
            avg_adsc=("adsc", "mean"),
        ).reset_index().rename(columns={"final_decision": "decision"})

        return summary.merge(avg, on="decision", how="left").sort_values(
            "count", ascending=False
        ).reset_index(drop=True)

    def persona_summary(self) -> pd.DataFrame:
        """Persona distribution with average BCI and income."""
        seg = self.segmentation_result["persona"].value_counts().reset_index()
        seg.columns = ["persona", "count"]
        seg["pct"] = (seg["count"] / len(seg.index.union(self.features.index)) * 100).round(2)
        return seg.sort_values("count", ascending=False).reset_index(drop=True)


# ── Main pipeline ─────────────────────────────────────────────────────────────

class InferencePipeline:
    """
    End-to-end income estimation and affordability decision pipeline.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.
    cv_folds : int
        Cross-validation folds for SegmentModelTrainer. Default 5.
    include_tabpfn : bool
        Include TabPFN v2 in model search (requires tabpfn). Default True.
    include_lstm : bool
        Include LSTM models (requires torch). Default True.
    use_persona_defaults : bool
        Narrow model search space per persona (Phase 4). Default True.
    smoothing_alpha : float
        EMA weight for PersonaStabilitySmoother. Default 0.50.
    switch_delta : float
        Min probability margin to accept a persona switch. Default 0.15.
    moe_confidence_threshold : float
        Below this persona confidence, blend top-2 models. Default 0.70.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        cv_folds: int = 5,
        include_tabpfn: bool = True,
        include_lstm: bool = True,
        use_persona_defaults: bool = True,
        smoothing_alpha: float = 0.50,
        switch_delta: float = 0.15,
        moe_confidence_threshold: float = 0.70,
    ):
        self.config_path = config_path
        self.cv_folds = cv_folds
        self.include_tabpfn = include_tabpfn
        self.include_lstm = include_lstm
        self.use_persona_defaults = use_persona_defaults
        self.smoothing_alpha = smoothing_alpha
        self.switch_delta = switch_delta
        self.moe_confidence_threshold = moe_confidence_threshold

        with open(config_path) as f:
            self._cfg = yaml.safe_load(f)

        # Instantiate stateless / config-driven components immediately
        self.feature_engineer     = FeatureEngineer()
        self.bci_scorer           = BCIScorer(config_path=config_path)
        self.affordability_engine = AffordabilityEngine(config_path=config_path)
        self.policy_engine        = PolicyEngine(config_path=config_path)
        self.stability_smoother   = PersonaStabilitySmoother(
            smoothing_alpha=smoothing_alpha,
            switch_delta=switch_delta,
            config_path=config_path,
        )

        # Components that require fitting
        self.segmentation_pipeline: Optional[SegmentationPipeline] = None
        self.label_engineer:        Optional[LabelEngineer]        = None
        self.segment_trainer:       Optional[SegmentModelTrainer]  = None
        self.moe_blender:           Optional[MixtureOfExperts]     = None

        self.fitted_ = False

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit(
        self,
        monthly_agg_train: pd.DataFrame,
        y_verified_income: pd.Series,
        payroll_income_col: Optional[str] = "payroll_income",
    ) -> "InferencePipeline":
        """
        Fit all trainable components on historical data.

        Parameters
        ----------
        monthly_agg_train : pd.DataFrame
            Monthly aggregate transaction data (customer × month rows).
            Must include columns expected by FeatureEngineer.
        y_verified_income : pd.Series
            Verified gross monthly income (THB) indexed by customer_id.
            Used to train SegmentModelTrainer.
        payroll_income_col : str, optional
            Column in monthly_agg_train holding the known payroll income
            for PAYROLL customers.
        """
        logger.info("InferencePipeline.fit() — start")

        # ── Step 1: Feature engineering ──────────────────────────────────────
        logger.info("  Step 1/4: FeatureEngineer.build_features()")
        features = self.feature_engineer.build_features(monthly_agg_train)
        features = features.set_index("customer_id") if "customer_id" in features.columns else features
        logger.info(f"  Features: {features.shape[0]:,} customers × {features.shape[1]} features")

        # ── Step 2: Segmentation pipeline ────────────────────────────────────
        logger.info("  Step 2/4: SegmentationPipeline.fit()")
        self.segmentation_pipeline = SegmentationPipeline(
            config_path=self.config_path,
            moe_confidence_threshold=self.moe_confidence_threshold,
        )
        self.segmentation_pipeline.fit(features)
        seg_result = self.segmentation_pipeline.run(features)
        logger.info(f"  Persona distribution (train):\n{seg_result['persona'].value_counts()}")

        # ── Step 3: Label engineer ────────────────────────────────────────────
        logger.info("  Step 3/4: LabelEngineer.fit()")
        # Align verified income with feature index
        common_idx = features.index.intersection(y_verified_income.index)
        if len(common_idx) < len(features):
            logger.warning(
                f"  y_verified_income covers {len(common_idx):,}/{len(features):,} "
                f"training customers — fitting label engineer on the intersection."
            )
        y_labeled = y_verified_income.loc[common_idx]

        # Join features into X so LabelEngineer can access feature columns
        # (e.g., median_credit_6m, retention_ratio_6m for L1 retained inflow proxy).
        X_labeled = seg_result.loc[common_idx].join(
            features.loc[common_idx], how="left", rsuffix="_feat"
        )

        self.label_engineer = LabelEngineer(
            shrinkage_alpha=self._cfg.get("label_engineering", {}).get("shrinkage_alpha", 1.0),
            winsor_pct=self._cfg.get("label_engineering", {}).get("winsor_pct", 0.025),
        )
        self.label_engineer.fit(X_labeled, y_labeled, segment_col="persona")

        # ── Step 4: Segment model trainer ─────────────────────────────────────
        logger.info("  Step 4/4: SegmentModelTrainer.fit()")
        smt_cfg = self._cfg.get("advanced_modeling", {}).get("segment_trainer", {})
        self.segment_trainer = SegmentModelTrainer(
            label_engineer=self.label_engineer,
            cv_folds=self.cv_folds,
            use_persona_defaults=self.use_persona_defaults,
            include_tabpfn=self.include_tabpfn,
            include_lstm=self.include_lstm,
            max_rows_per_segment=smt_cfg.get("max_rows_per_segment", 20_000),
            random_state=smt_cfg.get("random_state", 42),
        )
        self.segment_trainer.fit(
            seg_result.join(features, how="left", rsuffix="_feat"),
            y_labeled,
        )

        # MoE blender wraps the trained segment trainer
        self.moe_blender = MixtureOfExperts(
            self.segment_trainer,
            moe_confidence_threshold=self.moe_confidence_threshold,
        )

        self.fitted_ = True
        logger.info(
            f"InferencePipeline.fit() complete — "
            f"ml_personas={sorted(self.segment_trainer.ml_personas_)}  "
            f"policy_floor={sorted(self.segment_trainer.policy_floor_personas_)}"
        )
        return self

    # ── Scoring ───────────────────────────────────────────────────────────────

    def run(
        self,
        monthly_agg: pd.DataFrame,
        prev_run_state: Optional[Dict[str, Any]] = None,
        payroll_income_col: Optional[str] = "payroll_income",
    ) -> InferencePipelineResult:
        """
        Score a customer population end-to-end.

        Parameters
        ----------
        monthly_agg : pd.DataFrame
            Monthly aggregate transaction data for scoring population.
        prev_run_state : dict, optional
            run_state from the previous InferencePipelineResult.
            Enables persona stability smoothing across monthly runs.
        payroll_income_col : str, optional
            Column with known payroll income for PAYROLL customers.

        Returns
        -------
        InferencePipelineResult
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before run().")

        logger.info("InferencePipeline.run() — start")
        run_ts = datetime.utcnow().isoformat()

        # ── 1. Feature engineering ────────────────────────────────────────────
        features = self.feature_engineer.build_features(monthly_agg)
        if "customer_id" in features.columns:
            features = features.set_index("customer_id")
        logger.info(f"  Features: {features.shape[0]:,} × {features.shape[1]}")

        # ── 2. Segmentation ───────────────────────────────────────────────────
        seg_result = self.segmentation_pipeline.run(features)

        # ── 3. Persona stability smoothing ────────────────────────────────────
        # Only smooth customers with L0/L1/L2 — PAYROLL/THIN bypass.
        ml_mask = seg_result["persona"].isin(["L0", "L1", "L2"])

        if ml_mask.any() and seg_result[_PROB_COLS].notna().any(axis=None):
            prev_personas    = prev_run_state["personas"].reindex(seg_result.index) \
                               if prev_run_state else None
            prev_smoothed    = prev_run_state["smoothed_probs"].reindex(seg_result.index) \
                               if prev_run_state else None

            smooth_result = self.stability_smoother.smooth(
                new_probs=seg_result.loc[ml_mask, _PROB_COLS],
                current_personas=prev_personas[ml_mask] if prev_personas is not None else None,
                prev_smoothed_probs=prev_smoothed.loc[ml_mask] if prev_smoothed is not None else None,
            )
            # Apply stable personas back
            seg_result.loc[ml_mask, "persona"] = smooth_result["persona"].values
            seg_result.loc[ml_mask, _PROB_COLS] = smooth_result[_PROB_COLS].values

            n_switched = smooth_result["persona_switched"].sum()
            logger.info(f"  Stability smoother: {n_switched} persona switches "
                        f"({n_switched / ml_mask.sum():.1%} of ML customers)")

        # ── 4. Income estimation ──────────────────────────────────────────────
        # Build the full feature + segmentation DataFrame the trainer expects.
        scorer_df = seg_result.join(features, how="left", rsuffix="_feat")

        income_meta = self.moe_blender.predict(features, seg_result)
        income_meta = income_meta.rename("income_estimate")

        # Assemble income_result DataFrame for BCI + PolicyEngine
        income_result = self._build_income_result(
            income_estimate=income_meta,
            seg_result=seg_result,
            features=features,
            payroll_income_col=payroll_income_col,
        )

        # ── 5. BCI scoring ────────────────────────────────────────────────────
        # BCI drives decisioning and DSCR cap — income is NOT haircut here.
        # adjusted_income = income_estimate (PolicyIncome), set in BCIScorer.
        bci_df = seg_result.join(features[
            [c for c in features.columns if c not in seg_result.columns]
        ], how="left")
        bci_result = self.bci_scorer.compute(
            features=bci_df,
            income_results=income_result,
            persona_col="persona",
        )

        # ── 6. Affordability ──────────────────────────────────────────────────
        affordability_result = self.affordability_engine.compute(
            bci_results=bci_result,
            features=features,
            persona_col="segment",   # bci_result has "segment" column (=persona)
        )

        # ── 7. Policy decision ────────────────────────────────────────────────
        policy_result = self.policy_engine.decide(bci_result, affordability_result)

        # ── 8. Consolidated output ────────────────────────────────────────────
        final_output = self._build_final_output(
            features=features,
            seg_result=seg_result,
            income_result=income_result,
            bci_result=bci_result,
            affordability_result=affordability_result,
            policy_result=policy_result,
        )

        # ── 9. Build run state for next month ─────────────────────────────────
        run_state: Dict[str, Any] = {
            "personas":       seg_result["persona"].copy(),
            "smoothed_probs": seg_result[_PROB_COLS].copy(),
            "run_date":       run_ts,
        }

        logger.info(
            f"InferencePipeline.run() complete — "
            f"{len(final_output):,} customers  "
            f"decisions: {policy_result['final_decision'].value_counts().to_dict()}"
        )

        return InferencePipelineResult(
            final_output=final_output,
            features=features,
            segmentation_result=seg_result,
            income_result=income_result,
            bci_result=bci_result,
            affordability_result=affordability_result,
            policy_result=policy_result,
            run_state=run_state,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Persist all fitted components to directory."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        with open(f"{directory}/inference_pipeline.pkl", "wb") as f:
            pickle.dump(self, f)
        logger.info(f"InferencePipeline saved to {directory}/")

    @classmethod
    def load(cls, directory: str) -> "InferencePipeline":
        with open(f"{directory}/inference_pipeline.pkl", "rb") as f:
            return pickle.load(f)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_income_result(
        self,
        income_estimate: pd.Series,
        seg_result: pd.DataFrame,
        features: pd.DataFrame,
        payroll_income_col: Optional[str],
    ) -> pd.DataFrame:
        """
        Build the income_results DataFrame expected by BCIScorer.compute().

        Required columns:
          income_estimate, income_interval_width, model_confidence,
          income_source, income_q25, income_q75
        """
        ir = pd.DataFrame(index=seg_result.index)
        ir["income_estimate"] = income_estimate.reindex(seg_result.index)

        # PAYROLL: use payroll income directly when available
        payroll_mask = seg_result["persona"] == "PAYROLL"
        if payroll_mask.any() and payroll_income_col and payroll_income_col in features.columns:
            payroll_income = features.loc[payroll_mask, payroll_income_col]
            ir.loc[payroll_mask, "income_estimate"] = payroll_income.values

        # THIN: policy floor (BOT minimum)
        thin_mask = seg_result["persona"] == "THIN"
        min_income = self._cfg["bot_norms"]["minimum_income_thb"]
        ir.loc[thin_mask, "income_estimate"] = ir.loc[thin_mask, "income_estimate"].fillna(min_income)

        # PT: pass-through formula — usable_income = median_credit_6m × min(retention_6m, retention_3m)
        # Cap at avg_recurring_credit_6m when fixed_amount_similarity > 0.70 (salary-like PT).
        pt_mask = seg_result["persona"] == "PT"
        if pt_mask.any():
            pt_idx = seg_result.index[pt_mask]

            median_credit = (
                features.loc[pt_idx, "median_credit_6m"]
                if "median_credit_6m" in features.columns
                else features.loc[pt_idx, "avg_monthly_credit_6m"]
            ).fillna(0.0)

            ret_6m = features.loc[pt_idx, "retention_ratio_6m"].clip(0, 1) \
                if "retention_ratio_6m" in features.columns \
                else pd.Series(0.5, index=pt_idx)
            ret_3m = features.loc[pt_idx, "retention_ratio_3m"].clip(0, 1) \
                if "retention_ratio_3m" in features.columns \
                else ret_6m.copy()
            retention_robust = ret_6m.combine(ret_3m, min)

            pt_income = (median_credit * retention_robust).clip(lower=min_income)

            # Cap at P50 recurring credit when salary-like patterns are detected
            if "fixed_amount_similarity" in features.columns and "avg_recurring_credit_6m" in features.columns:
                salary_like = features.loc[pt_idx, "fixed_amount_similarity"] > 0.70
                if salary_like.any():
                    p50_recurring = features.loc[pt_idx, "avg_recurring_credit_6m"].fillna(pt_income)
                    pt_income = pt_income.where(~salary_like, pt_income.combine(p50_recurring, min))

            ir.loc[pt_idx, "income_estimate"] = pt_income.values

        # Derived interval width: wider for low-confidence personas
        # persona_confidence is NaN for PAYROLL/THIN/PT → set explicit defaults
        conf = seg_result["persona_confidence"].reindex(ir.index).fillna(0.5)
        conf[payroll_mask] = 1.0
        conf[thin_mask]    = 0.30    # THIN estimates have low confidence
        conf[pt_mask]      = 0.35    # PT formula has modest confidence

        ir["model_confidence"] = conf
        # Interval width ~ income × (1 - confidence) × scaling factor
        ir["income_interval_width"] = (
            ir["income_estimate"].fillna(min_income) * (1 - conf) * 0.6
        ).clip(lower=0)

        ir["income_q25"] = (ir["income_estimate"] - ir["income_interval_width"] * 0.5).clip(lower=0)
        ir["income_q75"] = ir["income_estimate"] + ir["income_interval_width"] * 0.5

        # income_source
        source = pd.Series("ESTIMATED", index=ir.index)
        source[payroll_mask] = "PAYROLL"
        source[thin_mask]    = "POLICY_FLOOR"
        source[pt_mask]      = "PT_FORMULA"
        ir["income_source"] = source

        return ir

    def _build_final_output(
        self,
        features: pd.DataFrame,
        seg_result: pd.DataFrame,
        income_result: pd.DataFrame,
        bci_result: pd.DataFrame,
        affordability_result: pd.DataFrame,
        policy_result: pd.DataFrame,
    ) -> pd.DataFrame:
        """Consolidate all pipeline outputs into a single customer-level record."""
        out = pd.DataFrame(index=features.index)

        # Persona
        out["persona"] = seg_result["persona"]
        out["persona_confidence"] = seg_result["persona_confidence"]

        # Income
        out["income_source"]      = income_result["income_source"]
        out["income_estimate_raw"] = income_result["income_estimate"]
        out["income_q25"]         = income_result["income_q25"]
        out["income_q75"]         = income_result["income_q75"]
        out["income_interval_width"] = income_result["income_interval_width"]

        # BCI
        out["bci_score"]      = bci_result["bci_score"]
        out["bci_band"]       = bci_result["bci_band"]
        out["income_haircut"] = bci_result["income_haircut"]
        out["adjusted_income"] = bci_result["adjusted_income"]

        # Affordability
        out["existing_obligations"]  = affordability_result["existing_obligations"]
        out["allowable_obligation"]  = affordability_result["allowable_obligation"]
        out["adsc"]                  = affordability_result["adsc"]
        out["dscr_used"]             = affordability_result["dscr_used"]
        out["dscr_actual"]           = affordability_result["dscr_actual"]
        out["is_affordable"]         = affordability_result["is_affordable"]

        # Decision
        out["final_decision"]  = policy_result["final_decision"]
        out["decision_reason"] = policy_result["decision_reason"]

        return out
