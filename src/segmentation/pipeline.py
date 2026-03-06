"""
Segmentation Pipeline  (Phase 3)
──────────────────────────────────
Three-stage persona assignment:

  Stage 0 — THIN gate
    Customers with < 6 months of history (data_tier == "THIN") are routed
    directly to the THIN bucket. No ML model runs on them.
    Income policy floor is applied downstream.

  Stage 1 — PAYROLL bypass
    Customers with confirmed SCB payroll deposits (has_payroll_credit == 1)
    are assigned PAYROLL and bypass clustering entirely.
    Their income estimate uses the payroll_income column directly.

  Stage 2 — PersonaRouter (supervised LightGBM, Phase 3)
    All remaining customers are scored by PersonaRouter:
      Router Stage 1: LightGBM binary  → THIN vs non-THIN (soft threshold)
      Router Stage 2: LightGBM multiclass → L0 / L1 / L2 probabilities
    Pseudo-labels come from K-means (Phase 2); router learns to generalise.
    Falls back to K-means + rule-based THIN gate if router not fitted.

Output columns added to the dataframe
──────────────────────────────────────
  persona            : PAYROLL | L0 | L1 | L2 | THIN
  segment            : same as persona (backward compatibility)
  L0_prob, L1_prob, L2_prob
                     : soft probabilities (NaN for PAYROLL/THIN)
  persona_confidence : max(L0_prob, L1_prob, L2_prob); NaN for PAYROLL/THIN
  thin_prob          : P(THIN) from router Stage 1 (NaN for PAYROLL)
  si, ci, vi, ddi    : raw composite indices
  si_norm, ci_norm, vi_norm, ddi_norm
                     : normalised composite indices (0–1 scale)
  use_moe_blend      : True if persona_confidence < moe_confidence_threshold

Fitting
───────
  pipeline.fit(train_features_df)
    → fits IndexComputer (normalisation bounds)
    → fits PersonaClusterer (K-means centroids, for interpretability)
    → generates K-means pseudo-labels
    → fits PersonaRouter on pseudo-labels (Phase 3)

Scoring
───────
  result_df = pipeline.run(features_df)
"""

import pandas as pd
import numpy as np
import yaml
import logging
import pickle
from pathlib import Path
from typing import Optional

from .rules import RuleBasedSegmenter, PAYROLL, UNASSIGNED
from .clustering import PersonaClusterer, PERSONA_LABELS
from .router import PersonaRouter
from income_estimation.indices import IndexComputer

logger = logging.getLogger(__name__)

PERSONA_ORDER = ["PAYROLL", "L0", "L1", "L2", "THIN"]

# THIN threshold: customers below this data tier bypass clustering
_THIN_TIER = "THIN"


class SegmentationPipeline:
    """
    Full segmentation pipeline: THIN gate → PAYROLL bypass → K-means persona.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.
    thin_col : str
        Column holding data tier label (output of FeatureEngineer).
        Default "data_tier".
    moe_confidence_threshold : float
        Minimum P(persona) to use a single persona model (vs. MoE blend).
        Overrides config if provided explicitly.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        thin_col: str = "data_tier",
        moe_confidence_threshold: Optional[float] = None,
    ):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        seg_cfg = cfg.get("segmentation", {})
        adv_cfg = cfg.get("advanced_modeling", {})

        self.thin_col = thin_col
        self.moe_confidence_threshold = (
            moe_confidence_threshold
            or adv_cfg.get("moe_confidence_threshold", 0.70)
        )

        # Rule segmenter: PAYROLL only
        self.rule_segmenter = RuleBasedSegmenter(
            payroll_flag_col=seg_cfg.get("payroll_flag_col", "has_payroll_credit"),
        )

        # Index computer: SI/CI/VI/DDI
        self.index_computer = IndexComputer()

        # Persona clusterer: K-means → L0/L1/L2 (Phase 2 — used to generate pseudo-labels)
        smt_cfg = adv_cfg.get("segment_trainer", {})
        rnd = smt_cfg.get("random_state", 42)
        self.clusterer = PersonaClusterer(
            n_clusters=3,
            n_init=20,
            rbf_gamma=adv_cfg.get("rbf_gamma", 4.0),
            random_state=rnd,
        )

        # Supervised router: trained on K-means pseudo-labels (Phase 3)
        router_cfg = adv_cfg.get("router", {})
        self.router = PersonaRouter(
            thin_threshold=router_cfg.get("thin_threshold", 0.50),
            n_estimators_s1=router_cfg.get("n_estimators_s1", 200),
            n_estimators_s2=router_cfg.get("n_estimators_s2", 300),
            learning_rate=router_cfg.get("learning_rate", 0.05),
            max_depth=router_cfg.get("max_depth", 5),
            random_state=rnd,
        )

        self.fitted_ = False

        # Persona priority for reporting / policy lookup
        self.persona_priority = {
            "PAYROLL": 1,
            "L0":      2,
            "L1":      3,
            "L2":      4,
            "THIN":    5,
        }

    # ── Fitting ─────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "SegmentationPipeline":
        """
        Fit the index normaliser and persona clusterer on the training population.

        Only non-PAYROLL, non-THIN customers are used to fit the clusterer.
        PAYROLL customers are excluded because their income is already verified;
        THIN customers have insufficient data to anchor cluster shapes.

        Parameters
        ----------
        df : pd.DataFrame
            Training customer features (output of FeatureEngineer).
        """
        logger.info(f"SegmentationPipeline.fit(): {len(df):,} customers")

        # Identify THIN and PAYROLL masks
        thin_mask = self._thin_mask(df)
        payroll_mask = self._payroll_mask(df)
        cluster_mask = ~thin_mask & ~payroll_mask

        logger.info(
            f"  THIN: {thin_mask.sum():,}  "
            f"PAYROLL: {payroll_mask.sum():,}  "
            f"→ clusterer: {cluster_mask.sum():,}"
        )

        # 1. Fit IndexComputer on the full non-THIN population (includes PAYROLL
        #    so normalisation bounds reflect the full range of income patterns)
        non_thin = df[~thin_mask]
        self.index_computer.fit(non_thin)

        # 2. Fit PersonaClusterer on non-PAYROLL, non-THIN customers
        if cluster_mask.sum() < 10:
            raise ValueError(
                f"Only {cluster_mask.sum()} customers available for clustering "
                f"(after excluding THIN and PAYROLL). Need at least 10."
            )
        cluster_df = self.index_computer.transform(df[cluster_mask])
        self.clusterer.fit(cluster_df)

        # 3. Collect K-means pseudo-labels for ALL non-PAYROLL customers ──────
        #    THIN label comes from the data tier gate; L0/L1/L2 from K-means.
        non_payroll_mask = ~payroll_mask
        pseudo_labels = pd.Series("THIN", index=df.index)
        if cluster_mask.any():
            kmeans_labels = self.clusterer.predict(cluster_df)
            pseudo_labels[cluster_mask] = kmeans_labels.values

        # 4. Fit supervised router on non-PAYROLL customers ───────────────────
        #    Stage 1 needs THIN + non-THIN; Stage 2 needs L0/L1/L2 only.
        #    We compute indices for non-PAYROLL non-THIN so router has them.
        non_payroll_df = df[non_payroll_mask].copy()
        # Add index features for non-THIN customers (needed by Stage 2)
        non_payroll_non_thin = non_payroll_mask & ~thin_mask
        if non_payroll_non_thin.any():
            indexed_cols = self.index_computer.transform(df[non_payroll_non_thin])
            for col in ["si", "ci", "vi", "ddi", "si_norm", "ci_norm", "vi_norm", "ddi_norm"]:
                non_payroll_df.loc[non_payroll_non_thin[non_payroll_mask].values, col] = \
                    indexed_cols[col].values

        router_labels = pseudo_labels[non_payroll_mask]
        self.router.fit(non_payroll_df, router_labels)

        self.fitted_ = True
        logger.info("SegmentationPipeline fitted (K-means + supervised router).")
        return self

    # ── Scoring ─────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign personas to all customers and add metadata columns.

        Parameters
        ----------
        df : pd.DataFrame
            Customer features (output of FeatureEngineer).

        Returns
        -------
        pd.DataFrame
            Input df extended with:
              persona, segment, L0_prob, L1_prob, L2_prob,
              persona_confidence, si, ci, vi, ddi,
              si_norm, ci_norm, vi_norm, ddi_norm
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before run().")

        result = df.copy()
        n = len(result)

        # Initialise output columns
        result["persona"] = pd.NA
        result["thin_prob"] = np.nan
        for col in [f"{p}_prob" for p in PERSONA_LABELS]:
            result[col] = np.nan
        for col in ["si", "ci", "vi", "ddi", "si_norm", "ci_norm", "vi_norm", "ddi_norm"]:
            result[col] = np.nan

        logger.info(f"SegmentationPipeline.run(): {n:,} customers")

        # ── PAYROLL bypass (always rule-based) ───────────────────────────────
        payroll_mask = self._payroll_mask(result)
        result.loc[payroll_mask, "persona"] = "PAYROLL"
        logger.info(f"  PAYROLL bypass:  {payroll_mask.sum():,}")

        non_payroll_mask = ~payroll_mask

        if non_payroll_mask.any():
            non_payroll_df = result[non_payroll_mask].copy()

            # ── Compute indices for ALL non-PAYROLL (needed by router Stage 2) ──
            # IndexComputer was fitted on non-THIN; values for THIN customers
            # will have low DDI_norm. That's fine — router Stage 1 handles them.
            indexed = self.index_computer.transform(non_payroll_df)
            for col in ["si", "ci", "vi", "ddi", "si_norm", "ci_norm", "vi_norm", "ddi_norm"]:
                result.loc[non_payroll_mask, col] = indexed[col].values

            if self.router.fitted_:
                # ── Phase 3: Supervised router ───────────────────────────────
                personas, proba, thin_prob = self.router.predict_full(indexed)
                result.loc[non_payroll_mask, "persona"] = personas.values
                result.loc[non_payroll_mask, "thin_prob"] = thin_prob
                for col in [f"{p}_prob" for p in PERSONA_LABELS]:
                    result.loc[non_payroll_mask, col] = proba[col].values
                logger.info(
                    f"  Router THIN:     {(personas == 'THIN').sum():,}  "
                    f"L0: {(personas == 'L0').sum():,}  "
                    f"L1: {(personas == 'L1').sum():,}  "
                    f"L2: {(personas == 'L2').sum():,}"
                )
            else:
                # ── Fallback: Phase 2 K-means + rule-based THIN gate ─────────
                logger.info("  Router not fitted — falling back to K-means.")
                thin_mask_local = indexed[self.thin_col] == _THIN_TIER \
                    if self.thin_col in indexed.columns \
                    else (indexed["months_data_available"] < 6
                          if "months_data_available" in indexed.columns
                          else pd.Series(False, index=indexed.index))
                result.loc[non_payroll_mask & result.index.isin(
                    indexed.index[thin_mask_local]), "persona"] = "THIN"
                cluster_local = ~thin_mask_local
                if cluster_local.any():
                    cl_idx = indexed[cluster_local]
                    result.loc[non_payroll_mask & result.index.isin(cl_idx.index), "persona"] = \
                        self.clusterer.predict(cl_idx).values
                    proba = self.clusterer.predict_proba(cl_idx)
                    for col in [f"{p}_prob" for p in PERSONA_LABELS]:
                        result.loc[non_payroll_mask & result.index.isin(cl_idx.index), col] = \
                            proba[col].values

        # ── persona_confidence ───────────────────────────────────────────────
        prob_cols = [f"{p}_prob" for p in PERSONA_LABELS]
        result["persona_confidence"] = result[prob_cols].max(axis=1)

        # ── MoE blending flag ────────────────────────────────────────────────
        # True for L0/L1/L2 customers where confidence is below the MoE gate.
        # PAYROLL and THIN are excluded (they never blend).
        is_clustered = result["persona"].isin(["L0", "L1", "L2"])
        result["use_moe_blend"] = (
            result["persona_confidence"] < self.moe_confidence_threshold
        ) & is_clustered

        # ── Backward-compat: segment column ─────────────────────────────────
        result["segment"] = result["persona"]

        # Summary log
        dist = result["persona"].value_counts()
        logger.info("Persona distribution:")
        for persona in PERSONA_ORDER:
            cnt = dist.get(persona, 0)
            logger.info(f"  {persona}: {cnt:,}  ({cnt / n * 100:.1f}%)")

        moe_pct = result["use_moe_blend"].mean() * 100
        logger.info(
            f"  MoE blending: {result['use_moe_blend'].sum():,} customers "
            f"({moe_pct:.1f}%) below P={self.moe_confidence_threshold} threshold"
        )

        return result

    # ── Diagnostics ─────────────────────────────────────────────────────────

    def get_summary(self, result: pd.DataFrame) -> pd.DataFrame:
        """Return persona distribution summary with BCI/policy context."""
        if "persona" not in result.columns:
            raise ValueError("Run pipeline first.")

        n = len(result)
        rows = []
        for persona in PERSONA_ORDER:
            mask = result["persona"] == persona
            cnt = mask.sum()
            sub = result[mask]
            conf = sub["persona_confidence"].mean() if "persona_confidence" in sub.columns else np.nan
            rows.append({
                "persona": persona,
                "count": cnt,
                "pct": round(cnt / n * 100, 2),
                "avg_confidence": round(conf, 3) if not np.isnan(conf) else None,
                "priority": self.persona_priority.get(persona, 99),
            })
        return (
            pd.DataFrame(rows)
            .sort_values("priority")
            .reset_index(drop=True)
        )

    def centroid_summary(self) -> pd.DataFrame:
        """Return fitted K-means centroids per persona (for monitoring)."""
        if not self.fitted_:
            raise RuntimeError("Call fit() first.")
        return self.clusterer.centroid_summary()

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        with open(f"{directory}/segmentation_pipeline.pkl", "wb") as f:
            pickle.dump(self, f)
        logger.info(f"SegmentationPipeline saved to {directory}/")

    @classmethod
    def load(cls, directory: str) -> "SegmentationPipeline":
        with open(f"{directory}/segmentation_pipeline.pkl", "rb") as f:
            return pickle.load(f)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _thin_mask(self, df: pd.DataFrame) -> pd.Series:
        """True for customers in the THIN data tier."""
        if self.thin_col in df.columns:
            return df[self.thin_col] == _THIN_TIER
        # Fallback: use raw month count if data_tier column not present
        if "months_data_available" in df.columns:
            return df["months_data_available"] < 6
        return pd.Series(False, index=df.index)

    def _payroll_mask(self, df: pd.DataFrame) -> pd.Series:
        """True for customers with confirmed SCB payroll deposits."""
        col = self.rule_segmenter.payroll_flag_col
        if col in df.columns:
            return df[col] == 1
        return pd.Series(False, index=df.index)
