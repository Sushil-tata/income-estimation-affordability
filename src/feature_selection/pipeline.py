"""
Feature Selection Pipeline
────────────────────────────
Orchestrates the full feature selection flow:

  Stage 1 — Unsupervised filter (fast, label-free)
    Step 1a: VarianceFilter     — remove near-zero variance
    Step 1b: CorrelationCluster — remove redundant correlated features

  Stage 2 — Supervised ranking (label-aware, multiple methods)
    Step 2a: SHAPRanker         — LightGBM SHAP values
    Step 2b: BorutaSelector     — all-relevant shadow feature comparison
    Step 2c: MRMRSelector       — minimum redundancy maximum relevance

  Stage 3 — Consensus vote
    Feature kept if selected by >= min_votes methods

  Stage 4 — Optional stability filter
    BootstrapStabilityAnalyzer across resamples

  Stage 5 — Optional segment-specific selection
    Run per segment, union or intersection of selections

Output: final_features_ list + report DataFrame
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict
import logging
import yaml

from .unsupervised import VarianceFilter, CorrelationCluster
from .supervised import SHAPRanker, BorutaSelector, MRMRSelector
from .stability import BootstrapStabilityAnalyzer

logger = logging.getLogger(__name__)


class FeatureSelectionPipeline:
    """
    End-to-end feature selection for income estimation.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.
    top_k : int or float
        Features each supervised method targets. Default 0.6 (60% of post-filter features).
    min_votes : int
        Min supervised methods that must agree to keep a feature. Default 2 (out of 3).
    run_boruta : bool
        Boruta is slow — disable for quick iterations. Default True.
    run_stability : bool
        Run bootstrap stability check. Default False (expensive).
    variance_threshold : float
        VarianceFilter threshold. Default 0.01.
    corr_threshold : float
        CorrelationCluster threshold. Default 0.92.
    n_bootstrap : int
        Bootstrap runs for stability. Default 20.
    min_stability : float
        Min bootstrap stability to keep feature. Default 0.60.
    random_state : int
        Seed. Default 42.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        top_k=0.6,
        min_votes: int = 2,
        run_boruta: bool = True,
        run_stability: bool = False,
        variance_threshold: float = 0.01,
        corr_threshold: float = 0.92,
        n_bootstrap: int = 20,
        min_stability: float = 0.60,
        random_state: int = 42,
    ):
        self.top_k = top_k
        self.min_votes = min_votes
        self.run_boruta = run_boruta
        self.run_stability = run_stability
        self.variance_threshold = variance_threshold
        self.corr_threshold = corr_threshold
        self.n_bootstrap = n_bootstrap
        self.min_stability = min_stability
        self.random_state = random_state

        # State
        self.variance_filter_ = None
        self.corr_cluster_ = None
        self.shap_ranker_ = None
        self.boruta_ = None
        self.mrmr_ = None
        self.stability_ = None

        self.post_unsupervised_features_: List[str] = []
        self.method_selections_: Dict[str, List[str]] = {}
        self.vote_counts_: Optional[pd.Series] = None
        self.final_features_: List[str] = []
        self.selection_report_: Optional[pd.DataFrame] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: Optional[List[str]] = None,
        segment_col: Optional[str] = "segment",
    ) -> "FeatureSelectionPipeline":
        """
        Run full feature selection pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Customer-level feature matrix.
        y : pd.Series
            Income target (verified income).
        feature_cols : list, optional
            Columns to consider. Non-numeric and segment cols excluded.
        segment_col : str
            Segment column — excluded from features automatically.
        """
        # Resolve feature columns
        exclude = {segment_col} if segment_col else set()
        if feature_cols:
            cols = [c for c in feature_cols if c not in exclude]
        else:
            cols = [c for c in X.select_dtypes(include=[np.number]).columns
                    if c not in exclude]

        logger.info(f"FeatureSelectionPipeline: starting with {len(cols)} candidate features, "
                    f"{len(X):,} customers")

        # ── Stage 1: Unsupervised ───────────────────────────────────────────
        logger.info("Stage 1a: VarianceFilter...")
        self.variance_filter_ = VarianceFilter(self.variance_threshold)
        X_vf = self.variance_filter_.fit_transform(X[cols])

        logger.info("Stage 1b: CorrelationCluster...")
        self.corr_cluster_ = CorrelationCluster(self.corr_threshold)
        X_filtered = self.corr_cluster_.fit_transform(X_vf)
        self.post_unsupervised_features_ = X_filtered.columns.tolist()

        logger.info(f"Post-unsupervised: {len(self.post_unsupervised_features_)} features remaining")

        # ── Stage 2: Supervised methods ─────────────────────────────────────
        filtered_cols = self.post_unsupervised_features_

        logger.info("Stage 2a: SHAPRanker...")
        self.shap_ranker_ = SHAPRanker(top_k=self.top_k, random_state=self.random_state)
        self.shap_ranker_.fit(X, y, feature_cols=filtered_cols)
        self.method_selections_["shap"] = self.shap_ranker_.selected_features_

        if self.run_boruta:
            logger.info("Stage 2b: BorutaSelector...")
            self.boruta_ = BorutaSelector(random_state=self.random_state)
            self.boruta_.fit(X, y, feature_cols=filtered_cols)
            self.method_selections_["boruta"] = self.boruta_.selected_features_

        logger.info("Stage 2c: MRMRSelector...")
        self.mrmr_ = MRMRSelector(k=self.top_k)
        self.mrmr_.fit(X, y, feature_cols=filtered_cols)
        self.method_selections_["mrmr"] = self.mrmr_.selected_features_

        # ── Stage 3: Consensus vote ─────────────────────────────────────────
        n_methods = len(self.method_selections_)
        vote_counts = pd.Series(0, index=filtered_cols, dtype=int)
        for method, feats in self.method_selections_.items():
            for f in feats:
                if f in vote_counts.index:
                    vote_counts[f] += 1

        self.vote_counts_ = vote_counts.sort_values(ascending=False)
        consensus = vote_counts[vote_counts >= self.min_votes].index.tolist()

        logger.info(f"Consensus (votes >= {self.min_votes}/{n_methods}): "
                    f"{len(consensus)} features")

        # ── Stage 4: Optional stability ─────────────────────────────────────
        if self.run_stability and len(consensus) > 0:
            logger.info("Stage 4: BootstrapStabilityAnalyzer...")
            self.stability_ = BootstrapStabilityAnalyzer(
                selector_class=SHAPRanker,
                selector_kwargs={"top_k": self.top_k, "random_state": self.random_state},
                n_bootstrap=self.n_bootstrap,
                min_stability=self.min_stability,
                random_state=self.random_state,
            )
            self.stability_.fit(X, y, feature_cols=consensus)
            stable = self.stability_.stable_features_
            self.final_features_ = [f for f in consensus if f in stable]
            logger.info(f"Post-stability: {len(self.final_features_)} features")
        else:
            self.final_features_ = consensus

        # Build report
        self.selection_report_ = self._build_report(filtered_cols)

        logger.info(f"Feature selection complete: {len(cols)} → "
                    f"{len(self.post_unsupervised_features_)} → "
                    f"{len(self.final_features_)} final features")
        return self

    def fit_per_segment(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        segment_col: str = "segment",
        feature_cols: Optional[List[str]] = None,
        combine: str = "union",
    ) -> Dict[str, List[str]]:
        """
        Run feature selection within each segment independently.

        Parameters
        ----------
        combine : str
            How to combine segment selections: 'union' or 'intersection'.
            'union': a feature is kept if selected for any segment.
            'intersection': only features selected for ALL segments.

        Returns
        -------
        dict : segment → selected feature list
        """
        segments = X[segment_col].unique()
        segment_features: Dict[str, List[str]] = {}

        for seg in segments:
            mask = X[segment_col] == seg
            if mask.sum() < 200:
                logger.warning(f"Segment {seg}: only {mask.sum()} rows — skipping segment-specific selection")
                continue

            logger.info(f"Feature selection for segment: {seg} ({mask.sum():,} rows)")
            X_seg = X[mask].reset_index(drop=True)
            y_seg = y[mask].reset_index(drop=True)

            seg_pipeline = FeatureSelectionPipeline(
                top_k=self.top_k,
                min_votes=self.min_votes,
                run_boruta=False,   # Skip Boruta for speed in sub-segments
                run_stability=False,
                variance_threshold=self.variance_threshold,
                corr_threshold=self.corr_threshold,
                random_state=self.random_state,
            )
            seg_pipeline.fit(X_seg, y_seg, feature_cols=feature_cols, segment_col=segment_col)
            segment_features[seg] = seg_pipeline.final_features_

        # Combine
        all_selected = [set(v) for v in segment_features.values()]
        if all_selected:
            if combine == "union":
                combined = set.union(*all_selected)
            else:
                combined = set.intersection(*all_selected)
        else:
            combined = set()

        self.final_features_ = list(combined)
        logger.info(f"Per-segment selection ({combine}): {len(self.final_features_)} features")

        return segment_features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.final_features_ if c in X.columns]
        return X[available]

    def _build_report(self, all_features: List[str]) -> pd.DataFrame:
        rows = []
        for f in all_features:
            row = {"feature": f}
            for method, selected in self.method_selections_.items():
                row[f"selected_by_{method}"] = f in selected
            row["vote_count"] = self.vote_counts_.get(f, 0)
            row["final_selected"] = f in self.final_features_
            if self.stability_ and self.stability_.stability_scores_ is not None:
                row["stability_score"] = self.stability_.stability_scores_.get(f, None)
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("vote_count", ascending=False)
        return df

    def report(self) -> pd.DataFrame:
        """Full feature selection report with method votes and stability."""
        return self.selection_report_ if self.selection_report_ is not None else pd.DataFrame()

    def summary(self) -> dict:
        return {
            "initial_features": len(self.variance_filter_.kept_) if self.variance_filter_ else None,
            "post_variance_filter": len(self.variance_filter_.kept_) if self.variance_filter_ else None,
            "post_correlation_cluster": len(self.post_unsupervised_features_),
            "method_selections": {k: len(v) for k, v in self.method_selections_.items()},
            "consensus_threshold": f">= {self.min_votes} methods",
            "final_features": len(self.final_features_),
        }
