"""
Segmentation Pipeline
──────────────────────
Orchestrates the full segmentation flow:
  Step 1 → Rule-based segmenter  (PAYROLL, SALARY_LIKE)
  Step 2 → Behavioral clusterer  (SME, GIG_FREELANCE, PASSIVE_INVESTOR, THIN)

Output: customer-level segment label + segment confidence score
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path

from .rules import RuleBasedSegmenter, PAYROLL, SALARY_LIKE, UNASSIGNED
from .clustering import BehavioralClusterer

logger = logging.getLogger(__name__)

SEGMENT_ORDER = ["PAYROLL", "SALARY_LIKE", "SME", "GIG_FREELANCE", "PASSIVE_INVESTOR", "THIN"]


class SegmentationPipeline:
    """
    Full segmentation pipeline.

    Usage
    -----
    pipeline = SegmentationPipeline(config_path="config/config.yaml")
    result_df = pipeline.run(features_df)

    Output columns added to dataframe:
      - segment          : Segment label
      - segment_priority : Numeric priority (1=highest confidence)
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        seg_cfg = config["segmentation"]

        self.rule_segmenter = RuleBasedSegmenter(
            cv_threshold=seg_cfg["salary_like_cv_threshold"],
            salary_min_months=seg_cfg["salary_like_min_months"],
        )

        self.clusterer = BehavioralClusterer(
            thin_min_months=seg_cfg["thin_min_months_required"],
            thin_min_tx_monthly=seg_cfg["thin_min_transactions"],
            sme_business_credit_ratio=seg_cfg["sme_business_credit_ratio"],
        )

        self.segment_priority = {
            "PAYROLL": 1,
            "SALARY_LIKE": 2,
            "SME": 3,
            "GIG_FREELANCE": 4,
            "PASSIVE_INVESTOR": 5,
            "THIN": 6,
        }

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run full segmentation pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Customer-level monthly aggregate features.

        Returns
        -------
        pd.DataFrame
            Input dataframe with segment and segment_priority columns added.
        """
        result = df.copy()

        logger.info(f"Running segmentation on {len(df):,} customers...")

        # Step 1: Rule-based assignment
        rule_segments = self.rule_segmenter.assign(df)

        # Step 2: Behavioral clustering for UNASSIGNED
        unassigned_mask = rule_segments == UNASSIGNED
        behavioral_segments = pd.Series(UNASSIGNED, index=df.index)

        if unassigned_mask.sum() > 0:
            behavioral_segments[unassigned_mask] = self.clusterer.assign(
                df[unassigned_mask]
            )

        # Combine
        final_segments = rule_segments.copy()
        final_segments[unassigned_mask] = behavioral_segments[unassigned_mask]

        result["segment"] = final_segments
        result["segment_priority"] = result["segment"].map(self.segment_priority)

        # Summary log
        dist = result["segment"].value_counts()
        logger.info("Segmentation complete:")
        for seg, count in dist.items():
            logger.info(f"  {seg}: {count:,} ({count/len(df)*100:.1f}%)")

        return result

    def get_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return segment distribution summary."""
        if "segment" not in df.columns:
            raise ValueError("Run pipeline first before calling get_summary()")

        summary = (
            df["segment"]
            .value_counts()
            .reset_index()
        )
        summary.columns = ["segment", "count"]
        summary["pct"] = (summary["count"] / len(df) * 100).round(2)
        summary["priority"] = summary["segment"].map(self.segment_priority)
        summary = summary.sort_values("priority").reset_index(drop=True)
        return summary
