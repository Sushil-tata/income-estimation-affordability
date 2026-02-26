"""
Data validation utilities.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

REQUIRED_FEATURE_COLS = [
    "cv_monthly_credit_12m",
    "months_with_zero_credit",
    "months_data_available",
    "dominant_credit_source_share",
    "months_with_salary_pattern",
    "has_payroll_credit",
    "avg_monthly_credit_12m",
    "avg_total_debit_12m",
    "avg_eom_balance_3m",
    "avg_commitment_amount_12m",
    "transaction_count_avg_monthly",
    "business_mcc_credit_share",
]


def validate_features(df: pd.DataFrame, required_cols: Optional[List[str]] = None) -> dict:
    """
    Validate feature dataframe before running the pipeline.

    Returns a validation report dict.
    """
    cols = required_cols or REQUIRED_FEATURE_COLS
    report = {"passed": True, "issues": []}

    # Check required columns
    missing = [c for c in cols if c not in df.columns]
    if missing:
        report["passed"] = False
        report["issues"].append(f"Missing required columns: {missing}")

    # Check for entirely null columns
    if not missing:
        for col in cols:
            null_pct = df[col].isna().mean()
            if null_pct > 0.50:
                report["issues"].append(f"Column '{col}' is {null_pct:.0%} null")

    # Check for duplicate indices
    if df.index.duplicated().any():
        report["passed"] = False
        report["issues"].append("Duplicate customer IDs found in index")

    # Check for negative credits/balances where they shouldn't be
    if "avg_monthly_credit_12m" in df.columns:
        neg_credits = (df["avg_monthly_credit_12m"] < 0).sum()
        if neg_credits > 0:
            report["issues"].append(f"{neg_credits} customers with negative avg credits")

    if report["issues"]:
        logger.warning(f"Validation issues found: {report['issues']}")
    else:
        logger.info(f"Feature validation passed for {len(df):,} customers")

    return report


def validate_income_labels(y: pd.Series) -> dict:
    """
    Validate income labels for training.
    """
    report = {"passed": True, "issues": []}

    if y.isna().any():
        null_count = y.isna().sum()
        report["passed"] = False
        report["issues"].append(f"{null_count} null income labels — drop before training")

    if (y <= 0).any():
        neg_count = (y <= 0).sum()
        report["issues"].append(f"{neg_count} zero/negative income values")

    if y.max() > 1_000_000:
        report["issues"].append(f"Extreme income values detected (max: {y.max():,.0f} THB) — check for outliers")

    pct_below_15k = (y < 15_000).mean()
    logger.info(f"Income label summary: median={y.median():,.0f}, mean={y.mean():,.0f}, "
                f"pct_below_15k={pct_below_15k:.1%}")

    return report
