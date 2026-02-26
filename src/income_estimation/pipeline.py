"""
Income Estimation Pipeline
────────────────────────────
Orchestrates the full two-stage income estimation flow:

  Stage 1 → Income Band Classification  (IncomeBandClassifier)
  Stage 2 → Within-Band Quantile Regression (IncomeRegressor)

Hierarchy respected:
  PAYROLL → income directly from deposits (bypass estimation)
  Others  → Stage 1 → Stage 2 → estimated income + interval
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Optional

from .features import FeatureEngineer
from .band_model import IncomeBandClassifier
from .regression import IncomeRegressor

logger = logging.getLogger(__name__)

PAYROLL_SEGMENT = "PAYROLL"


class IncomeEstimationPipeline:
    """
    End-to-end income estimation pipeline.

    Usage
    -----
    pipeline = IncomeEstimationPipeline(config_path="config/config.yaml")

    # Training
    pipeline.fit(X_train, y_income, segment_col="segment")

    # Inference
    results = pipeline.predict(X_score, segment_col="segment")

    Output columns:
      income_source        : PAYROLL | ESTIMATED
      income_estimate      : Final gross income estimate (THB/month)
      income_q25           : Lower bound
      income_q75           : Upper bound
      income_band          : Predicted band
      model_confidence     : Band classifier confidence (used in BCI)
      income_interval_width: Q75 - Q25 (used in BCI stability)
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.bot_min_income = self.config["bot_norms"]["minimum_income_thb"]
        self.band_classifier = IncomeBandClassifier(config_path)
        self.regressor = IncomeRegressor(config_path)
        self.feature_engineer = FeatureEngineer(
            lookback_months=self.config["features"]["lookback_months"]
        )
        self.is_fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_income: pd.Series,
        segment_col: str = "segment",
        payroll_income_col: Optional[str] = "payroll_income",
        feature_cols: Optional[list] = None,
    ) -> None:
        """
        Train the full income estimation pipeline.

        Excludes PAYROLL customers from training (their income is known).

        Parameters
        ----------
        X_train : pd.DataFrame
            Feature matrix with segment column.
        y_income : pd.Series
            Verified gross income labels (THB/month).
        segment_col : str
            Column name for behavioral segment.
        payroll_income_col : str
            Column containing actual payroll income (used for payroll customers only).
        feature_cols : list, optional
            Feature columns to use.
        """
        # Exclude payroll — they don't need estimation
        non_payroll_mask = X_train[segment_col] != PAYROLL_SEGMENT
        X_model = X_train[non_payroll_mask]
        y_model = y_income[non_payroll_mask]

        logger.info(f"Training on {len(X_model):,} non-payroll customers "
                    f"(excluded {non_payroll_mask.sum()} payroll customers)")

        # Stage 1: Band classifier
        self.band_classifier.fit(X_model, y_model, segment_col=segment_col, feature_cols=feature_cols)

        # Get band predictions for Stage 2 training
        band_preds = self.band_classifier.predict(X_model, segment_col=segment_col)
        X_model_with_band = X_model.copy()
        X_model_with_band["predicted_band"] = band_preds["predicted_band"]

        # Stage 2: Quantile regression within bands
        self.regressor.fit(
            X_model_with_band, y_model, band_col="predicted_band", feature_cols=feature_cols
        )

        self.is_fitted = True
        logger.info("Income estimation pipeline training complete.")

    def predict(
        self,
        X: pd.DataFrame,
        segment_col: str = "segment",
        payroll_income_col: Optional[str] = "payroll_income",
    ) -> pd.DataFrame:
        """
        Predict income for all customers.

        PAYROLL customers → income taken directly from payroll_income_col.
        Others → two-stage estimation.

        Returns
        -------
        pd.DataFrame with columns:
          income_source, income_estimate, income_q25, income_q75,
          income_band, model_confidence, income_interval_width,
          meets_bot_minimum
        """
        result = pd.DataFrame(index=X.index)

        payroll_mask = X[segment_col] == PAYROLL_SEGMENT

        # ── PAYROLL: use verified income ─────────────────────────────────
        if payroll_mask.any():
            payroll_income = (
                X.loc[payroll_mask, payroll_income_col]
                if payroll_income_col in X.columns
                else pd.Series(np.nan, index=X[payroll_mask].index)
            )
            result.loc[payroll_mask, "income_source"] = "PAYROLL"
            result.loc[payroll_mask, "income_estimate"] = payroll_income
            result.loc[payroll_mask, "income_q25"] = payroll_income
            result.loc[payroll_mask, "income_q75"] = payroll_income
            result.loc[payroll_mask, "income_band"] = payroll_income.apply(
                self.band_classifier.assign_band
            )
            result.loc[payroll_mask, "model_confidence"] = 1.0
            result.loc[payroll_mask, "income_interval_width"] = 0.0

        # ── NON-PAYROLL: two-stage estimation ────────────────────────────
        non_payroll_mask = ~payroll_mask
        if non_payroll_mask.any():
            X_est = X[non_payroll_mask]

            # Stage 1: band classification
            band_preds = self.band_classifier.predict(X_est, segment_col=segment_col)
            X_est_with_band = X_est.copy()
            X_est_with_band["predicted_band"] = band_preds["predicted_band"]

            # Stage 2: quantile regression
            reg_preds = self.regressor.predict_batch(X_est_with_band, band_col="predicted_band")

            result.loc[non_payroll_mask, "income_source"] = "ESTIMATED"
            result.loc[non_payroll_mask, "income_estimate"] = reg_preds["income_estimate"]
            result.loc[non_payroll_mask, "income_q25"] = reg_preds["income_q25"]
            result.loc[non_payroll_mask, "income_q75"] = reg_preds["income_q75"]
            result.loc[non_payroll_mask, "income_band"] = band_preds["predicted_band"]
            result.loc[non_payroll_mask, "model_confidence"] = band_preds["model_confidence"]
            result.loc[non_payroll_mask, "income_interval_width"] = reg_preds["income_interval_width"]

        # BOT minimum check
        result["meets_bot_minimum"] = result["income_estimate"] >= self.bot_min_income

        return result

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_income: pd.Series,
        segment_col: str = "segment",
    ) -> dict:
        """Full pipeline evaluation with segment-level error breakdown."""
        preds = self.predict(X_test, segment_col=segment_col)

        non_payroll_mask = X_test[segment_col] != PAYROLL_SEGMENT
        X_eval = X_test[non_payroll_mask]
        y_eval = y_income[non_payroll_mask]
        pred_eval = preds.loc[non_payroll_mask]

        errors = pd.DataFrame({
            "actual": y_eval,
            "predicted": pred_eval["income_estimate"],
            "segment": X_eval[segment_col],
            "band": pred_eval["income_band"],
            "abs_error": np.abs(y_eval - pred_eval["income_estimate"]),
            "pct_error": np.abs(y_eval - pred_eval["income_estimate"]) / y_eval.replace(0, np.nan),
        })

        segment_errors = errors.groupby("segment")["abs_error"].agg(["mean", "std", "median"])
        band_errors = errors.groupby("band")["abs_error"].agg(["mean", "std", "median"])

        logger.info(f"\nError by segment:\n{segment_errors}")
        logger.info(f"\nError by band:\n{band_errors}")

        return {
            "overall_mae": errors["abs_error"].mean(),
            "overall_mape": errors["pct_error"].mean(),
            "segment_errors": segment_errors,
            "band_errors": band_errors,
            "predictions": preds,
        }

    def save(self, directory: str) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.band_classifier.save(f"{directory}/band_classifier.pkl")
        self.regressor.save(f"{directory}/income_regressor.pkl")
        logger.info(f"Pipeline saved to {directory}/")

    @classmethod
    def load(cls, directory: str, config_path: str = "config/config.yaml") -> "IncomeEstimationPipeline":
        pipeline = cls(config_path)
        pipeline.band_classifier = IncomeBandClassifier.load(f"{directory}/band_classifier.pkl")
        pipeline.regressor = IncomeRegressor.load(f"{directory}/income_regressor.pkl")
        pipeline.is_fitted = True
        return pipeline
