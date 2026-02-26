"""
Income Regressor — Stage 2
────────────────────────────
Quantile regression within each income band for point estimate + uncertainty interval.

Addresses the Tweedie variance problem by:
  1. Containing predictions within band boundaries
  2. Predicting median (Q50) as robust point estimate instead of mean
  3. Producing Q25–Q75 confidence interval → feeds BCI

One LightGBM quantile model per quantile, per segment.
"""

import pandas as pd
import numpy as np
import yaml
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

logger = logging.getLogger(__name__)

BAND_BOUNDARIES = {
    "BAND_1_BELOW_15K": (0, 15_000),
    "BAND_2_15K_30K": (15_000, 30_000),
    "BAND_3_30K_50K": (30_000, 50_000),
    "BAND_4_50K_100K": (50_000, 100_000),
    "BAND_5_ABOVE_100K": (100_000, 500_000),
}


class IncomeRegressor:
    """
    Quantile regression model for within-band income estimation.

    Trains one model per quantile [Q25, Q50, Q75] per income band.
    Point estimate = Q50 (median), clipped to band boundaries.
    Confidence interval = [Q25, Q75].

    Parameters
    ----------
    config_path : str
        Path to config.yaml.
    quantiles : list
        Quantiles to estimate. Default [0.25, 0.50, 0.75].
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        quantiles: List[float] = None,
    ):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.model_params = config["models"]["income_regression"]
        self.quantiles = quantiles or self.model_params.get("quantiles", [0.25, 0.50, 0.75])

        # models[band][quantile] = lgb model
        self.models: Dict[str, Dict[float, lgb.Booster]] = {}
        self.feature_cols: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_income: pd.Series,
        band_col: str = "predicted_band",
        feature_cols: Optional[List[str]] = None,
    ) -> None:
        """
        Train quantile models per income band.

        Parameters
        ----------
        X_train : pd.DataFrame
            Feature matrix including predicted_band column.
        y_income : pd.Series
            Verified gross income (THB/month).
        band_col : str
            Column containing income band assignments.
        feature_cols : list, optional
            Feature columns to use for regression.
        """
        self.feature_cols = feature_cols or [c for c in X_train.columns if c != band_col]
        bands = X_train[band_col].unique()

        for band in bands:
            mask = X_train[band_col] == band
            X_band = X_train.loc[mask, self.feature_cols]
            y_band = y_income[mask]

            if len(X_band) < 50:
                logger.warning(f"Band {band}: only {len(X_band)} records — skipping band-specific model")
                continue

            # Clip targets to band boundaries
            lo, hi = BAND_BOUNDARIES.get(band, (0, 9_999_999))
            y_band_clipped = y_band.clip(lower=lo, upper=hi)

            logger.info(f"Training quantile models for band {band}: {len(X_band):,} records")
            self.models[band] = {}

            for q in self.quantiles:
                params = self._get_params(q)
                dtrain = lgb.Dataset(X_band, label=y_band_clipped)
                self.models[band][q] = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=params.pop("n_estimators", 500),
                )

        logger.info("Quantile regression training complete.")

    def predict(self, X: pd.DataFrame, band_col: str = "predicted_band") -> pd.DataFrame:
        """
        Predict income point estimate and confidence interval.

        Returns
        -------
        pd.DataFrame with columns:
          income_q25, income_q50, income_q75,
          income_estimate (= q50, clipped to band),
          income_interval_width,
          band_used
        """
        result_rows = []

        for idx, row in X.iterrows():
            band = row[band_col]
            feat = row[self.feature_cols].values.reshape(1, -1)
            feat_df = pd.DataFrame(feat, columns=self.feature_cols)

            lo, hi = BAND_BOUNDARIES.get(band, (0, 9_999_999))

            preds = {}
            for q in self.quantiles:
                if band in self.models and q in self.models[band]:
                    val = self.models[band][q].predict(feat_df)[0]
                else:
                    # Fallback: use band midpoint
                    val = (lo + hi) / 2
                preds[q] = np.clip(val, lo, hi)

            q25 = preds.get(0.25, (lo + hi) / 2)
            q50 = preds.get(0.50, (lo + hi) / 2)
            q75 = preds.get(0.75, (lo + hi) / 2)

            result_rows.append({
                "income_q25": q25,
                "income_q50": q50,
                "income_q75": q75,
                "income_estimate": q50,
                "income_interval_width": q75 - q25,
                "band_used": band,
            })

        return pd.DataFrame(result_rows, index=X.index)

    def predict_batch(self, X: pd.DataFrame, band_col: str = "predicted_band") -> pd.DataFrame:
        """
        Batch prediction (more efficient than row-by-row for large datasets).
        """
        all_results = []

        for band in X[band_col].unique():
            mask = X[band_col] == band
            X_band = X[mask]
            feat = X_band[self.feature_cols]

            lo, hi = BAND_BOUNDARIES.get(band, (0, 9_999_999))

            band_preds = {"band_used": band}
            for q in self.quantiles:
                if band in self.models and q in self.models[band]:
                    vals = self.models[band][q].predict(feat)
                else:
                    vals = np.full(len(X_band), (lo + hi) / 2)
                band_preds[q] = np.clip(vals, lo, hi)

            band_df = pd.DataFrame(index=X_band.index)
            band_df["income_q25"] = band_preds.get(0.25, np.full(len(X_band), (lo + hi) / 2))
            band_df["income_q50"] = band_preds.get(0.50, np.full(len(X_band), (lo + hi) / 2))
            band_df["income_q75"] = band_preds.get(0.75, np.full(len(X_band), (lo + hi) / 2))
            band_df["income_estimate"] = band_df["income_q50"]
            band_df["income_interval_width"] = band_df["income_q75"] - band_df["income_q25"]
            band_df["band_used"] = band
            all_results.append(band_df)

        return pd.concat(all_results).reindex(X.index)

    def evaluate(self, X_test: pd.DataFrame, y_income: pd.Series, band_col: str = "predicted_band") -> dict:
        """Evaluate regression performance."""
        preds = self.predict_batch(X_test, band_col)

        mae = mean_absolute_error(y_income, preds["income_estimate"])
        mape = mean_absolute_percentage_error(y_income, preds["income_estimate"])
        median_ae = np.median(np.abs(y_income - preds["income_estimate"]))

        # Band-level error analysis (critical for variance diagnosis)
        eval_df = preds.copy()
        eval_df["actual_income"] = y_income
        eval_df["abs_error"] = np.abs(y_income - preds["income_estimate"])
        eval_df["pct_error"] = eval_df["abs_error"] / y_income.replace(0, np.nan)

        band_errors = eval_df.groupby("band_used")["abs_error"].agg(["mean", "std", "median"])
        logger.info(f"\nMAE: {mae:,.0f} THB | MedAE: {median_ae:,.0f} THB | MAPE: {mape:.2%}")
        logger.info(f"\nError by band:\n{band_errors}")

        return {
            "mae": mae,
            "mape": mape,
            "median_ae": median_ae,
            "band_errors": band_errors,
            "predictions": preds,
        }

    def _get_params(self, quantile: float) -> dict:
        params = {k: v for k, v in self.model_params.items() if k not in ("type", "quantiles")}
        params["objective"] = "quantile"
        params["alpha"] = quantile
        params["metric"] = "quantile"
        params["verbose"] = -1
        return params

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "IncomeRegressor":
        with open(path, "rb") as f:
            return pickle.load(f)
