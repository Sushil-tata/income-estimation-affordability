"""
Income Band Classifier — Stage 1
──────────────────────────────────
LightGBM multi-class classifier predicting income band.
Trained on 160K verified lending income records.

Income Bands (THB/month):
  BAND_1_BELOW_15K   : < 15,000
  BAND_2_15K_30K     : 15,000 – 30,000
  BAND_3_30K_50K     : 30,000 – 50,000
  BAND_4_50K_100K    : 50,000 – 100,000
  BAND_5_ABOVE_100K  : > 100,000

Outputs:
  - Predicted income band (label)
  - Probability per band (used in BCI model confidence component)
  - Model confidence score = max_prob - 2nd_max_prob
"""

import pandas as pd
import numpy as np
import yaml
import pickle
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, log_loss
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

BAND_BOUNDARIES = [0, 15_000, 30_000, 50_000, 100_000, 9_999_999]
BAND_LABELS = [
    "BAND_1_BELOW_15K",
    "BAND_2_15K_30K",
    "BAND_3_30K_50K",
    "BAND_4_50K_100K",
    "BAND_5_ABOVE_100K",
]
BAND_MIDPOINTS = {
    "BAND_1_BELOW_15K": 10_000,
    "BAND_2_15K_30K": 22_500,
    "BAND_3_30K_50K": 40_000,
    "BAND_4_50K_100K": 75_000,
    "BAND_5_ABOVE_100K": 150_000,
}


class IncomeBandClassifier:
    """
    LightGBM income band classifier with segment-aware training.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.
    use_segments : bool
        Whether to train separate models per segment or a unified model
        with segment as a feature.
    """

    def __init__(self, config_path: str = "config/config.yaml", use_segments: bool = True):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.model_params = config["models"]["income_band_classifier"]
        self.band_labels = BAND_LABELS
        self.band_midpoints = BAND_MIDPOINTS
        self.use_segments = use_segments
        self.models = {}         # segment → lgb model
        self.feature_cols = []
        self.le = LabelEncoder()

    @staticmethod
    def assign_band(income: float) -> str:
        """Assign income band label to a given income value."""
        for i in range(len(BAND_BOUNDARIES) - 1):
            if BAND_BOUNDARIES[i] <= income < BAND_BOUNDARIES[i + 1]:
                return BAND_LABELS[i]
        return BAND_LABELS[-1]

    def prepare_labels(self, income_series: pd.Series) -> pd.Series:
        """Convert raw income values to band labels."""
        return income_series.apply(self.assign_band)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_income: pd.Series,
        segment_col: Optional[str] = "segment",
        feature_cols: Optional[List[str]] = None,
    ) -> None:
        """
        Train the income band classifier.

        Parameters
        ----------
        X_train : pd.DataFrame
            Feature matrix (includes segment column if use_segments=True).
        y_income : pd.Series
            Verified gross income (THB/month) — training labels.
        segment_col : str
            Column name for behavioral segment.
        feature_cols : list, optional
            Feature columns to use. If None, all non-segment columns used.
        """
        y_bands = self.prepare_labels(y_income)

        if feature_cols:
            self.feature_cols = feature_cols
        else:
            self.feature_cols = [c for c in X_train.columns if c != segment_col]

        logger.info(f"Training income band classifier on {len(X_train):,} records...")
        logger.info(f"Band distribution:\n{y_bands.value_counts()}")

        if self.use_segments and segment_col in X_train.columns:
            self._fit_per_segment(X_train, y_bands, segment_col)
        else:
            self._fit_unified(X_train[self.feature_cols], y_bands)

        logger.info("Training complete.")

    def _fit_unified(self, X: pd.DataFrame, y_bands: pd.Series) -> None:
        y_enc = self.le.fit_transform(y_bands)
        params = {k: v for k, v in self.model_params.items() if k != "type"}
        params["objective"] = "multiclass"
        params["num_class"] = len(self.band_labels)
        params["metric"] = "multi_logloss"
        params["verbose"] = -1

        dtrain = lgb.Dataset(X, label=y_enc)
        self.models["unified"] = lgb.train(
            params,
            dtrain,
            num_boost_round=params.pop("n_estimators", 500),
        )

    def _fit_per_segment(self, X: pd.DataFrame, y_bands: pd.Series, segment_col: str) -> None:
        self.le.fit(y_bands)
        segments = X[segment_col].unique()

        for seg in segments:
            mask = X[segment_col] == seg
            X_seg = X.loc[mask, self.feature_cols]
            y_seg = y_bands[mask]

            if len(X_seg) < 100:
                logger.warning(f"Segment {seg}: only {len(X_seg)} records — using unified model")
                continue

            logger.info(f"Training segment {seg}: {len(X_seg):,} records")
            y_enc = self.le.transform(y_seg)

            params = {k: v for k, v in self.model_params.items() if k != "type"}
            n_est = params.pop("n_estimators", 500)
            params["objective"] = "multiclass"
            params["num_class"] = len(self.band_labels)
            params["metric"] = "multi_logloss"
            params["verbose"] = -1

            dtrain = lgb.Dataset(X_seg, label=y_enc)
            self.models[seg] = lgb.train(params, dtrain, num_boost_round=n_est)

    def predict(self, X: pd.DataFrame, segment_col: Optional[str] = "segment") -> pd.DataFrame:
        """
        Predict income band and probabilities.

        Returns
        -------
        pd.DataFrame with columns:
          predicted_band, band_probability_*, model_confidence
        """
        results = []

        if self.use_segments and segment_col in X.columns:
            for seg in X[segment_col].unique():
                mask = X[segment_col] == seg
                model_key = seg if seg in self.models else "unified"
                if model_key not in self.models:
                    model_key = list(self.models.keys())[0]

                probs = self.models[model_key].predict(X.loc[mask, self.feature_cols])
                seg_result = self._probs_to_output(probs, X[mask].index)
                results.append(seg_result)
            return pd.concat(results).reindex(X.index)
        else:
            probs = self.models["unified"].predict(X[self.feature_cols])
            return self._probs_to_output(probs, X.index)

    def _probs_to_output(self, probs: np.ndarray, index: pd.Index) -> pd.DataFrame:
        band_classes = self.le.classes_
        prob_df = pd.DataFrame(probs, columns=[f"prob_{b}" for b in band_classes], index=index)

        prob_df["predicted_band"] = band_classes[np.argmax(probs, axis=1)]
        sorted_probs = np.sort(probs, axis=1)[:, ::-1]
        prob_df["model_confidence"] = sorted_probs[:, 0] - sorted_probs[:, 1]
        prob_df["predicted_band_midpoint"] = prob_df["predicted_band"].map(self.band_midpoints)

        return prob_df

    def evaluate(self, X_test: pd.DataFrame, y_income: pd.Series, segment_col: Optional[str] = "segment") -> dict:
        """Evaluate classifier performance."""
        y_true_bands = self.prepare_labels(y_income)
        preds = self.predict(X_test, segment_col)

        report = classification_report(y_true_bands, preds["predicted_band"], output_dict=True)
        logger.info(f"\nClassification Report:\n{classification_report(y_true_bands, preds['predicted_band'])}")

        return {
            "classification_report": report,
            "predictions": preds,
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "IncomeBandClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)
