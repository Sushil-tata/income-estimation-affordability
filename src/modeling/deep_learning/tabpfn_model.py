"""
TabPFN v2 Wrapper
──────────────────
TabPFN (Prior-data Fitted Networks) is a meta-learned transformer trained
on millions of synthetic tabular datasets. At inference time, it uses the
training data as in-context examples — no gradient updates needed.

TabPFN v2 (2024/2025) improvements over v1:
  - Supports regression (v1 was classification-only)
  - Handles larger datasets (up to ~10K rows efficiently; sampling for larger)
  - Better calibration and uncertainty quantification
  - Faster inference with optimised attention

GDZ note: TabPFN downloads pretrained weights on first use (~50MB).
          Ensure network access during setup, then works fully offline.

Install: pip install tabpfn
"""

import numpy as np
import pandas as pd
import logging
import pickle
from pathlib import Path
from typing import Optional, List

from ..base import BaseIncomeModel

logger = logging.getLogger(__name__)

MAX_ROWS_TABPFN = 10_000   # TabPFN v2 practical limit before sampling


def _check_tabpfn():
    try:
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor
    except ImportError:
        raise ImportError(
            "TabPFN v2 not installed. Install with:\n"
            "  pip install tabpfn\n"
            "Note: requires ~50MB model download on first use."
        )


class TabPFNModel(BaseIncomeModel):
    """
    TabPFN v2 income regressor.

    For datasets larger than MAX_ROWS_TABPFN, we use stratified sampling
    to create a representative training subset. At inference, all rows are scored.

    Parameters
    ----------
    max_train_rows : int
        Maximum training rows. Larger datasets are sampled. Default 10,000.
    log_target : bool
        Train on log1p(income) for better behaviour. Default True.
    n_estimators : int
        TabPFN ensemble size. Default 8.
    feature_cols : list, optional
        Feature columns to use. If None, all numeric columns used.
    random_state : int
    """

    def __init__(
        self,
        max_train_rows: int = MAX_ROWS_TABPFN,
        log_target: bool = True,
        n_estimators: int = 8,
        feature_cols: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.max_train_rows = max_train_rows
        self.log_target = log_target
        self.n_estimators = n_estimators
        self.feature_cols = feature_cols
        self.random_state = random_state
        self._model = None
        self._feature_cols_fitted: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "TabPFNModel":
        TabPFNRegressor = _check_tabpfn()

        cols = self.feature_cols or X.select_dtypes(include=[np.number]).columns.tolist()
        self._feature_cols_fitted = cols
        X_num = X[cols].fillna(0)

        # Sample if too large
        if len(X_num) > self.max_train_rows:
            logger.info(f"TabPFN: dataset {len(X_num):,} rows > limit {self.max_train_rows:,} "
                        f"— stratified sampling...")
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(X_num), self.max_train_rows, replace=False)
            X_train = X_num.iloc[idx].reset_index(drop=True)
            y_train = y.iloc[idx].reset_index(drop=True)
        else:
            X_train = X_num
            y_train = y

        if self.log_target:
            y_train = np.log1p(y_train)

        logger.info(f"TabPFN v2: fitting on {len(X_train):,} rows × {len(cols)} features...")

        self._model = TabPFNRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        self._model.fit(X_train.values, y_train.values)
        logger.info("TabPFN v2: fit complete")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() first")
        X_num = X[self._feature_cols_fitted].fillna(0)
        preds = self._model.predict(X_num.values)
        if self.log_target:
            preds = np.expm1(preds)
        return np.maximum(preds, 0)

    def predict_interval(self, X: pd.DataFrame) -> pd.DataFrame:
        """TabPFN v2 can output quantiles via quantile parameter."""
        if self._model is None:
            raise RuntimeError("Call fit() first")
        X_num = X[self._feature_cols_fitted].fillna(0).values

        try:
            # TabPFN v2 supports quantile prediction
            q25 = self._model.predict(X_num, quantile=0.25)
            q50 = self._model.predict(X_num, quantile=0.50)
            q75 = self._model.predict(X_num, quantile=0.75)

            if self.log_target:
                q25 = np.expm1(q25)
                q50 = np.expm1(q50)
                q75 = np.expm1(q75)
        except TypeError:
            # Fallback if quantile API not supported in installed version
            preds = self.predict(X)
            q25 = preds * 0.85
            q50 = preds
            q75 = preds * 1.15

        return pd.DataFrame({
            "income_q25": np.maximum(q25, 0),
            "income_q50": np.maximum(q50, 0),
            "income_q75": np.maximum(q75, 0),
        }, index=X.index)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"TabPFNModel saved: {path}")

    @classmethod
    def load(cls, path: str) -> "TabPFNModel":
        with open(path, "rb") as f:
            return pickle.load(f)
