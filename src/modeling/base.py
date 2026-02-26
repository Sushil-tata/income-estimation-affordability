"""
Abstract base class for all income estimation models.
Enforces a consistent interface across LightGBM, LSTM, TabPFN, AutoGluon.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any


class BaseIncomeModel(ABC):
    """
    All income models must implement this interface.

    Subclasses: LGBMIncomeModel, VanillaLSTM, BiLSTM, LSTMAttention, TCN,
                TabPFNModel, AutoGluonModel
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "BaseIncomeModel":
        """Train the model."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return point income estimates (THB/month)."""

    def predict_interval(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return prediction interval [q25, q50, q75].
        Default: return point estimate only (override for interval-capable models).
        """
        preds = self.predict(X)
        return pd.DataFrame({
            "income_q25": preds * 0.85,
            "income_q50": preds,
            "income_q75": preds * 1.15,
        }, index=X.index)

    def evaluate(self, X: pd.DataFrame, y: pd.Series, segment: Optional[pd.Series] = None) -> pd.DataFrame:
        """Standard evaluation using income metrics."""
        from .loss_functions import evaluate_income_predictions
        preds = pd.Series(self.predict(X), index=X.index)
        return evaluate_income_predictions(y, preds, segment)

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseIncomeModel":
        """Load model from disk."""

    @property
    def model_name(self) -> str:
        return self.__class__.__name__
