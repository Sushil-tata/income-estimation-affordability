"""
Custom Loss Functions
──────────────────────
LightGBM custom objectives and sklearn-compatible scoring functions
for income estimation.

Why custom losses?
  - MSE: punishes outliers heavily → noisy income labels dominate
  - Tweedie: appropriate distribution but high variance with tabular features
  - Huber: robust to outliers, smooth near 0
  - Quantile/Pinball: directly optimise P30, P40, P50 etc. per segment
  - MAPE: relative error — important for income spanning large range (15K–800K THB)
  - Log-RMSE: equivalent to MAPE-like behaviour in log space

Each LightGBM custom objective returns (gradient, hessian).
Each sklearn scorer follows the sign convention (higher = better → negate MAE etc.)

Classes/Functions:
  LossRegistry       — Central registry mapping loss name → (objective_fn, eval_fn)
  huber_objective    — Huber loss LightGBM objective
  quantile_objective — Pinball / quantile loss (alpha configurable)
  mape_objective     — Mean Absolute Percentage Error objective
  log_rmse_objective — Log-space RMSE objective
  tweedie_objective  — Tweedie with configurable variance power p
  segment_quantile_objective — Routes to different quantile targets per segment
"""

import numpy as np
import pandas as pd
from typing import Callable, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


# ── Huber Loss ────────────────────────────────────────────────────────────────

def huber_objective(delta: float = 10_000.0):
    """
    Huber loss LightGBM objective.

    Quadratic for |residual| < delta, linear for larger residuals.
    delta should be set relative to the income scale (~THB).
    Default delta=10,000 THB means small errors penalised quadratically,
    large outlier errors penalised linearly (robust).

    Parameters
    ----------
    delta : float
        Transition point. Default 10,000 THB.

    Note: LightGBM 4.x objective signature is f(preds, dataset).
    """
    def objective(preds: np.ndarray, dataset) -> Tuple[np.ndarray, np.ndarray]:
        y_true = dataset.get_label()
        residual = preds - y_true
        abs_res = np.abs(residual)
        mask = abs_res <= delta
        grad = np.where(mask, residual, delta * np.sign(residual))
        hess = np.where(mask, np.ones_like(residual), delta / (abs_res + 1e-8))
        return grad, hess

    def eval_fn(preds: np.ndarray, dataset) -> Tuple[str, float, bool]:
        y_true = dataset.get_label()
        residual = np.abs(preds - y_true)
        loss = np.where(
            residual <= delta,
            0.5 * residual ** 2,
            delta * (residual - 0.5 * delta)
        ).mean()
        return "huber", float(loss), False   # False = lower is better

    return objective, eval_fn


# ── Quantile (Pinball) Loss ───────────────────────────────────────────────────

def quantile_objective(alpha: float = 0.50):
    """
    Pinball / quantile loss for quantile alpha.

    Optimising this directly produces a model predicting the alpha-quantile
    of the conditional distribution.

    Parameters
    ----------
    alpha : float
        Target quantile. 0.25 = P25, 0.50 = median, etc.
    """
    def objective(preds: np.ndarray, dataset) -> Tuple[np.ndarray, np.ndarray]:
        y_true = dataset.get_label()
        residual = y_true - preds
        grad = np.where(residual >= 0, -alpha, 1 - alpha)
        hess = np.ones_like(grad) * 0.5   # Constant hessian for pinball
        return grad, hess

    def eval_fn(preds: np.ndarray, dataset) -> Tuple[str, float, bool]:
        y_true = dataset.get_label()
        residual = y_true - preds
        loss = np.where(residual >= 0, alpha * residual, (alpha - 1) * residual).mean()
        return f"pinball_q{int(alpha*100)}", float(loss), False

    return objective, eval_fn


# ── MAPE Loss ─────────────────────────────────────────────────────────────────

def mape_objective(eps: float = 1_000.0):
    """
    MAPE-like loss: minimise |y_pred - y_true| / max(y_true, eps).

    eps prevents division by zero for very small incomes.
    With eps=1000 THB, relative errors are stable above that threshold.

    Parameters
    ----------
    eps : float
        Minimum denominator. Default 1,000 THB.
    """
    def objective(preds: np.ndarray, dataset) -> Tuple[np.ndarray, np.ndarray]:
        y_true = dataset.get_label()
        denom = np.maximum(np.abs(y_true), eps)
        residual = preds - y_true
        grad = residual / (denom * len(y_true))
        hess = np.ones_like(grad) / (denom * len(y_true))
        return grad, hess

    def eval_fn(preds: np.ndarray, dataset) -> Tuple[str, float, bool]:
        y_true = dataset.get_label()
        denom = np.maximum(np.abs(y_true), eps)
        mape = np.mean(np.abs(preds - y_true) / denom)
        return "mape", float(mape), False

    return objective, eval_fn


# ── Log-RMSE Loss ─────────────────────────────────────────────────────────────

def log_rmse_objective(shift: float = 1.0):
    """
    RMSE in log space: minimise (log(y_pred + shift) - log(y_true + shift))^2.

    Naturally handles the large income range (15K–800K THB) by treating
    proportional errors equally at all income levels.
    """
    def objective(preds: np.ndarray, dataset) -> Tuple[np.ndarray, np.ndarray]:
        y_true = dataset.get_label()
        log_pred = np.log(np.maximum(preds + shift, shift))
        log_true = np.log(np.maximum(y_true + shift, shift))
        residual = log_pred - log_true
        grad = residual / (preds + shift)
        hess = (1 - residual) / (preds + shift) ** 2
        return grad, hess

    def eval_fn(preds: np.ndarray, dataset) -> Tuple[str, float, bool]:
        y_true = dataset.get_label()
        log_pred = np.log(np.maximum(preds + shift, shift))
        log_true = np.log(np.maximum(y_true + shift, shift))
        rmse = np.sqrt(np.mean((log_pred - log_true) ** 2))
        return "log_rmse", float(rmse), False

    return objective, eval_fn


# ── Tweedie Loss (extended) ───────────────────────────────────────────────────

def tweedie_objective(p: float = 1.5):
    """
    Tweedie deviance loss with configurable variance power p.

    p=1.0  → Poisson (count-like)
    p=1.5  → Compound Poisson-Gamma (good for semi-continuous income)
    p=2.0  → Gamma (positive, log-normal like)
    p=3.0  → Inverse Gaussian

    Parameters
    ----------
    p : float
        Variance power. Default 1.5.
    """
    def objective(preds: np.ndarray, dataset) -> Tuple[np.ndarray, np.ndarray]:
        y_true = dataset.get_label()
        y_pred_safe = np.maximum(preds, 1.0)
        grad = -(y_true * y_pred_safe ** (1 - p) - y_pred_safe ** (2 - p))
        hess = y_true * (1 - p) * y_pred_safe ** (-p) + (2 - p) * y_pred_safe ** (1 - p)
        return grad, hess

    def eval_fn(preds: np.ndarray, dataset) -> Tuple[str, float, bool]:
        y_true = dataset.get_label()
        y_pred_safe = np.maximum(preds, 1.0)
        if p == 2:
            loss = 2 * np.mean(np.log(y_pred_safe / y_true) + y_true / y_pred_safe - 1)
        else:
            loss = np.mean(
                y_true ** (2 - p) / ((1 - p) * (2 - p))
                - y_true * y_pred_safe ** (1 - p) / (1 - p)
                + y_pred_safe ** (2 - p) / (2 - p)
            )
        return f"tweedie_p{p}", float(loss), False

    return objective, eval_fn


# ── Segment-Specific Quantile Objective ──────────────────────────────────────

class SegmentQuantileObjective:
    """
    Routes each sample to a different quantile target based on its segment.

    This allows a single model to be trained with segment-aware quantile targets:
      PAYROLL       → P50
      SALARY_LIKE   → P40
      SME           → P30
      GIG_FREELANCE → P25

    Parameters
    ----------
    segment_quantiles : dict
        Segment → quantile target mapping.
    segment_labels : np.ndarray
        Segment assignment for each training sample (aligned with X, y).
    """

    DEFAULTS = {
        "PAYROLL":          0.50,
        "SALARY_LIKE":      0.40,
        "SME":              0.30,
        "GIG_FREELANCE":    0.25,
        "PASSIVE_INVESTOR": 0.40,
        "THIN":             0.35,
    }

    def __init__(
        self,
        segment_quantiles: Optional[Dict[str, float]] = None,
        segment_labels: Optional[np.ndarray] = None,
    ):
        self.segment_quantiles = segment_quantiles or self.DEFAULTS
        self.segment_labels = segment_labels

    def objective(self, preds: np.ndarray, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """LightGBM custom objective with per-sample alpha."""
        y_true = dataset.get_label()
        alpha = self._get_alpha_vector(len(y_true))
        residual = y_true - preds
        grad = np.where(residual >= 0, -alpha, 1 - alpha)
        hess = np.ones_like(grad) * 0.5
        return grad, hess

    def eval_fn(self, preds: np.ndarray, dataset) -> Tuple[str, float, bool]:
        y_true = dataset.get_label()
        alpha = self._get_alpha_vector(len(y_true))
        residual = y_true - preds
        loss = np.where(residual >= 0, alpha * residual, (alpha - 1) * residual).mean()
        return "seg_quantile", float(loss), False

    def _get_alpha_vector(self, n: int) -> np.ndarray:
        if self.segment_labels is None:
            return np.full(n, 0.40)
        default_alpha = 0.40
        alpha = np.array([
            self.segment_quantiles.get(seg, default_alpha)
            for seg in self.segment_labels
        ], dtype=float)
        return alpha

    def set_segment_labels(self, labels: np.ndarray):
        self.segment_labels = labels


# ── Loss Registry ─────────────────────────────────────────────────────────────

class LossRegistry:
    """
    Central registry mapping loss name to (objective_fn, eval_fn) pair.

    Allows config-driven loss selection in SegmentModelTrainer.

    Usage:
        obj, eval_fn = LossRegistry.get("huber", delta=5000)
        params["objective"] = obj
        model = lgb.train(params, dtrain, feval=eval_fn)

    Available losses:
        huber_5k, huber_10k, huber_20k
        quantile_p25, quantile_p30, quantile_p40, quantile_p50
        mape, log_rmse
        tweedie_p15, tweedie_p20, tweedie_p25
        lgb_tweedie   (built-in LightGBM tweedie, no custom objective needed)
        lgb_quantile  (built-in LightGBM quantile)
    """

    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, factory: Callable, **kwargs):
        cls._registry[name] = (factory, kwargs)

    @classmethod
    def get(cls, name: str, **override_kwargs) -> Tuple[Callable, Callable]:
        """Return (objective_fn, eval_fn) for the named loss."""
        if name not in cls._available():
            raise ValueError(f"Unknown loss: '{name}'. Available: {cls._available()}")

        factory, default_kwargs = cls._registry[name]
        kwargs = {**default_kwargs, **override_kwargs}
        return factory(**kwargs)

    @classmethod
    def _available(cls) -> list:
        return list(cls._registry.keys())

    @classmethod
    def list_losses(cls) -> pd.DataFrame:
        rows = []
        for name, (factory, kwargs) in cls._registry.items():
            rows.append({"name": name, "factory": factory.__name__, "defaults": str(kwargs)})
        return pd.DataFrame(rows)


# ── Populate Registry ─────────────────────────────────────────────────────────

LossRegistry.register("huber_5k",      huber_objective,    delta=5_000)
LossRegistry.register("huber_10k",     huber_objective,    delta=10_000)
LossRegistry.register("huber_20k",     huber_objective,    delta=20_000)
LossRegistry.register("quantile_p25",  quantile_objective, alpha=0.25)
LossRegistry.register("quantile_p30",  quantile_objective, alpha=0.30)
LossRegistry.register("quantile_p40",  quantile_objective, alpha=0.40)
LossRegistry.register("quantile_p50",  quantile_objective, alpha=0.50)
LossRegistry.register("mape",          mape_objective,     eps=1_000)
LossRegistry.register("log_rmse",      log_rmse_objective, shift=1.0)
LossRegistry.register("tweedie_p10",   tweedie_objective,  p=1.0)
LossRegistry.register("tweedie_p15",   tweedie_objective,  p=1.5)
LossRegistry.register("tweedie_p20",   tweedie_objective,  p=2.0)
LossRegistry.register("tweedie_p25",   tweedie_objective,  p=2.5)


# ── Evaluation Metrics ────────────────────────────────────────────────────────

def evaluate_income_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    segment: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Comprehensive income prediction evaluation.

    Reports MAE, MAPE, MedAE, and quantile coverage per segment.
    """
    abs_err = np.abs(y_true - y_pred)
    pct_err = abs_err / np.maximum(y_true, 1_000)

    overall = {
        "segment": "ALL",
        "n": len(y_true),
        "mae": abs_err.mean(),
        "median_ae": abs_err.median(),
        "mape": pct_err.mean(),
        "rmse": np.sqrt((y_true - y_pred).pow(2).mean()),
        "log_rmse": np.sqrt(
            (np.log1p(y_true) - np.log1p(np.maximum(y_pred, 1))).pow(2).mean()
        ),
        "pct_within_20pct": (pct_err < 0.20).mean(),
        "pct_within_50pct": (pct_err < 0.50).mean(),
    }
    rows = [overall]

    if segment is not None:
        for seg in sorted(segment.unique()):
            mask = segment == seg
            ae = abs_err[mask]
            pe = pct_err[mask]
            rows.append({
                "segment": seg,
                "n": mask.sum(),
                "mae": ae.mean(),
                "median_ae": ae.median(),
                "mape": pe.mean(),
                "rmse": np.sqrt(((y_true - y_pred)[mask]).pow(2).mean()),
                "log_rmse": np.sqrt(
                    (np.log1p(y_true[mask]) - np.log1p(np.maximum(y_pred[mask], 1))).pow(2).mean()
                ),
                "pct_within_20pct": (pe < 0.20).mean(),
                "pct_within_50pct": (pe < 0.50).mean(),
            })

    return pd.DataFrame(rows).round(4)
