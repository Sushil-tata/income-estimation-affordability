"""
Label Engineering
──────────────────
Constructs segment-optimised income labels from transaction features
and verified income records.

Core insight: "verified income" from a lending application is noisy.
For each segment, there is a better proxy label constructable from
transaction behavior that reduces noise before the model ever sees it.

Label strategies:
─────────────────
  1. RAW              : verified_income as-is (baseline)
  2. COMPOSITE        : b1*median_credit + b2*stability - b3*volatility  (OLS-fit coefficients)
  3. QUANTILE         : P-th percentile of monthly credits (segment-specific P)
  4. SHRUNK_COMPOSITE : Composite shrunk toward segment-level mean (James-Stein)
  5. ROBUST           : Winsorized verified income (trim top/bottom 2.5%)
  6. LOG              : log(verified_income) — for log-scale model targets

Segment-specific default quantile targets:
  PAYROLL          → P50  (stable, use median)
  SALARY_LIKE      → P40  (slight conservatism)
  SME              → P30  (gross revenue >> income; conservative)
  GIG_FREELANCE    → P25  (highly volatile; very conservative)
  PASSIVE_INVESTOR → P40
  THIN             → P35  (limited data; lean conservative)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from sklearn.linear_model import Ridge
from scipy import stats
import logging

logger = logging.getLogger(__name__)


# Segment-specific quantile targets for QUANTILE strategy
SEGMENT_QUANTILE_TARGETS = {
    "PAYROLL":          0.50,
    "SALARY_LIKE":      0.40,
    "SME":              0.30,
    "GIG_FREELANCE":    0.25,
    "PASSIVE_INVESTOR": 0.40,
    "THIN":             0.35,
}

# Composite label feature columns (must exist in X)
COMPOSITE_FEATURES = [
    "median_monthly_credit_12m",        # b1  (primary income signal)
    "cv_monthly_credit_12m",            # b2  (stability — lower = better)
    "avg_commitment_amount_12m",        # b3  (expense volatility proxy)
    "avg_recurring_credit_12m",         # b4  (recurring salary signal)
    "avg_investment_credit_12m",        # b5  (passive income signal)
]


class LabelEngineer:
    """
    Constructs and evaluates segment-specific income labels.

    Workflow:
      1. fit()     — estimate composite coefficients from verified income subset
      2. transform() — apply best label strategy per segment
      3. evaluate() — compare label strategies by cross-validated MAE

    Parameters
    ----------
    strategies : list
        Label strategies to evaluate. Default all five.
    shrinkage_alpha : float
        Ridge regularisation for composite label OLS. Default 1.0.
    winsor_pct : float
        Winsorization percentile for ROBUST strategy. Default 0.025 (2.5%).
    random_state : int
    """

    STRATEGIES = ["raw", "composite", "quantile", "shrunk_composite", "robust", "log"]

    def __init__(
        self,
        strategies: Optional[List[str]] = None,
        shrinkage_alpha: float = 1.0,
        winsor_pct: float = 0.025,
        random_state: int = 42,
    ):
        self.strategies = strategies or self.STRATEGIES
        self.shrinkage_alpha = shrinkage_alpha
        self.winsor_pct = winsor_pct
        self.random_state = random_state

        # Fitted state
        self.composite_coefs_: Optional[pd.Series] = None   # Global OLS coefficients
        self.segment_coefs_: Dict[str, pd.Series] = {}      # Per-segment OLS coefficients
        self.segment_means_: Dict[str, float] = {}          # Segment income means (for shrinkage)
        self.grand_mean_: float = 0.0
        self.best_strategy_per_segment_: Dict[str, str] = {}
        self.evaluation_results_: Optional[pd.DataFrame] = None

    # ── PUBLIC API ──────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y_verified: pd.Series,
        segment_col: str = "segment",
    ) -> "LabelEngineer":
        """
        Estimate composite label coefficients from verified income records.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with COMPOSITE_FEATURES and segment_col.
        y_verified : pd.Series
            Verified gross income (THB/month). ~160K records.
        segment_col : str
            Column identifying behavioral segment.
        """
        logger.info(f"LabelEngineer.fit: {len(X):,} verified income records")

        # Global composite coefficients
        self.grand_mean_ = float(y_verified.mean())
        self.composite_coefs_ = self._fit_composite(X, y_verified)

        # Per-segment composite coefficients + means
        if segment_col in X.columns:
            for seg in X[segment_col].unique():
                mask = X[segment_col] == seg
                if mask.sum() < 100:
                    logger.warning(f"Segment {seg}: only {mask.sum()} rows — using global coefficients")
                    self.segment_coefs_[seg] = self.composite_coefs_
                else:
                    self.segment_coefs_[seg] = self._fit_composite(X[mask], y_verified[mask])
                self.segment_means_[seg] = float(y_verified[mask].mean())

        logger.info("LabelEngineer fitted.")
        logger.info(f"  Grand mean income: {self.grand_mean_:,.0f} THB")
        logger.info(f"  Composite coefs: {self.composite_coefs_.round(3).to_dict()}")
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y_verified: pd.Series,
        strategy: str = "auto",
        segment_col: str = "segment",
    ) -> pd.Series:
        """
        Apply label strategy and return engineered label.

        Parameters
        ----------
        strategy : str
            'auto' uses best strategy per segment (requires evaluate() first).
            Otherwise one of: 'raw', 'composite', 'quantile', 'shrunk_composite',
            'robust', 'log'.
        """
        if strategy == "auto":
            if not self.best_strategy_per_segment_:
                logger.warning("No evaluate() run yet — using 'robust' as default")
                strategy = "robust"
            else:
                return self._apply_per_segment_strategy(X, y_verified, segment_col)

        return self._apply_strategy(X, y_verified, strategy, segment_col)

    def evaluate(
        self,
        X: pd.DataFrame,
        y_verified: pd.Series,
        segment_col: str = "segment",
        cv_folds: int = 5,
    ) -> pd.DataFrame:
        """
        Cross-validate each label strategy by asking: when we train on this label,
        how well do we predict verified income on a hold-out set?

        Best strategy per segment is selected and stored in best_strategy_per_segment_.

        Returns
        -------
        pd.DataFrame : strategy × segment MAE table
        """
        from sklearn.model_selection import KFold
        import lightgbm as lgb

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        feature_cols = [c for c in X.select_dtypes(include=[np.number]).columns
                        if c != segment_col]

        results = []
        segments = X[segment_col].unique() if segment_col in X.columns else ["ALL"]

        for seg in segments:
            if segment_col in X.columns:
                mask = X[segment_col] == seg
                X_seg = X[mask].reset_index(drop=True)
                y_seg = y_verified[mask].reset_index(drop=True)
            else:
                X_seg, y_seg = X, y_verified

            if len(X_seg) < 200:
                continue

            for strat in self.strategies:
                try:
                    y_label = self._apply_strategy(X_seg, y_seg, strat, segment_col)
                except Exception as e:
                    logger.warning(f"Strategy {strat} failed for {seg}: {e}")
                    continue

                fold_maes = []
                for train_idx, val_idx in kf.split(X_seg):
                    X_tr = X_seg.iloc[train_idx][feature_cols]
                    y_tr = y_label.iloc[train_idx]
                    X_val = X_seg.iloc[val_idx][feature_cols]
                    y_val_true = y_seg.iloc[val_idx]   # Always evaluate vs verified income

                    model = lgb.LGBMRegressor(
                        n_estimators=200, learning_rate=0.1,
                        max_depth=4, verbose=-1, n_jobs=-1,
                        random_state=self.random_state,
                    )
                    model.fit(X_tr.fillna(0), y_tr)
                    preds = model.predict(X_val.fillna(0))
                    fold_maes.append(np.mean(np.abs(y_val_true - preds)))

                results.append({
                    "segment": seg,
                    "strategy": strat,
                    "cv_mae": np.mean(fold_maes),
                    "cv_mae_std": np.std(fold_maes),
                    "n_samples": len(X_seg),
                })

        self.evaluation_results_ = pd.DataFrame(results)

        # Select best strategy per segment
        for seg in self.evaluation_results_["segment"].unique():
            seg_df = self.evaluation_results_[self.evaluation_results_["segment"] == seg]
            best = seg_df.loc[seg_df["cv_mae"].idxmin(), "strategy"]
            self.best_strategy_per_segment_[seg] = best
            logger.info(f"  Segment {seg}: best label strategy = {best} "
                        f"(MAE = {seg_df['cv_mae'].min():,.0f})")

        return self.evaluation_results_

    def get_all_labels(
        self,
        X: pd.DataFrame,
        y_verified: pd.Series,
        segment_col: str = "segment",
    ) -> pd.DataFrame:
        """Return all strategy labels side by side for analysis."""
        labels = {"verified_income": y_verified}
        for strat in self.strategies:
            try:
                labels[f"label_{strat}"] = self._apply_strategy(X, y_verified, strat, segment_col)
            except Exception as e:
                logger.warning(f"Strategy {strat} failed: {e}")
        return pd.DataFrame(labels)

    # ── STRATEGY IMPLEMENTATIONS ─────────────────────────────────────────────

    def _apply_strategy(
        self,
        X: pd.DataFrame,
        y_verified: pd.Series,
        strategy: str,
        segment_col: str,
    ) -> pd.Series:
        if strategy == "raw":
            return y_verified.copy()

        elif strategy == "robust":
            lo = y_verified.quantile(self.winsor_pct)
            hi = y_verified.quantile(1 - self.winsor_pct)
            return y_verified.clip(lo, hi)

        elif strategy == "log":
            return np.log1p(y_verified)

        elif strategy == "composite":
            return self._composite_label(X, segment_col, shrink=False)

        elif strategy == "shrunk_composite":
            return self._composite_label(X, segment_col, shrink=True)

        elif strategy == "quantile":
            return self._quantile_label(X, y_verified, segment_col)

        else:
            raise ValueError(f"Unknown strategy: {strategy}. "
                             f"Choose from {self.STRATEGIES}")

    def _apply_per_segment_strategy(
        self,
        X: pd.DataFrame,
        y_verified: pd.Series,
        segment_col: str,
    ) -> pd.Series:
        result = pd.Series(np.nan, index=X.index)
        for seg, strat in self.best_strategy_per_segment_.items():
            mask = X[segment_col] == seg if segment_col in X.columns else pd.Series(True, index=X.index)
            if mask.any():
                result[mask] = self._apply_strategy(
                    X[mask], y_verified[mask], strat, segment_col
                ).values
        result.fillna(y_verified, inplace=True)
        return result

    def _composite_label(
        self,
        X: pd.DataFrame,
        segment_col: str,
        shrink: bool = False,
    ) -> pd.Series:
        """
        income_label = b0 + b1*median_credit + b2*(1 - CV) + b3*commitment - ...

        Coefficients fitted per segment via Ridge regression.
        Shrinkage: blend segment estimate with grand mean:
          shrunk = (1 - λ) * composite + λ * grand_mean
          λ = 1 / (1 + n_seg / n_total)   (James-Stein type)
        """
        feat_cols = [c for c in COMPOSITE_FEATURES if c in X.columns]
        if not feat_cols:
            raise ValueError(f"No composite features found. Need: {COMPOSITE_FEATURES}")

        labels = pd.Series(np.nan, index=X.index)

        segments = X[segment_col].unique() if segment_col in X.columns else ["ALL"]

        for seg in segments:
            if segment_col in X.columns:
                mask = X[segment_col] == seg
            else:
                mask = pd.Series(True, index=X.index)

            X_seg = X[mask][feat_cols].fillna(0)
            coefs = self.segment_coefs_.get(seg, self.composite_coefs_)
            available_feats = [f for f in coefs.index if f != "intercept" and f in X_seg.columns]
            intercept = coefs.get("intercept", 0)

            pred = intercept + X_seg[available_feats].values @ coefs[available_feats].values

            if shrink and seg in self.segment_means_:
                n_seg = mask.sum()
                n_total = len(X)
                lam = 1 / (1 + n_seg / max(n_total, 1))
                seg_mean = self.segment_means_[seg]
                pred = (1 - lam) * pred + lam * seg_mean

            pred = np.clip(pred, 5_000, 1_000_000)
            labels[mask] = pred

        return labels.fillna(y_verified if hasattr(self, '_y') else 0)

    def _quantile_label(
        self,
        X: pd.DataFrame,
        y_verified: pd.Series,
        segment_col: str,
    ) -> pd.Series:
        """
        Segment-specific quantile of verified income.

        E.g. SME: P30 of all SME verified incomes → conservative income target.
        Applied as: each customer's label = that quantile of their segment's verified distribution.
        """
        result = pd.Series(np.nan, index=X.index)

        segments = X[segment_col].unique() if segment_col in X.columns else ["ALL"]

        for seg in segments:
            q = SEGMENT_QUANTILE_TARGETS.get(seg, 0.40)
            if segment_col in X.columns:
                mask = X[segment_col] == seg
            else:
                mask = pd.Series(True, index=X.index)

            y_seg = y_verified[mask]
            quantile_val = y_seg.quantile(q)
            # Each customer in segment gets the segment-level quantile as a shrinkage anchor
            # blended with their own verified income
            blended = 0.70 * y_seg + 0.30 * quantile_val
            result[mask] = blended.values

        return result.fillna(y_verified)

    def _fit_composite(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Fit Ridge regression for composite label coefficients."""
        feat_cols = [c for c in COMPOSITE_FEATURES if c in X.columns]
        if not feat_cols:
            return pd.Series({"intercept": float(y.mean())})

        X_feat = X[feat_cols].fillna(0)
        ridge = Ridge(alpha=self.shrinkage_alpha, fit_intercept=True)
        ridge.fit(X_feat, y)

        coefs = pd.Series(ridge.coef_, index=feat_cols)
        coefs["intercept"] = ridge.intercept_
        return coefs
