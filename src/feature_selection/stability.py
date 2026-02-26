"""
Feature Stability Analyzer
────────────────────────────
Measures how consistently features are selected across bootstrap samples
or across different observation window vintages (Jan–Jul 2024).

A feature selected in only 1 of 7 vintages is an overfit artifact.
A feature selected in 6/7 vintages is genuinely stable and should be trusted.

Classes:
  BootstrapStabilityAnalyzer  — Bootstrap resampling stability
  VintageStabilityAnalyzer    — Cross-vintage (observation point) stability
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Type, Callable
import logging

logger = logging.getLogger(__name__)


class BootstrapStabilityAnalyzer:
    """
    Measures feature selection stability via bootstrap resampling.

    Runs a given selector on n_bootstrap subsamples of the training data
    and records how often each feature is selected.

    Stability score = fraction of bootstrap runs in which feature is selected.
    Features with stability >= min_stability are deemed stable.

    Parameters
    ----------
    selector_class : class
        A feature selector class (e.g. SHAPRanker, BorutaSelector).
    selector_kwargs : dict
        Constructor kwargs for the selector.
    n_bootstrap : int
        Number of bootstrap samples. Default 20.
    sample_frac : float
        Fraction of training data per bootstrap. Default 0.70.
    min_stability : float
        Minimum stability score to keep a feature. Default 0.60.
    random_state : int
        Seed. Default 42.
    """

    def __init__(
        self,
        selector_class,
        selector_kwargs: dict = None,
        n_bootstrap: int = 20,
        sample_frac: float = 0.70,
        min_stability: float = 0.60,
        random_state: int = 42,
    ):
        self.selector_class = selector_class
        self.selector_kwargs = selector_kwargs or {}
        self.n_bootstrap = n_bootstrap
        self.sample_frac = sample_frac
        self.min_stability = min_stability
        self.random_state = random_state
        self.stability_scores_: Optional[pd.Series] = None
        self.stable_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series,
            feature_cols: Optional[List[str]] = None) -> "BootstrapStabilityAnalyzer":
        cols = feature_cols or X.select_dtypes(include=[np.number]).columns.tolist()
        rng = np.random.RandomState(self.random_state)
        selection_counts = pd.Series(0, index=cols, dtype=float)

        logger.info(f"BootstrapStability: {self.n_bootstrap} bootstrap runs × "
                    f"{self.sample_frac:.0%} sample on {len(X):,} rows...")

        for i in range(self.n_bootstrap):
            idx = rng.choice(len(X), size=int(len(X) * self.sample_frac), replace=True)
            X_boot = X.iloc[idx].reset_index(drop=True)
            y_boot = y.iloc[idx].reset_index(drop=True)

            try:
                selector = self.selector_class(**self.selector_kwargs)
                selector.fit(X_boot, y_boot, feature_cols=cols)
                selected = selector.selected_features_
                for f in selected:
                    if f in selection_counts.index:
                        selection_counts[f] += 1
            except Exception as e:
                logger.warning(f"Bootstrap run {i+1} failed: {e}")
                continue

            if (i + 1) % 5 == 0:
                logger.debug(f"  Bootstrap run {i+1}/{self.n_bootstrap} done")

        self.stability_scores_ = (selection_counts / self.n_bootstrap).sort_values(ascending=False)
        self.stable_features_ = self.stability_scores_[
            self.stability_scores_ >= self.min_stability
        ].index.tolist()

        logger.info(f"BootstrapStability: {len(self.stable_features_)} stable features "
                    f"(stability >= {self.min_stability:.0%})")
        return self

    def report(self) -> pd.DataFrame:
        if self.stability_scores_ is None:
            return pd.DataFrame()
        df = self.stability_scores_.reset_index()
        df.columns = ["feature", "stability_score"]
        df["stable"] = df["feature"].isin(self.stable_features_)
        return df.sort_values("stability_score", ascending=False)


class VintageStabilityAnalyzer:
    """
    Measures feature selection stability across observation window vintages.

    The dev window spans Jan 2024 – Jul 2024 (7 observation points).
    A feature stable across vintages is temporally robust.

    Parameters
    ----------
    selector_class : class
        Feature selector class.
    selector_kwargs : dict
        Constructor kwargs.
    min_stability : float
        Minimum fraction of vintages in which feature must be selected.
    vintage_col : str
        Column identifying the observation vintage / snapshot date.
    """

    def __init__(
        self,
        selector_class,
        selector_kwargs: dict = None,
        min_stability: float = 0.60,
        vintage_col: str = "observation_date",
    ):
        self.selector_class = selector_class
        self.selector_kwargs = selector_kwargs or {}
        self.min_stability = min_stability
        self.vintage_col = vintage_col
        self.vintage_results_: dict = {}
        self.stability_scores_: Optional[pd.Series] = None
        self.stable_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series,
            feature_cols: Optional[List[str]] = None) -> "VintageStabilityAnalyzer":
        if self.vintage_col not in X.columns:
            raise ValueError(f"vintage_col '{self.vintage_col}' not found in X")

        vintages = sorted(X[self.vintage_col].unique())
        cols = feature_cols or [c for c in X.select_dtypes(include=[np.number]).columns
                                if c != self.vintage_col]
        all_features = set(cols)
        selection_counts = {f: 0 for f in all_features}

        logger.info(f"VintageStability: {len(vintages)} vintages — {vintages}")

        for v in vintages:
            mask = X[self.vintage_col] == v
            X_v = X[mask][cols].reset_index(drop=True)
            y_v = y[mask].reset_index(drop=True)

            if len(X_v) < 100:
                logger.warning(f"Vintage {v}: only {len(X_v)} rows — skipping")
                continue

            try:
                selector = self.selector_class(**self.selector_kwargs)
                selector.fit(X_v, y_v, feature_cols=cols)
                selected = selector.selected_features_
                self.vintage_results_[v] = selected
                for f in selected:
                    if f in selection_counts:
                        selection_counts[f] += 1
            except Exception as e:
                logger.warning(f"Vintage {v} failed: {e}")

        n_vintages = len(self.vintage_results_)
        self.stability_scores_ = pd.Series(
            {f: cnt / n_vintages for f, cnt in selection_counts.items()}
        ).sort_values(ascending=False)

        self.stable_features_ = self.stability_scores_[
            self.stability_scores_ >= self.min_stability
        ].index.tolist()

        logger.info(f"VintageStability: {len(self.stable_features_)} stable features "
                    f"across {n_vintages} vintages")
        return self

    def report(self) -> pd.DataFrame:
        if self.stability_scores_ is None:
            return pd.DataFrame()
        df = self.stability_scores_.reset_index()
        df.columns = ["feature", "vintage_stability"]
        df["stable"] = df["feature"].isin(self.stable_features_)

        # Add vintage-level detail
        for v, feats in self.vintage_results_.items():
            df[f"selected_in_{v}"] = df["feature"].isin(feats)

        return df.sort_values("vintage_stability", ascending=False)
