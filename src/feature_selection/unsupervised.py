"""
Unsupervised Feature Selection
────────────────────────────────
Label-free filters applied before any model is trained.
Fast, cheap, and independent of the target variable.

Classes:
  VarianceFilter      — Remove near-zero variance features
  CorrelationCluster  — Remove redundant features within correlation clusters
  PCAReducer          — Optional linear dimensionality reduction (interpretability trade-off)
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VarianceFilter:
    """
    Remove features with variance below a threshold.
    Also removes quasi-constant features (one value dominates > pct_threshold).

    Parameters
    ----------
    variance_threshold : float
        Minimum variance. Default 0.01.
    pct_threshold : float
        If one value accounts for > pct_threshold of rows, drop. Default 0.95.
    """

    def __init__(self, variance_threshold: float = 0.01, pct_threshold: float = 0.95):
        self.variance_threshold = variance_threshold
        self.pct_threshold = pct_threshold
        self.dropped_: List[str] = []
        self.kept_: List[str] = []

    def fit(self, X: pd.DataFrame) -> "VarianceFilter":
        dropped = []
        for col in X.columns:
            if X[col].dtype == object:
                continue
            if X[col].var() < self.variance_threshold:
                dropped.append((col, "low_variance"))
                continue
            top_val_pct = X[col].value_counts(normalize=True).iloc[0]
            if top_val_pct >= self.pct_threshold:
                dropped.append((col, f"quasi_constant({top_val_pct:.0%})"))

        self.dropped_ = [c for c, _ in dropped]
        self.kept_ = [c for c in X.columns if c not in self.dropped_]

        logger.info(f"VarianceFilter: dropped {len(self.dropped_)} / {len(X.columns)} features")
        for col, reason in dropped[:10]:
            logger.debug(f"  Dropped {col}: {reason}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.kept_]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def report(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature": self.dropped_,
            "reason": ["low_variance_or_quasi_constant"] * len(self.dropped_)
        })


class CorrelationCluster:
    """
    Identifies clusters of highly correlated features and retains one representative
    per cluster (the one with highest variance, as a proxy for signal richness).

    Algorithm:
      1. Compute absolute Pearson correlation matrix
      2. Build adjacency: features with |r| > threshold are linked
      3. Find connected components (clusters)
      4. Within each cluster, keep the feature with highest variance
      5. Drop the rest

    Parameters
    ----------
    threshold : float
        Correlation threshold above which features are considered redundant. Default 0.92.
    method : str
        Correlation method: 'pearson', 'spearman'. Default 'pearson'.
    """

    def __init__(self, threshold: float = 0.92, method: str = "pearson"):
        self.threshold = threshold
        self.method = method
        self.dropped_: List[str] = []
        self.kept_: List[str] = []
        self.clusters_: List[List[str]] = []
        self.cluster_representatives_: dict = {}

    def fit(self, X: pd.DataFrame) -> "CorrelationCluster":
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_num = X[numeric_cols].fillna(0)

        logger.info(f"CorrelationCluster: computing {self.method} correlation on {len(numeric_cols)} features...")
        corr = X_num.corr(method=self.method).abs()

        # Build clusters using union-find
        clusters = self._find_clusters(corr, numeric_cols)
        self.clusters_ = clusters

        kept = []
        dropped = []
        for cluster in clusters:
            if len(cluster) == 1:
                kept.append(cluster[0])
                self.cluster_representatives_[cluster[0]] = cluster[0]
            else:
                # Keep highest-variance feature as representative
                variances = X_num[cluster].var()
                representative = variances.idxmax()
                kept.append(representative)
                self.cluster_representatives_[representative] = cluster
                for feat in cluster:
                    if feat != representative:
                        dropped.append(feat)

        # Preserve non-numeric columns
        non_numeric = [c for c in X.columns if c not in numeric_cols]
        self.kept_ = kept + non_numeric
        self.dropped_ = dropped

        logger.info(f"CorrelationCluster: {len(clusters)} clusters → "
                    f"kept {len(kept)} / {len(numeric_cols)} numeric features "
                    f"(dropped {len(dropped)} redundant)")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.kept_ if c in X.columns]
        return X[available]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def _find_clusters(self, corr: pd.DataFrame, cols: List[str]) -> List[List[str]]:
        """Find connected components in the correlation graph."""
        n = len(cols)
        col_idx = {c: i for i, c in enumerate(cols)}
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if j <= i:
                    continue
                if corr.loc[c1, c2] >= self.threshold:
                    union(i, j)

        # Group by root
        from collections import defaultdict
        groups = defaultdict(list)
        for i, col in enumerate(cols):
            groups[find(i)].append(col)

        return list(groups.values())

    def report(self) -> pd.DataFrame:
        rows = []
        for rep, cluster in self.cluster_representatives_.items():
            if isinstance(cluster, list):
                for feat in cluster:
                    rows.append({"representative": rep, "feature": feat,
                                 "kept": feat == rep})
        return pd.DataFrame(rows)
