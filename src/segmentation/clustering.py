"""
Persona Clusterer
──────────────────
Assigns income personas (L0 / L1 / L2) to non-PAYROLL, non-THIN customers
using K-means clustering on composite indices (SI, CI, VI, DDI).

Persona system
──────────────
  L0 — Stable structured income
       Highest SI centroid: regular monthly credits, dominant single source.
       Typical profiles: formal employees without SCB payroll tag,
       fixed-income earners, stable freelancers.

  L1 — Structured irregular income
       Medium SI centroid: income present but variable in amount or timing.
       Typical profiles: SME owners, consultants, project-based contractors.

  L2 — Volatile / informal income
       Lowest SI + highest VI centroid: lumpy credits, multiple micro-sources,
       seasonal gaps. Target conservative (P30) income estimate.
       Typical profiles: gig workers, market traders, informal earners.

Deterministic labelling
───────────────────────
  After K-means converges, cluster centroids are sorted by SI_norm descending.
  Rank 0 → L0, Rank 1 → L1, Rank 2 → L2.
  This eliminates random cluster-number permutation across retraining runs.

Soft probabilities (for MoE blending)
──────────────────────────────────────
  P(persona_k | customer) via RBF-softmax over centroid distances:
    w_k = exp(−γ · ||x − c_k||²)
    P_k = w_k / Σ_j w_j
  MoE gate: if max(P) ≥ threshold → single persona model
             else blend top-2 models weighted by their probabilities.
"""

import numpy as np
import pandas as pd
import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# Canonical persona order (always L0, L1, L2)
PERSONA_LABELS: List[str] = ["L0", "L1", "L2"]

# Normalised index columns fed into K-means
CLUSTER_INPUT_COLS: List[str] = ["si_norm", "ci_norm", "vi_norm", "ddi_norm"]


class PersonaClusterer:
    """
    K-means persona assignment on composite income indices.

    Parameters
    ----------
    n_clusters : int
        Number of clusters. Should stay at 3 (L0/L1/L2). Default 3.
    n_init : int
        K-means random restarts — more restarts → more stable centroids.
        Default 20.
    max_iter : int
        Max K-means EM iterations. Default 300.
    rbf_gamma : float
        RBF kernel bandwidth for soft probability computation.
        Lower gamma → flatter probabilities (more uncertainty acknowledged).
        Higher gamma → sharper probabilities (more confident assignments).
        Default 4.0. Calibrated so a customer at their cluster centroid
        (typical inter-centroid distance ≈ 0.7 in normalised index space)
        scores ~85% confidence, which is comfortably above the 0.70 MoE gate.
        Tune via config advanced_modeling.rbf_gamma.
    random_state : int
        K-means seed. Default 42.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_init: int = 20,
        max_iter: int = 300,
        rbf_gamma: float = 4.0,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.rbf_gamma = rbf_gamma
        self.random_state = random_state

        self._kmeans: Optional[KMeans] = None
        self._cluster_to_persona: Optional[Dict[int, str]] = None
        self._persona_to_cluster: Optional[Dict[str, int]] = None
        self.centroids_: Optional[np.ndarray] = None
        self.fitted_ = False

    # ── Fitting ─────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame) -> "PersonaClusterer":
        """
        Fit K-means on normalised index columns and establish the
        deterministic cluster → persona mapping.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain si_norm, ci_norm, vi_norm, ddi_norm columns.
            Should exclude PAYROLL and THIN customers (they bypass clustering).
        """
        X_arr = self._extract(X)

        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._kmeans.fit(X_arr)
        self.centroids_ = self._kmeans.cluster_centers_   # shape: (n_clusters, 4)

        # ── Deterministic persona assignment via centroid SI ranking ───────
        # Sort cluster centroids by SI_norm (column 0) descending.
        # Highest SI → L0, medium → L1, lowest → L2.
        si_col = CLUSTER_INPUT_COLS.index("si_norm")
        centroid_si = self.centroids_[:, si_col]
        sorted_by_si_desc = np.argsort(centroid_si)[::-1]  # highest SI first

        self._cluster_to_persona = {
            int(sorted_by_si_desc[rank]): PERSONA_LABELS[rank]
            for rank in range(self.n_clusters)
        }
        self._persona_to_cluster = {
            persona: cluster for cluster, persona in self._cluster_to_persona.items()
        }

        self.fitted_ = True
        self._log_centroids()
        return self

    # ── Prediction ──────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict persona label for each customer.

        Returns
        -------
        pd.Series of {"L0", "L1", "L2"} with index matching X.
        """
        self._check_fitted()
        cluster_ids = self._kmeans.predict(self._extract(X))
        labels = [self._cluster_to_persona[int(c)] for c in cluster_ids]
        return pd.Series(labels, index=X.index, name="persona")

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Soft persona probabilities via RBF-softmax over centroid distances.

        Formula:
          w_k  = exp(−γ · ||x − centroid_k||²)
          P_k  = w_k / Σ_j w_j

        Returns
        -------
        pd.DataFrame with columns [L0_prob, L1_prob, L2_prob] and index
        matching X. Probabilities sum to 1 per row.
        """
        self._check_fitted()
        X_arr = self._extract(X)

        # Squared Euclidean distance from each customer to each centroid
        # Shape: (n_customers, n_clusters)
        dists_sq = np.array([
            np.sum((X_arr - self.centroids_[k]) ** 2, axis=1)
            for k in range(self.n_clusters)
        ]).T

        # RBF weights → softmax normalisation
        weights = np.exp(-self.rbf_gamma * dists_sq)
        row_sums = weights.sum(axis=1, keepdims=True)
        # Guard against numerical underflow (all distances very large)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        probs = weights / row_sums          # shape: (n_customers, n_clusters)

        # Map cluster columns → canonical L0/L1/L2 order
        prob_df = pd.DataFrame(
            {
                f"{persona}_prob": probs[:, self._persona_to_cluster[persona]]
                for persona in PERSONA_LABELS
            },
            index=X.index,
        )
        return prob_df

    # ── Diagnostics ─────────────────────────────────────────────────────────

    def centroid_summary(self) -> pd.DataFrame:
        """
        Return persona centroid values as a DataFrame (for monitoring/reporting).

        Columns: si_norm, ci_norm, vi_norm, ddi_norm
        Rows:    L0, L1, L2
        """
        self._check_fitted()
        rows = {}
        for cluster_id, persona in self._cluster_to_persona.items():
            rows[persona] = dict(zip(CLUSTER_INPUT_COLS, self.centroids_[cluster_id]))
        return pd.DataFrame(rows).T[CLUSTER_INPUT_COLS].sort_index()

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"PersonaClusterer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "PersonaClusterer":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _extract(self, X: pd.DataFrame) -> np.ndarray:
        missing = [c for c in CLUSTER_INPUT_COLS if c not in X.columns]
        if missing:
            raise ValueError(
                f"PersonaClusterer: missing index columns {missing}. "
                f"Run IndexComputer.transform() first."
            )
        return X[CLUSTER_INPUT_COLS].fillna(0.0).values.astype(float)

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("PersonaClusterer must be fitted before predict.")

    def _log_centroids(self) -> None:
        logger.info("PersonaClusterer fitted — centroid assignments:")
        for cluster_id in range(self.n_clusters):
            persona = self._cluster_to_persona[cluster_id]
            c = self.centroids_[cluster_id]
            logger.info(
                f"  Cluster {cluster_id} → {persona}: "
                f"SI={c[0]:.3f}  CI={c[1]:.3f}  VI={c[2]:.3f}  DDI={c[3]:.3f}"
            )


# ── Backward-compatibility stub ──────────────────────────────────────────────
# BehavioralClusterer is superseded by PersonaClusterer as of Phase 2.
# Any code importing BehavioralClusterer will still work but should migrate.

class BehavioralClusterer:
    """
    Deprecated: replaced by PersonaClusterer in Phase 2.

    BehavioralClusterer used rule-scored assignment to SME / GIG_FREELANCE /
    PASSIVE_INVESTOR / THIN.  PersonaClusterer uses K-means on composite
    indices to produce L0 / L1 / L2 personas, which are more data-driven
    and inclusive for the Thai unstructured income market.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "BehavioralClusterer is deprecated as of Phase 2. "
            "Use PersonaClusterer (via SegmentationPipeline) instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def assign(self, df: pd.DataFrame) -> pd.Series:  # noqa: D102
        raise NotImplementedError(
            "BehavioralClusterer.assign() has been removed. "
            "Use SegmentationPipeline.run() which calls PersonaClusterer."
        )
