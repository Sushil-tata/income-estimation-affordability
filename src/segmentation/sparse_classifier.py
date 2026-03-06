"""
SPARSE Data-Sufficiency Classifier
────────────────────────────────────
Binary classifier that predicts whether reliable income estimation is feasible
for a given customer, based on the quality and depth of available transaction data.

Architecture role (v5.0 Layer B)
──────────────────────────────────
  SPARSE is the operational state for cases where this classifier predicts that
  reliable income estimation is not feasible from available information.

  This classifier is a trained binary model, not an OQS threshold rule.
  It learns from proxy labels (bootstrap phase) and generalises to cases that
  have poor data quality but do not precisely match any single rule trigger.

Label derivation — bootstrap phase (no verified labels)
─────────────────────────────────────────────────────────
  Proxy SPARSE = 1 when any of the following conditions hold:
    1. months_data_available < 2          → INSUFFICIENT_DEPTH (below eligibility floor)
    2. zero_credit_gap_ratio > 0.60       → EXTREME_GAPS (>60% months have no credit)
    3. transaction_count_avg < 1.5
       AND months_data_available < 4      → LOW_DENSITY
    4. dormancy_gap_max >= months − 1     → NEAR_DORMANT (account almost fully dormant)

  These rules generate proxy labels. The classifier *learns* from them, meaning it
  picks up combinations and intermediate cases the rules miss.

  Target state (P-03): replace proxy labels with verified outcome labels
  (income verification failures, VH-band error cases) after the 6-month vintage.

Output schema (per customer, from predict_full())
──────────────────────────────────────────────────
  is_sparse            : bool   — True → route to REFER, skip income estimation
  p_sparse             : float  — calibrated P(SPARSE) ∈ [0, 1]
  sparse_reason_code   : str    — dominant trigger or 'MODEL_PREDICTED'
  oqs_score            : float  — Observation Quality Score ∈ [0, 1]
  feature_masking_count: int    — number of feature groups masked [0, 6]
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from sklearn.calibration import CalibratedClassifierCV
    _HAS_SKL = True
except ImportError:
    _HAS_SKL = False

logger = logging.getLogger(__name__)

# ── Feature Validity Matrix thresholds (v5.0 Layer A) ─────────────────────────
# Each entry: minimum months required for the feature group to be valid.
# If a customer has fewer months, that group is masked (counted here).
_GROUP_MIN_MONTHS = {
    "G1_credit_3m":   2,
    "G2_credit_6m":   4,
    "G3_debit_3m":    2,
    "G4_debit_6m":    4,
    "G5_seasonality": 5,
    "G6_merchant":    3,
}

# SPARSE classifier input features — strictly no _12m required features
_SPARSE_FEATURES = [
    "months_data_available",          # Core depth signal
    "oqs_score",                      # Composite data quality
    "feature_masking_count",          # Groups masked due to insufficient history
    "transaction_count_avg_monthly",  # Account activity density
    "months_with_zero_credit",        # Credit gaps
    "dormancy_gap_max",               # Max consecutive dormant months
    "credit_cv_6m",                   # Credit consistency in recent window
    "avg_eom_balance_3m",             # Account level indicator
    "balance_volatility_6m",          # Balance stability
    "months_below_1000_balance",      # Near-zero balance frequency
    "pass_through_score",             # High = transit money, harder to estimate income
    "retention_ratio_6m",             # Usability of credits as income signal
    "has_payroll_credit",             # Strong negative predictor for SPARSE
]

# Reason codes — assigned to SPARSE cases in priority order
_REASON_INSUFFICIENT_DEPTH = "INSUFFICIENT_DEPTH"
_REASON_EXTREME_GAPS        = "EXTREME_GAPS"
_REASON_LOW_DENSITY         = "LOW_DENSITY"
_REASON_NEAR_DORMANT        = "NEAR_DORMANT"
_REASON_MODEL_PREDICTED     = "MODEL_PREDICTED"

# Safe feature defaults — used when a feature is absent in the input matrix.
# Defaults are set conservatively (lean toward estimable, not SPARSE) so that
# the classifier's positive SPARSE predictions are data-driven, not imputed.
_FEATURE_DEFAULTS = {
    "months_data_available":          4.0,
    "oqs_score":                      0.5,
    "feature_masking_count":          2.0,
    "transaction_count_avg_monthly":  8.0,
    "months_with_zero_credit":        0.0,
    "dormancy_gap_max":               0.0,
    "credit_cv_6m":                   0.5,
    "avg_eom_balance_3m":             5000.0,
    "balance_volatility_6m":          0.5,
    "months_below_1000_balance":      0.0,
    "pass_through_score":             0.3,
    "retention_ratio_6m":             0.5,
    "has_payroll_credit":             0.0,
}


class SparseClassifier:
    """
    Data-sufficiency binary classifier for SPARSE operational state detection.

    Parameters
    ----------
    sparse_threshold : float
        P(SPARSE) >= this threshold → SPARSE routing. [PROVISIONAL — P-03]
    n_estimators : int
        LightGBM trees.
    learning_rate : float
        LightGBM learning rate.
    max_depth : int
        LightGBM max tree depth.
    num_leaves : int
        LightGBM num leaves.
    min_child_samples : int
        LightGBM min samples per leaf.
    oqs_depth_weight : float
        Weight for months-depth component of OQS (0–1).
    oqs_density_weight : float
        Weight for transaction-density component of OQS (0–1).
        oqs_depth_weight + oqs_density_weight should sum to 1.
    calibration_method : str
        Probability calibration: 'isotonic' or 'sigmoid'.
        Isotonic is preferred when N > 1000; sigmoid for smaller datasets.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        sparse_threshold: float = 0.50,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        num_leaves: int = 15,
        min_child_samples: int = 20,
        oqs_depth_weight: float = 0.50,
        oqs_density_weight: float = 0.50,
        calibration_method: str = "isotonic",
        random_state: int = 42,
    ):
        self.sparse_threshold    = sparse_threshold
        self.n_estimators        = n_estimators
        self.learning_rate       = learning_rate
        self.max_depth           = max_depth
        self.num_leaves          = num_leaves
        self.min_child_samples   = min_child_samples
        self.oqs_depth_weight    = oqs_depth_weight
        self.oqs_density_weight  = oqs_density_weight
        self.calibration_method  = calibration_method
        self.random_state        = random_state

        self._model = None
        self.fitted_ = False

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> "SparseClassifier":
        """
        Fit the SPARSE classifier.

        Parameters
        ----------
        X : pd.DataFrame
            Customer-level feature matrix (FeatureEngineer output).
            Minimum required columns: months_data_available,
            transaction_count_avg_monthly, months_with_zero_credit, dormancy_gap_max.
        y : pd.Series, optional
            Binary SPARSE labels (1 = SPARSE, 0 = not SPARSE), indexed as X.
            If None, bootstrap proxy labels are derived automatically.

            Bootstrap labels (no y): use when deploying on a fresh portfolio with
            no verified income outcomes yet.

            Verified labels (with y): use when refitting after the 6-month vintage.
            Verified labels should flag customers where income verification failed
            or where the income model produced a VH-band error. This is the target
            state for provisional item P-03.

        Returns
        -------
        self
        """
        if not _HAS_LGB:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")
        if not _HAS_SKL:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

        X_aug = self._augment(X)
        X_fit = self._select_features(X_aug)

        if y is None:
            logger.info(
                "SparseClassifier.fit(): no labels supplied — "
                "deriving bootstrap proxy labels from data-quality rules"
            )
            y_fit = self._derive_proxy_labels(X_aug)
        else:
            y_fit = y.reindex(X_aug.index).fillna(0).astype(int)

        n_total  = len(y_fit)
        n_sparse = int(y_fit.sum())
        logger.info(
            f"SparseClassifier.fit(): N={n_total:,}  "
            f"SPARSE={n_sparse:,} ({n_sparse / max(n_total, 1):.1%})"
        )

        if n_sparse < 5:
            logger.warning(
                f"SparseClassifier: only {n_sparse} SPARSE cases — classifier "
                "may not generalise. Add SPARSE-like cases to training data or "
                "reduce sparse_threshold."
            )

        base_lgb = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1,
        )

        # CalibratedClassifierCV: cv=2 minimum when few SPARSE cases
        cv_folds = min(5, max(2, n_sparse))
        self._model = CalibratedClassifierCV(
            base_lgb,
            method=self.calibration_method,
            cv=cv_folds,
        )
        self._model.fit(X_fit, y_fit)

        self.fitted_ = True
        logger.info("SparseClassifier.fit() complete")
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return calibrated probabilities per customer.

        Returns
        -------
        pd.DataFrame
            Columns: ['p_not_sparse', 'p_sparse'], indexed as X.
        """
        self._assert_fitted()
        X_aug  = self._augment(X)
        X_pred = self._select_features(X_aug)
        proba  = self._model.predict_proba(X_pred)
        return pd.DataFrame(
            {"p_not_sparse": proba[:, 0], "p_sparse": proba[:, 1]},
            index=X_aug.index,
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Return binary SPARSE flag (True = SPARSE) using sparse_threshold.
        """
        p = self.predict_proba(X)["p_sparse"]
        return (p >= self.sparse_threshold).rename("is_sparse")

    def predict_full(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return the complete SPARSE output record per customer.

        This is the primary method called by InferencePipeline. Returns
        a fully-specified record for every customer — SPARSE and non-SPARSE —
        so that no downstream system receives an undefined output.

        Returns
        -------
        pd.DataFrame
            Columns:
              is_sparse            : bool
              p_sparse             : float   calibrated P(SPARSE) ∈ [0, 1]
              sparse_reason_code   : str     dominant trigger or MODEL_PREDICTED
              oqs_score            : float   Observation Quality Score ∈ [0, 1]
              feature_masking_count: int     feature groups masked [0, 6]
            Index: customer_id (same as X).
        """
        self._assert_fitted()
        X_aug      = self._augment(X)
        proba      = self.predict_proba(X_aug)
        is_sparse  = proba["p_sparse"] >= self.sparse_threshold
        reason     = self._assign_reason_codes(X_aug, is_sparse)

        result = pd.DataFrame(
            {
                "is_sparse":             is_sparse,
                "p_sparse":              proba["p_sparse"].round(4),
                "sparse_reason_code":    reason,
                "oqs_score":             X_aug["oqs_score"].round(4),
                "feature_masking_count": X_aug["feature_masking_count"].astype(int),
            },
            index=X_aug.index,
        )

        n_sparse = int(is_sparse.sum())
        logger.info(
            f"SparseClassifier: {n_sparse:,}/{len(result):,} "
            f"({n_sparse / max(len(result), 1):.1%}) → SPARSE"
        )
        return result

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Pickle the fitted classifier to directory/sparse_classifier.pkl."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        with open(f"{directory}/sparse_classifier.pkl", "wb") as f:
            pickle.dump(self, f)
        logger.info(f"SparseClassifier saved → {directory}/sparse_classifier.pkl")

    @classmethod
    def load(cls, directory: str) -> "SparseClassifier":
        with open(f"{directory}/sparse_classifier.pkl", "rb") as f:
            return pickle.load(f)

    @classmethod
    def from_config(cls, cfg: dict) -> "SparseClassifier":
        """Construct from the 'sparse_classifier' section of config.yaml."""
        sc = cfg.get("sparse_classifier", {})
        return cls(
            sparse_threshold=sc.get("sparse_threshold", 0.50),
            n_estimators=sc.get("n_estimators", 200),
            learning_rate=sc.get("learning_rate", 0.05),
            max_depth=sc.get("max_depth", 4),
            num_leaves=sc.get("num_leaves", 15),
            min_child_samples=sc.get("min_child_samples", 20),
            oqs_depth_weight=sc.get("oqs_depth_weight", 0.50),
            oqs_density_weight=sc.get("oqs_density_weight", 0.50),
            calibration_method=sc.get("calibration_method", "isotonic"),
            random_state=sc.get("random_state", 42),
        )

    # ── OQS and masking count (public statics for use by other modules) ────────

    def compute_oqs(self, X: pd.DataFrame) -> pd.Series:
        """
        Compute OQS per customer.

        Formula (v5.0 Layer A):
          oqs = depth_weight × min(months_data_available / 6, 1)
              + density_weight × min(transaction_count_avg_monthly / 10, 1)
        """
        depth   = (X["months_data_available"] / 6.0).clip(0, 1)
        density = (X["transaction_count_avg_monthly"] / 10.0).clip(0, 1)
        return (self.oqs_depth_weight * depth + self.oqs_density_weight * density).clip(0, 1)

    @staticmethod
    def compute_masking_count(months: pd.Series) -> pd.Series:
        """
        Deterministically compute feature_masking_count from months_data_available.

        Based on the Feature Validity Matrix (v5.0 Layer A):
          G1 (credit 3M)   : requires ≥ 2M
          G2 (credit 6M)   : requires ≥ 4M
          G3 (debit 3M)    : requires ≥ 2M
          G4 (debit 6M)    : requires ≥ 4M
          G5 (seasonality) : requires ≥ 5M
          G6 (merchant MCC): requires ≥ 3M

        Returns counts in [0, 6]:
          0M–1M : 6 groups masked
          2M–2M : 4 groups masked (G2, G4, G5, G6)
          3M    : 3 groups masked (G2, G4, G5)
          4M    : 1 group masked  (G5)
          5M+   : 0 groups masked
        """
        min_months = list(_GROUP_MIN_MONTHS.values())  # [2, 4, 2, 4, 5, 3]

        def _count(m):
            return sum(1 for req in min_months if m < req)

        return months.apply(_count)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _augment(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Set customer_id as index (if column), and compute oqs_score and
        feature_masking_count if they are not already present in X.
        """
        X_aug = X.copy()
        if "customer_id" in X_aug.columns and X_aug.index.name != "customer_id":
            X_aug = X_aug.set_index("customer_id")

        if "oqs_score" not in X_aug.columns:
            X_aug["oqs_score"] = self.compute_oqs(X_aug)

        if "feature_masking_count" not in X_aug.columns:
            X_aug["feature_masking_count"] = self.compute_masking_count(
                X_aug["months_data_available"]
            )
        return X_aug

    def _select_features(self, X_aug: pd.DataFrame) -> pd.DataFrame:
        """
        Select SPARSE_FEATURES from the augmented matrix, filling any absent
        columns with conservative defaults (see _FEATURE_DEFAULTS).
        """
        result = pd.DataFrame(index=X_aug.index)
        for feat in _SPARSE_FEATURES:
            if feat in X_aug.columns:
                result[feat] = X_aug[feat]
            else:
                result[feat] = _FEATURE_DEFAULTS[feat]
        return result.fillna(_FEATURE_DEFAULTS).astype(float)

    def _derive_proxy_labels(self, X_aug: pd.DataFrame) -> pd.Series:
        """
        Derive bootstrap proxy SPARSE labels from data-quality rules.

        These are used when no verified outcome labels are available.
        The classifier learns from them, generalising beyond exact rule boundaries.

        Rules (evaluated cumulatively — any hit → SPARSE):
          1. months_data_available < 2                        → INSUFFICIENT_DEPTH
          2. months_with_zero_credit / months_data_available > 0.60 → EXTREME_GAPS
          3. transaction_count_avg_monthly < 1.5
             AND months_data_available < 4                    → LOW_DENSITY
          4. dormancy_gap_max >= months_data_available − 1    → NEAR_DORMANT
        """
        months   = X_aug["months_data_available"].clip(lower=0)
        zero_m   = X_aug.get("months_with_zero_credit",        pd.Series(0.0, index=X_aug.index))
        tx_avg   = X_aug.get("transaction_count_avg_monthly",  pd.Series(8.0, index=X_aug.index))
        dormancy = X_aug.get("dormancy_gap_max",               pd.Series(0.0, index=X_aug.index))

        gap_ratio = (zero_m / months.replace(0, np.nan)).fillna(0.0)

        sparse = pd.Series(False, index=X_aug.index)
        sparse |= (months < 2)
        sparse |= (gap_ratio > 0.60)
        sparse |= ((tx_avg < 1.5) & (months < 4))
        sparse |= (dormancy >= (months - 1).clip(lower=0))

        return sparse.astype(int)

    def _assign_reason_codes(
        self,
        X_aug: pd.DataFrame,
        is_sparse: pd.Series,
    ) -> pd.Series:
        """
        Assign human-readable reason codes to SPARSE cases (priority order).
        Non-SPARSE cases receive None.
        """
        months   = X_aug["months_data_available"].clip(lower=0)
        zero_m   = X_aug.get("months_with_zero_credit",        pd.Series(0.0, index=X_aug.index))
        tx_avg   = X_aug.get("transaction_count_avg_monthly",  pd.Series(8.0, index=X_aug.index))
        dormancy = X_aug.get("dormancy_gap_max",               pd.Series(0.0, index=X_aug.index))

        gap_ratio = (zero_m / months.replace(0, np.nan)).fillna(0.0)

        reason = pd.Series(None, index=X_aug.index, dtype=object)
        # Priority 1: depth
        reason[is_sparse & (months < 2)] = _REASON_INSUFFICIENT_DEPTH
        # Priority 2: gaps
        reason[is_sparse & reason.isna() & (gap_ratio > 0.60)] = _REASON_EXTREME_GAPS
        # Priority 3: density
        reason[is_sparse & reason.isna() & (tx_avg < 1.5) & (months < 4)] = _REASON_LOW_DENSITY
        # Priority 4: dormancy
        reason[is_sparse & reason.isna() & (dormancy >= (months - 1).clip(lower=0))] = _REASON_NEAR_DORMANT
        # Catch-all: model fired but no single dominant rule
        reason[is_sparse & reason.isna()] = _REASON_MODEL_PREDICTED

        return reason

    def _assert_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError(
                "SparseClassifier must be fitted before calling predict*(). "
                "Call fit() first."
            )
