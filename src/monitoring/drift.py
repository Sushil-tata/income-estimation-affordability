"""
Drift Monitor
──────────────
PSI-based population drift detection for all pipeline outputs.

Monitors four layers:
  1. Feature drift        — behavioral feature distributions
  2. Persona drift        — L0/L1/L2/THIN/PAYROLL composition
  3. Income drift         — income estimate distribution
  4. BCI drift            — BCI score distribution + band composition

PSI thresholds (industry standard)
───────────────────────────────────
  PSI < 0.10  : No significant shift — model is stable
  0.10–0.20   : Minor shift — monitor closely
  PSI > 0.20  : Major shift — investigate, consider retraining

Usage
─────
  # Fit on reference (training / dev) data
  monitor = DriftMonitor(n_bins=10)
  monitor.fit(reference_features, reference_pipeline_result)

  # Score each subsequent monthly run
  report = monitor.score(current_features, current_pipeline_result)

  # Summary
  print(report.summary())
  print(report.alerts())           # features / metrics with PSI > 0.20
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# PSI alert thresholds
_PSI_STABLE  = 0.10
_PSI_MONITOR = 0.20   # above this → alert


# ── PSI helpers ───────────────────────────────────────────────────────────────

def _psi_continuous(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    Population Stability Index for a continuous variable.

    Bins are determined from the reference distribution.
    PSI = Σ (actual% − expected%) × ln(actual% / expected%)
    """
    reference = reference[~np.isnan(reference)]
    current   = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    # Use reference quantiles as bin edges
    quantiles = np.linspace(0, 100, n_bins + 1)
    breakpoints = np.unique(np.percentile(reference, quantiles))

    if len(breakpoints) < 2:
        return 0.0

    # Bin both distributions using reference breakpoints
    ref_counts  = np.histogram(reference, bins=breakpoints)[0].astype(float)
    curr_counts = np.histogram(current,   bins=breakpoints)[0].astype(float)

    ref_pct  = (ref_counts  + eps) / (ref_counts.sum()  + eps * len(ref_counts))
    curr_pct = (curr_counts + eps) / (curr_counts.sum() + eps * len(curr_counts))

    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(psi)


def _psi_categorical(
    reference: pd.Series,
    current: pd.Series,
    eps: float = 1e-6,
) -> float:
    """
    PSI for a categorical variable.

    Uses all categories seen in the reference distribution.
    """
    all_cats = reference.unique()
    ref_total  = len(reference)
    curr_total = len(current)

    if ref_total == 0 or curr_total == 0:
        return np.nan

    psi = 0.0
    for cat in all_cats:
        ref_pct  = (reference == cat).sum() / ref_total + eps
        curr_pct = (current   == cat).sum() / curr_total + eps
        psi += (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)

    return float(psi)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class DriftReport:
    """
    Structured result from DriftMonitor.score().

    Attributes
    ----------
    feature_psi   : PSI per numeric feature.
    persona_psi   : PSI for the persona distribution (categorical).
    income_psi    : PSI for the income estimate distribution.
    bci_psi       : PSI for the BCI score distribution.
    decision_psi  : PSI for the final_decision distribution.
    band_psi      : PSI for the BCI band distribution.
    run_date      : Timestamp of this scoring run.
    n_reference   : Number of customers in reference set.
    n_current     : Number of customers in current run.
    """
    feature_psi:    pd.Series
    persona_psi:    float
    income_psi:     float
    bci_psi:        float
    decision_psi:   float
    band_psi:       float
    run_date:       str
    n_reference:    int
    n_current:      int
    metadata:       Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """
        Full PSI summary across all monitored dimensions.

        Returns DataFrame sorted by PSI descending with status flags.
        """
        rows = []

        # Per-feature PSI
        for feat, psi in self.feature_psi.items():
            rows.append({
                "dimension": "feature",
                "name":      feat,
                "psi":       round(psi, 4) if not np.isnan(psi) else np.nan,
                "status":    self._status(psi),
            })

        # Aggregate dimensions
        for name, psi in [
            ("persona_distribution", self.persona_psi),
            ("income_estimate",      self.income_psi),
            ("bci_score",            self.bci_psi),
            ("final_decision",       self.decision_psi),
            ("bci_band",             self.band_psi),
        ]:
            rows.append({
                "dimension": "model_output",
                "name":      name,
                "psi":       round(psi, 4) if not np.isnan(psi) else np.nan,
                "status":    self._status(psi),
            })

        return (
            pd.DataFrame(rows)
            .sort_values("psi", ascending=False)
            .reset_index(drop=True)
        )

    def alerts(self) -> pd.DataFrame:
        """Return only dimensions with PSI ≥ 0.20 (major shift — action required)."""
        return self.summary().query("psi >= @_PSI_MONITOR").reset_index(drop=True)

    def feature_alerts(self, top_n: int = 20) -> pd.DataFrame:
        """Top N most drifted features."""
        feat = self.summary().query("dimension == 'feature'")
        return feat.head(top_n).reset_index(drop=True)

    def is_healthy(self) -> bool:
        """True if no monitored dimension exceeds the alert threshold."""
        return len(self.alerts()) == 0

    @staticmethod
    def _status(psi: float) -> str:
        if np.isnan(psi):
            return "UNKNOWN"
        if psi < _PSI_STABLE:
            return "STABLE"
        if psi < _PSI_MONITOR:
            return "MONITOR"
        return "ALERT"


# ── DriftMonitor ──────────────────────────────────────────────────────────────

class DriftMonitor:
    """
    PSI-based drift monitor for the income estimation framework.

    Parameters
    ----------
    n_bins : int
        Number of quantile bins for continuous PSI. Default 10.
    feature_cols : list, optional
        Specific feature columns to monitor. If None, all numeric features.
    max_features : int
        Cap on number of features monitored (for performance). Default 60.
    """

    def __init__(
        self,
        n_bins: int = 10,
        feature_cols: Optional[List[str]] = None,
        max_features: int = 60,
    ):
        self.n_bins       = n_bins
        self.feature_cols = feature_cols
        self.max_features = max_features

        # Reference state (set by fit())
        self._ref_features:  Optional[pd.DataFrame] = None
        self._ref_personas:  Optional[pd.Series]    = None
        self._ref_income:    Optional[np.ndarray]   = None
        self._ref_bci:       Optional[np.ndarray]   = None
        self._ref_decision:  Optional[pd.Series]    = None
        self._ref_band:      Optional[pd.Series]    = None
        self._monitor_cols:  List[str] = []

        self.fitted_ = False

    def fit(
        self,
        reference_features: pd.DataFrame,
        reference_pipeline_result,
    ) -> "DriftMonitor":
        """
        Store reference distributions from training / dev population.

        Parameters
        ----------
        reference_features : pd.DataFrame
            Feature matrix (FeatureEngineer output) for the reference period.
        reference_pipeline_result : InferencePipelineResult
            Result from InferencePipeline.run() on the reference population.
        """
        from datetime import datetime

        logger.info(
            f"DriftMonitor.fit(): {len(reference_features):,} reference customers"
        )

        # Select numeric feature columns to monitor
        num_cols = reference_features.select_dtypes(include=[np.number]).columns.tolist()
        if self.feature_cols:
            num_cols = [c for c in self.feature_cols if c in reference_features.columns]
        # Cap for performance — prioritise highest-variance features
        if len(num_cols) > self.max_features:
            variances = reference_features[num_cols].var().sort_values(ascending=False)
            num_cols = variances.head(self.max_features).index.tolist()

        self._monitor_cols = num_cols
        self._ref_features = reference_features[num_cols].copy()

        # Model output references
        fo = reference_pipeline_result.final_output
        seg = reference_pipeline_result.segmentation_result

        self._ref_personas = seg["persona"].copy()
        self._ref_income   = fo["income_estimate_raw"].dropna().values
        self._ref_bci      = fo["bci_score"].dropna().values
        self._ref_decision = fo["final_decision"].copy()
        self._ref_band     = fo["bci_band"].copy()
        self._ref_date     = datetime.utcnow().isoformat()

        self.fitted_ = True
        logger.info(
            f"DriftMonitor fitted: {len(self._monitor_cols)} feature columns, "
            f"{len(self._ref_personas)} reference customers"
        )
        return self

    def score(
        self,
        current_features: pd.DataFrame,
        current_pipeline_result,
    ) -> DriftReport:
        """
        Compute PSI between reference and current population.

        Parameters
        ----------
        current_features : pd.DataFrame
            Feature matrix for the current scoring period.
        current_pipeline_result : InferencePipelineResult
            Result from InferencePipeline.run() on the current population.

        Returns
        -------
        DriftReport
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before score().")

        from datetime import datetime

        logger.info(
            f"DriftMonitor.score(): {len(current_features):,} current customers"
        )

        # ── Feature-level PSI ─────────────────────────────────────────────────
        feature_psi = {}
        for col in self._monitor_cols:
            if col not in current_features.columns:
                feature_psi[col] = np.nan
                continue
            ref_arr  = self._ref_features[col].dropna().values
            curr_arr = current_features[col].dropna().values
            feature_psi[col] = _psi_continuous(ref_arr, curr_arr, self.n_bins)

        feature_psi_series = pd.Series(feature_psi).sort_values(ascending=False)

        # ── Output-level PSI ─────────────────────────────────────────────────
        fo  = current_pipeline_result.final_output
        seg = current_pipeline_result.segmentation_result

        persona_psi  = _psi_categorical(self._ref_personas, seg["persona"])
        income_psi   = _psi_continuous(
            self._ref_income, fo["income_estimate_raw"].dropna().values, self.n_bins
        )
        bci_psi      = _psi_continuous(
            self._ref_bci, fo["bci_score"].dropna().values, self.n_bins
        )
        decision_psi = _psi_categorical(self._ref_decision, fo["final_decision"])
        band_psi     = _psi_categorical(self._ref_band, fo["bci_band"])

        report = DriftReport(
            feature_psi=feature_psi_series,
            persona_psi=persona_psi,
            income_psi=income_psi,
            bci_psi=bci_psi,
            decision_psi=decision_psi,
            band_psi=band_psi,
            run_date=datetime.utcnow().isoformat(),
            n_reference=len(self._ref_features),
            n_current=len(current_features),
        )

        n_alerts = len(report.alerts())
        logger.info(
            f"DriftMonitor.score() complete — "
            f"{n_alerts} alert(s)  "
            f"persona_psi={persona_psi:.3f}  "
            f"income_psi={income_psi:.3f}  "
            f"bci_psi={bci_psi:.3f}"
        )
        return report

    def top_drifted_features(
        self, report: DriftReport, top_n: int = 10
    ) -> pd.DataFrame:
        """Convenience: return top N drifted features with status."""
        return report.feature_alerts(top_n)
