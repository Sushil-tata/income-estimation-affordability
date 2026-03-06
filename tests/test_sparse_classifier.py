"""
Tests for SparseClassifier (src/segmentation/sparse_classifier.py)

Coverage:
  - OQS computation
  - Feature masking count (deterministic from Feature Validity Matrix)
  - Bootstrap proxy label derivation
  - Fit / predict cycle (smoke test with synthetic data)
  - predict_full() output schema completeness
  - SPARSE reason code assignment
  - Clearly-SPARSE cases are classified as SPARSE
  - Clearly-estimable cases are classified as NOT SPARSE
  - from_config() construction
  - save() / load() persistence
"""

import os
import sys
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
import pytest

# Allow running from project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from segmentation.sparse_classifier import SparseClassifier


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_feature_row(**kwargs) -> dict:
    """Return a single-customer feature dict with sensible defaults."""
    defaults = {
        "months_data_available":          6,
        "transaction_count_avg_monthly":  15.0,
        "months_with_zero_credit":        0,
        "dormancy_gap_max":               0,
        "credit_cv_6m":                   0.15,
        "avg_eom_balance_3m":             20_000.0,
        "balance_volatility_6m":          0.20,
        "months_below_1000_balance":      0,
        "pass_through_score":             0.20,
        "retention_ratio_6m":             0.45,
        "has_payroll_credit":             0,
    }
    defaults.update(kwargs)
    return defaults


def _make_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.index = [f"CUST_{i:04d}" for i in range(len(df))]
    df.index.name = "customer_id"
    return df


def _fitted_classifier(n_non_sparse: int = 200, n_sparse: int = 50) -> SparseClassifier:
    """Return a fitted SparseClassifier on synthetic data."""
    rows = []
    # Non-SPARSE: rich data
    for _ in range(n_non_sparse):
        rows.append(_make_feature_row(
            months_data_available=np.random.randint(4, 7),
            transaction_count_avg_monthly=np.random.uniform(10, 30),
            months_with_zero_credit=np.random.randint(0, 2),
        ))
    # SPARSE: poor data
    for _ in range(n_sparse):
        rows.append(_make_feature_row(
            months_data_available=np.random.randint(1, 3),
            transaction_count_avg_monthly=np.random.uniform(0.5, 1.5),
            months_with_zero_credit=np.random.randint(2, 4),
            dormancy_gap_max=np.random.randint(2, 4),
        ))

    df = _make_df(rows)
    clf = SparseClassifier(n_estimators=50, random_state=42)
    clf.fit(df)
    return clf


# ── OQS computation ───────────────────────────────────────────────────────────

class TestOQS:
    def setup_method(self):
        self.clf = SparseClassifier()

    def test_full_depth_full_density(self):
        X = _make_df([_make_feature_row(months_data_available=6, transaction_count_avg_monthly=10)])
        oqs = self.clf.compute_oqs(X)
        assert abs(oqs.iloc[0] - 1.0) < 1e-6

    def test_zero_depth_zero_density(self):
        X = _make_df([_make_feature_row(months_data_available=0, transaction_count_avg_monthly=0)])
        oqs = self.clf.compute_oqs(X)
        assert oqs.iloc[0] == 0.0

    def test_half_depth(self):
        X = _make_df([_make_feature_row(months_data_available=3, transaction_count_avg_monthly=10)])
        oqs = self.clf.compute_oqs(X)
        # depth = 3/6 = 0.5, density = 1.0 → oqs = 0.75
        assert abs(oqs.iloc[0] - 0.75) < 1e-6

    def test_clipped_above_1(self):
        X = _make_df([_make_feature_row(months_data_available=100, transaction_count_avg_monthly=100)])
        oqs = self.clf.compute_oqs(X)
        assert oqs.iloc[0] <= 1.0


# ── Feature masking count ─────────────────────────────────────────────────────

class TestMaskingCount:
    def test_0_months(self):
        m = pd.Series([0])
        assert SparseClassifier.compute_masking_count(m).iloc[0] == 6   # all 6 groups masked

    def test_2_months(self):
        # G1(≥2 OK), G2(≥4 masked), G3(≥2 OK), G4(≥4 masked), G5(≥5 masked), G6(≥3 masked) → 4
        m = pd.Series([2])
        assert SparseClassifier.compute_masking_count(m).iloc[0] == 4

    def test_3_months(self):
        # G2, G4, G5 masked → 3
        m = pd.Series([3])
        assert SparseClassifier.compute_masking_count(m).iloc[0] == 3

    def test_4_months(self):
        # G5 masked → 1
        m = pd.Series([4])
        assert SparseClassifier.compute_masking_count(m).iloc[0] == 1

    def test_5_months(self):
        # all OK → 0
        m = pd.Series([5])
        assert SparseClassifier.compute_masking_count(m).iloc[0] == 0

    def test_6_months(self):
        m = pd.Series([6])
        assert SparseClassifier.compute_masking_count(m).iloc[0] == 0


# ── Bootstrap proxy label derivation ─────────────────────────────────────────

class TestProxyLabels:
    def setup_method(self):
        self.clf = SparseClassifier()

    def _derive(self, **kwargs) -> int:
        X = _make_df([_make_feature_row(**kwargs)])
        X_aug = self.clf._augment(X)
        return int(self.clf._derive_proxy_labels(X_aug).iloc[0])

    def test_insufficient_depth(self):
        # months < 2 → SPARSE
        assert self._derive(months_data_available=1) == 1

    def test_extreme_gap_ratio(self):
        # 4 zero months out of 5 = 0.80 > 0.60 → SPARSE
        assert self._derive(months_data_available=5, months_with_zero_credit=4) == 1

    def test_low_density_short_history(self):
        # tx < 1.5 AND months < 4 → SPARSE
        assert self._derive(
            months_data_available=3,
            transaction_count_avg_monthly=1.0,
        ) == 1

    def test_near_dormant(self):
        # dormancy_gap_max = 5 >= months(6) - 1 = 5 → SPARSE
        assert self._derive(months_data_available=6, dormancy_gap_max=5) == 1

    def test_rich_data_not_sparse(self):
        # Good data → not SPARSE
        assert self._derive(
            months_data_available=6,
            transaction_count_avg_monthly=20,
            months_with_zero_credit=0,
            dormancy_gap_max=0,
        ) == 0

    def test_low_density_but_sufficient_history(self):
        # tx = 1.0 but months = 6 ≥ 4 → rule 3 does NOT trigger
        # (no other rule fires either for this case)
        assert self._derive(
            months_data_available=6,
            transaction_count_avg_monthly=1.0,
            months_with_zero_credit=0,
            dormancy_gap_max=0,
        ) == 0


# ── Fit / predict smoke tests ─────────────────────────────────────────────────

class TestFitPredict:
    def setup_method(self):
        self.clf = _fitted_classifier()

    def test_fitted_flag(self):
        assert self.clf.fitted_

    def test_predict_proba_shape(self):
        X = _make_df([_make_feature_row() for _ in range(10)])
        proba = self.clf.predict_proba(X)
        assert proba.shape == (10, 2)
        assert list(proba.columns) == ["p_not_sparse", "p_sparse"]

    def test_proba_sum_to_1(self):
        X = _make_df([_make_feature_row() for _ in range(20)])
        proba = self.clf.predict_proba(X)
        sums = proba["p_not_sparse"] + proba["p_sparse"]
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_predict_returns_bool(self):
        X = _make_df([_make_feature_row() for _ in range(5)])
        pred = self.clf.predict(X)
        assert pred.dtype == bool

    def test_predict_before_fit_raises(self):
        clf = SparseClassifier()
        X = _make_df([_make_feature_row()])
        with pytest.raises(RuntimeError, match="must be fitted"):
            clf.predict(X)


# ── predict_full output schema ────────────────────────────────────────────────

class TestPredictFull:
    def setup_method(self):
        self.clf = _fitted_classifier()

    def test_required_columns_present(self):
        X = _make_df([_make_feature_row() for _ in range(5)])
        result = self.clf.predict_full(X)
        required = {
            "is_sparse", "p_sparse", "sparse_reason_code",
            "oqs_score", "feature_masking_count",
        }
        assert required.issubset(set(result.columns))

    def test_no_undefined_rows(self):
        X = _make_df([_make_feature_row() for _ in range(20)])
        result = self.clf.predict_full(X)
        # p_sparse is always defined
        assert result["p_sparse"].notna().all()
        # oqs_score is always defined
        assert result["oqs_score"].notna().all()

    def test_sparse_reason_code_none_for_non_sparse(self):
        # A customer with rich data should not be SPARSE and have None reason
        X = _make_df([_make_feature_row(
            months_data_available=6,
            transaction_count_avg_monthly=25,
            months_with_zero_credit=0,
            dormancy_gap_max=0,
            has_payroll_credit=1,
        )])
        result = self.clf.predict_full(X)
        if not result["is_sparse"].iloc[0]:
            assert pd.isna(result["sparse_reason_code"].iloc[0])

    def test_sparse_reason_code_set_for_sparse(self):
        # Force a SPARSE case: 1 month of data
        X = _make_df([_make_feature_row(months_data_available=1)])
        result = self.clf.predict_full(X)
        # Bootstrap proxy labels would classify this as SPARSE; classifier should too
        if result["is_sparse"].iloc[0]:
            assert result["sparse_reason_code"].iloc[0] is not None

    def test_index_preserved(self):
        rows = [_make_feature_row() for _ in range(5)]
        X = _make_df(rows)
        result = self.clf.predict_full(X)
        assert list(result.index) == list(X.index)


# ── SPARSE classification accuracy ────────────────────────────────────────────

class TestClassificationAccuracy:
    def setup_method(self):
        self.clf = _fitted_classifier(n_non_sparse=300, n_sparse=100)

    def test_clearly_sparse_classified_correctly(self):
        """Customers with only 1 month of data, 0 transactions → SPARSE."""
        rows = [
            _make_feature_row(
                months_data_available=1,
                transaction_count_avg_monthly=0.5,
                months_with_zero_credit=1,
                dormancy_gap_max=1,
            )
            for _ in range(10)
        ]
        X = _make_df(rows)
        pred = self.clf.predict(X)
        # At least 8/10 should be classified as SPARSE
        assert pred.sum() >= 8, f"Expected ≥8 SPARSE predictions, got {pred.sum()}"

    def test_clearly_estimable_not_classified_sparse(self):
        """Customers with 6M of data, high activity → NOT SPARSE."""
        rows = [
            _make_feature_row(
                months_data_available=6,
                transaction_count_avg_monthly=25,
                months_with_zero_credit=0,
                dormancy_gap_max=0,
                has_payroll_credit=1,
                credit_cv_6m=0.05,
            )
            for _ in range(10)
        ]
        X = _make_df(rows)
        pred = self.clf.predict(X)
        # At most 2/10 should be classified as SPARSE
        assert pred.sum() <= 2, f"Expected ≤2 SPARSE predictions, got {pred.sum()}"


# ── Reason code tests ─────────────────────────────────────────────────────────

class TestReasonCodes:
    def setup_method(self):
        self.clf = SparseClassifier()

    def _reason(self, **kwargs) -> Optional[str]:
        X = _make_df([_make_feature_row(**kwargs)])
        X_aug = self.clf._augment(X)
        is_sparse = pd.Series([True], index=X_aug.index)
        return self.clf._assign_reason_codes(X_aug, is_sparse).iloc[0]

    def test_insufficient_depth_priority(self):
        # months < 2 → INSUFFICIENT_DEPTH (highest priority)
        reason = self._reason(
            months_data_available=1,
            months_with_zero_credit=1,
            transaction_count_avg_monthly=0.5,
        )
        assert reason == "INSUFFICIENT_DEPTH"

    def test_extreme_gaps(self):
        reason = self._reason(
            months_data_available=5,
            months_with_zero_credit=4,  # gap_ratio = 0.8 > 0.6
        )
        assert reason == "EXTREME_GAPS"

    def test_low_density(self):
        reason = self._reason(
            months_data_available=3,
            transaction_count_avg_monthly=1.0,
            months_with_zero_credit=0,
        )
        assert reason == "LOW_DENSITY"

    def test_non_sparse_gets_none(self):
        X = _make_df([_make_feature_row()])
        X_aug = self.clf._augment(X)
        is_sparse = pd.Series([False], index=X_aug.index)
        reason = self.clf._assign_reason_codes(X_aug, is_sparse).iloc[0]
        assert pd.isna(reason)


# ── from_config construction ──────────────────────────────────────────────────

class TestFromConfig:
    def test_default_config(self):
        clf = SparseClassifier.from_config({})
        assert clf.sparse_threshold == 0.50
        assert clf.n_estimators == 200
        assert clf.calibration_method == "isotonic"

    def test_custom_config(self):
        cfg = {
            "sparse_classifier": {
                "sparse_threshold": 0.40,
                "n_estimators": 100,
                "calibration_method": "sigmoid",
                "oqs_depth_weight": 0.60,
                "oqs_density_weight": 0.40,
            }
        }
        clf = SparseClassifier.from_config(cfg)
        assert clf.sparse_threshold == 0.40
        assert clf.n_estimators == 100
        assert clf.calibration_method == "sigmoid"
        assert clf.oqs_depth_weight == 0.60


# ── Persistence (save / load) ─────────────────────────────────────────────────

class TestPersistence:
    def test_save_load_roundtrip(self):
        clf = _fitted_classifier(n_non_sparse=100, n_sparse=30)
        X = _make_df([_make_feature_row() for _ in range(5)])
        proba_before = clf.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            clf.save(tmpdir)
            clf2 = SparseClassifier.load(tmpdir)

        proba_after = clf2.predict_proba(X)
        pd.testing.assert_frame_equal(proba_before, proba_after)
        assert clf2.fitted_


# ── Verified label override ───────────────────────────────────────────────────

class TestVerifiedLabels:
    def test_fit_with_verified_labels(self):
        """Verify classifier accepts explicit binary labels (target state: P-03)."""
        rows = [_make_feature_row() for _ in range(100)]
        X = _make_df(rows)
        y = pd.Series(
            [1] * 20 + [0] * 80,
            index=X.index,
            name="is_sparse",
        )
        clf = SparseClassifier(n_estimators=50)
        clf.fit(X, y=y)
        assert clf.fitted_
        proba = clf.predict_proba(X)
        assert proba.shape == (100, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
