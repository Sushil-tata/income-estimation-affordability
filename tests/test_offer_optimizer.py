"""
Tests for Layer 9 — Offer / Action Optimization
  9A  CapacityEngine
  9B  ConfidenceEngine
  9C  OfferSelector
      ActionOptimizer (orchestrator)

Coverage:
  - Annuity factor formula (zero rate, positive rate)
  - CapacityEngine: correct column set, zero capacity guard, clipping
  - ConfidenceEngine: monotonicity, THIN/SPARSE hard zero, primary vs fallback paths
  - OfferSelector: THIN/SPARSE → REFER, zero capacity → REFER, candidate ranking,
      STP / MANUAL_REVIEW action codes, card line inclusion logic, EMI feasibility gate
  - ActionOptimizer: end-to-end schema, no upstream field mutation, from_config()
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from offer.capacity_engine import CapacityEngine
from offer.confidence_engine import ConfidenceEngine
from offer.offer_selector import OfferSelector, ActionCode, VerificationIntensity
from offer.action_optimizer import ActionOptimizer


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_row(**kwargs) -> dict:
    defaults = {
        "adjusted_income":      50_000.0,
        "adsc":                 10_000.0,
        "dscr_used":            0.40,
        "existing_obligations": 10_000.0,
        "persona":              "PAYROLL",
        "p_reliable10":         0.80,
        "p_over10":             0.05,
        "model_confidence":     0.90,
        "income_interval_width": 5_000.0,
        "income_estimate_raw":  50_000.0,
    }
    defaults.update(kwargs)
    return defaults


def _make_df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.index = [f"C{i:04d}" for i in range(len(df))]
    df.index.name = "customer_id"
    return df


def _default_df(**kwargs) -> pd.DataFrame:
    return _make_df([_make_row(**kwargs)])


# ── 9A: CapacityEngine ─────────────────────────────────────────────────────────

class TestAnnuityFactor:
    def test_zero_rate_equals_tenor(self):
        af = CapacityEngine._annuity_factor(0.0, 12)
        assert af == 12.0

    def test_near_zero_rate(self):
        af = CapacityEngine._annuity_factor(1e-10, 12)
        assert abs(af - 12.0) < 0.01

    def test_standard_rate(self):
        # 18% pa, 12 months → PV of annuity-certain ≈ 10.91
        af = CapacityEngine._annuity_factor(0.18 / 12, 12)
        assert 10.8 < af < 11.0

    def test_longer_tenor_bigger_factor(self):
        r = 0.18 / 12
        af12 = CapacityEngine._annuity_factor(r, 12)
        af60 = CapacityEngine._annuity_factor(r, 60)
        assert af60 > af12


class TestCapacityEngine:
    def setup_method(self):
        self.eng = CapacityEngine()

    def test_output_columns(self):
        df = _default_df()
        out = self.eng.compute(df)
        expected = {"max_obligation", "residual_capacity", "max_emi", "max_card_line"}
        for tenor in self.eng.product_tenors_months:
            expected.add(f"max_loan_{tenor}m")
        assert expected.issubset(set(out.columns))

    def test_index_preserved(self):
        df = _default_df()
        out = self.eng.compute(df)
        assert list(out.index) == list(df.index)

    def test_positive_capacity(self):
        df = _default_df(adsc=10_000.0)
        out = self.eng.compute(df)
        assert out["max_emi"].iloc[0] == 10_000.0

    def test_negative_adsc_clipped_to_zero(self):
        df = _default_df(adsc=-5_000.0)
        out = self.eng.compute(df)
        assert out["max_emi"].iloc[0] == 0.0
        assert out["residual_capacity"].iloc[0] == 0.0

    def test_zero_adsc_zero_loans(self):
        df = _default_df(adsc=0.0)
        out = self.eng.compute(df)
        for tenor in self.eng.product_tenors_months:
            assert out[f"max_loan_{tenor}m"].iloc[0] == 0.0
        assert out["max_card_line"].iloc[0] == 0.0

    def test_longer_tenor_larger_loan(self):
        df = _default_df(adsc=5_000.0)
        out = self.eng.compute(df)
        loan12 = out["max_loan_12m"].iloc[0]
        loan60 = out["max_loan_60m"].iloc[0]
        assert loan60 > loan12

    def test_card_line_formula(self):
        eng = CapacityEngine(card_line_months_equivalent=24)
        df = _default_df(adsc=10_000.0)
        out = eng.compute(df)
        assert out["max_card_line"].iloc[0] == 10_000.0 * 24

    def test_missing_column_uses_default(self):
        # Drop adsc — engine should use 0.0 default
        df = _default_df()
        df = df.drop(columns=["adsc"])
        out = self.eng.compute(df)
        assert out["max_emi"].iloc[0] == 0.0

    def test_no_upstream_modification(self):
        df = _default_df()
        original_income = df["adjusted_income"].iloc[0]
        self.eng.compute(df)
        assert df["adjusted_income"].iloc[0] == original_income

    def test_from_config(self):
        cfg = {"offer_optimization": {
            "product_tenors_months": [12, 24],
            "reference_rate_annual": 0.20,
            "card_line_months_equivalent": 24,
        }}
        eng = CapacityEngine.from_config(cfg)
        assert eng.product_tenors_months == [12, 24]
        assert eng.reference_rate_annual == 0.20
        assert eng.card_line_months_equivalent == 24


# ── 9B: ConfidenceEngine ──────────────────────────────────────────────────────

class TestConfidenceEngine:
    def setup_method(self):
        self.eng = ConfidenceEngine()

    def _compute(self, **kwargs) -> pd.DataFrame:
        return self.eng.compute(_default_df(**kwargs))

    def test_output_columns(self):
        out = self._compute()
        assert {"reliability_score", "confidence_multiplier", "confidence_reason_code"}.issubset(
            set(out.columns)
        )

    def test_thin_zero_multiplier(self):
        out = self._compute(persona="THIN")
        assert out["confidence_multiplier"].iloc[0] == 0.0

    def test_sparse_zero_multiplier(self):
        out = self._compute(persona="SPARSE")
        assert out["confidence_multiplier"].iloc[0] == 0.0

    def test_thin_reason_code(self):
        out = self._compute(persona="THIN")
        assert out["confidence_reason_code"].iloc[0] == "ZERO_PERSONA_BASE"

    def test_payroll_high_reliability_near_1(self):
        out = self._compute(persona="PAYROLL", p_reliable10=0.95, p_over10=0.01)
        mult = out["confidence_multiplier"].iloc[0]
        assert mult > 0.90

    def test_monotone_higher_reliability_higher_mult(self):
        """Higher p_reliable10 → higher multiplier (monotonicity check)."""
        out_low  = self._compute(persona="L1", p_reliable10=0.40, p_over10=0.30)
        out_high = self._compute(persona="L1", p_reliable10=0.90, p_over10=0.02)
        assert out_high["confidence_multiplier"].iloc[0] > out_low["confidence_multiplier"].iloc[0]

    def test_monotone_higher_over10_lower_mult(self):
        """Higher p_over10 → lower multiplier."""
        out_low  = self._compute(persona="L1", p_reliable10=0.70, p_over10=0.05)
        out_high = self._compute(persona="L1", p_reliable10=0.70, p_over10=0.50)
        assert out_high["confidence_multiplier"].iloc[0] < out_low["confidence_multiplier"].iloc[0]

    def test_multiplier_bounded_0_1(self):
        """Multiplier must be in [0, 1]."""
        out = self._compute(persona="PAYROLL", p_reliable10=1.0, p_over10=0.0)
        mult = out["confidence_multiplier"].iloc[0]
        assert 0.0 <= mult <= 1.0

    def test_fallback_path_no_primary_signals(self):
        df = _default_df(persona="L0")
        df = df.drop(columns=["p_reliable10", "p_over10"])
        out = self.eng.compute(df)
        assert out["confidence_reason_code"].iloc[0] == "FALLBACK_PROXY"
        assert 0.0 <= out["confidence_multiplier"].iloc[0] <= 1.0

    def test_primary_path_reason_code(self):
        out = self._compute(persona="PAYROLL", p_reliable10=0.80, p_over10=0.05)
        assert out["confidence_reason_code"].iloc[0] == "PRIMARY_RELIABILITY"

    def test_min_multiplier_floor(self):
        """Multiplier should not drop below min_multiplier for non-THIN/SPARSE."""
        eng = ConfidenceEngine(min_multiplier=0.40)
        out = eng.compute(_default_df(persona="L2", p_reliable10=0.01, p_over10=0.99))
        # Even with worst reliability, L2 base is 0.70; floor = 0.40
        assert out["confidence_multiplier"].iloc[0] >= 0.40

    def test_from_config(self):
        cfg = {"offer_optimization": {
            "reliability_weight_p_reliable": 0.70,
            "reliability_weight_p_over": 0.30,
            "min_multiplier": 0.25,
        }}
        eng = ConfidenceEngine.from_config(cfg)
        assert eng.reliability_weight_p_reliable == 0.70
        assert eng.min_multiplier == 0.25

    def test_batch_no_nan(self):
        rows = [_make_row(persona=p) for p in ["PAYROLL", "L0", "L1", "L2", "PT", "THIN", "SPARSE"]]
        df = _make_df(rows)
        out = self.eng.compute(df)
        assert out["confidence_multiplier"].notna().all()
        assert out["reliability_score"].notna().all()


# ── 9C: OfferSelector ─────────────────────────────────────────────────────────

class TestOfferSelector:
    def setup_method(self):
        self.cap_eng  = CapacityEngine()
        self.conf_eng = ConfidenceEngine()
        self.sel      = OfferSelector()

    def _run(self, **kwargs):
        df   = _default_df(**kwargs)
        cap  = self.cap_eng.compute(df)
        conf = self.conf_eng.compute(df)
        return self.sel.select(cap, conf, df), df

    def test_output_columns(self):
        out, _ = self._run()
        required = {
            "action_code", "verification_intensity",
            "offer_amount_recommended", "recommended_tenor_months",
            "max_offerable_amount", "include_card_line",
            "card_line_recommended", "stp_flag",
            "optimization_reason_codes",
        }
        assert required.issubset(set(out.columns))

    def test_thin_always_refer(self):
        out, _ = self._run(persona="THIN", adsc=100_000.0)
        assert out["action_code"].iloc[0] == ActionCode.REFER.value
        assert out["stp_flag"].iloc[0] == False

    def test_sparse_always_refer(self):
        out, _ = self._run(persona="SPARSE", adsc=100_000.0)
        assert out["action_code"].iloc[0] == ActionCode.REFER.value

    def test_zero_capacity_refer(self):
        out, _ = self._run(persona="PAYROLL", adsc=0.0)
        assert out["action_code"].iloc[0] == ActionCode.REFER.value

    def test_payroll_positive_capacity_stp(self):
        out, _ = self._run(
            persona="PAYROLL", adsc=15_000.0,
            p_reliable10=0.90, p_over10=0.02,
        )
        assert out["action_code"].iloc[0] == ActionCode.STP_APPROVE.value
        assert out["stp_flag"].iloc[0] == True

    def test_l2_gets_manual_review(self):
        out, _ = self._run(persona="L2", adsc=15_000.0)
        assert out["action_code"].iloc[0] == ActionCode.MANUAL_REVIEW.value
        assert out["stp_flag"].iloc[0] == False

    def test_pt_gets_manual_review(self):
        out, _ = self._run(persona="PT", adsc=15_000.0)
        assert out["action_code"].iloc[0] == ActionCode.MANUAL_REVIEW.value

    def test_offer_amount_positive_when_approved(self):
        out, _ = self._run(persona="PAYROLL", adsc=15_000.0)
        assert out["offer_amount_recommended"].iloc[0] > 0

    def test_offer_amount_zero_when_referred(self):
        out, _ = self._run(persona="THIN", adsc=15_000.0)
        assert out["offer_amount_recommended"].iloc[0] == 0.0

    def test_recommended_tenor_valid(self):
        out, _ = self._run(persona="PAYROLL", adsc=15_000.0)
        tenor = out["recommended_tenor_months"].iloc[0]
        assert tenor in self.sel.product_tenors_months

    def test_card_line_payroll_included(self):
        out, _ = self._run(
            persona="PAYROLL", adsc=15_000.0,
            p_reliable10=0.90, p_over10=0.02,
        )
        assert out["include_card_line"].iloc[0] == True
        assert out["card_line_recommended"].iloc[0] > 0

    def test_card_line_l2_not_included(self):
        out, _ = self._run(persona="L2", adsc=15_000.0)
        assert out["include_card_line"].iloc[0] == False

    def test_reason_codes_list(self):
        out, _ = self._run(persona="PAYROLL", adsc=15_000.0)
        codes = out["optimization_reason_codes"].iloc[0]
        assert isinstance(codes, list)
        assert len(codes) >= 1

    def test_refer_reason_code_populated(self):
        out, _ = self._run(persona="THIN")
        codes = out["optimization_reason_codes"].iloc[0]
        assert any("INELIGIBLE" in c for c in codes)

    def test_no_upstream_mutation(self):
        df = _default_df(persona="PAYROLL", adsc=15_000.0)
        original_adsc = df["adsc"].iloc[0]
        cap  = self.cap_eng.compute(df)
        conf = self.conf_eng.compute(df)
        self.sel.select(cap, conf, df)
        assert df["adsc"].iloc[0] == original_adsc

    def test_below_min_loan_amount_refer(self):
        """Tiny adsc → offer below min_loan_amount → REFER."""
        sel = OfferSelector(min_loan_amount_thb=1_000_000.0)
        df  = _default_df(persona="PAYROLL", adsc=100.0)
        cap  = self.cap_eng.compute(df)
        conf = self.conf_eng.compute(df)
        out  = sel.select(cap, conf, df)
        assert out["action_code"].iloc[0] == ActionCode.REFER.value

    def test_from_config(self):
        cfg = {"offer_optimization": {
            "product_tenors_months": [12, 24],
            "min_loan_amount_thb": 20_000.0,
            "stp_min_emi_thb": 2_000.0,
            "min_card_line_thb": 10_000.0,
            "reference_rate_annual": 0.20,
        }}
        sel = OfferSelector.from_config(cfg)
        assert sel.product_tenors_months == [12, 24]
        assert sel.min_loan_amount_thb == 20_000.0


# ── ActionOptimizer (orchestrator) ────────────────────────────────────────────

class TestActionOptimizer:
    def setup_method(self):
        self.opt = ActionOptimizer()

    def test_result_has_all_dataframes(self):
        df  = _default_df()
        res = self.opt.run(df)
        assert not res.capacity.empty
        assert not res.confidence.empty
        assert not res.offers.empty
        assert not res.combined.empty

    def test_combined_index_matches_input(self):
        rows = [_make_row(persona=p) for p in ["PAYROLL", "L0", "L1", "THIN", "SPARSE"]]
        df   = _make_df(rows)
        res  = self.opt.run(df)
        assert list(res.combined.index) == list(df.index)

    def test_summary_counts_sum_to_n(self):
        rows = [_make_row(persona=p) for p in ["PAYROLL", "L2", "THIN"]]
        df   = _make_df(rows)
        res  = self.opt.run(df)
        assert res.n_stp + res.n_manual + res.n_refer == res.n_customers

    def test_no_upstream_column_modified(self):
        df  = _default_df()
        original = df.copy()
        self.opt.run(df)
        pd.testing.assert_frame_equal(df, original)

    def test_thin_sparse_counted_as_refer(self):
        rows = [_make_row(persona="THIN"), _make_row(persona="SPARSE")]
        df   = _make_df(rows)
        res  = self.opt.run(df)
        assert res.n_refer == 2
        assert res.n_stp == 0

    def test_from_config_smoke(self):
        cfg = {
            "offer_optimization": {
                "product_tenors_months": [12, 24, 36],
                "reference_rate_annual": 0.18,
                "card_line_months_equivalent": 30,
                "min_loan_amount_thb": 10_000.0,
                "stp_min_emi_thb": 1_000.0,
                "min_card_line_thb": 5_000.0,
                "min_multiplier": 0.30,
                "reliability_alpha": 1.0,
            }
        }
        opt = ActionOptimizer.from_config(cfg)
        df  = _default_df()
        res = opt.run(df)
        assert res.n_customers == 1

    def test_combined_contains_all_layer_columns(self):
        df  = _default_df()
        res = self.opt.run(df)
        # Spot-check one column from each layer
        assert "max_emi"                in res.combined.columns   # 9A
        assert "confidence_multiplier"  in res.combined.columns   # 9B
        assert "action_code"            in res.combined.columns   # 9C

    def test_large_batch(self):
        """Smoke test on 500-row batch."""
        personas = ["PAYROLL", "L0", "L1", "L2", "PT", "THIN", "SPARSE"]
        rows = [
            _make_row(
                persona=personas[i % len(personas)],
                adsc=float(np.random.randint(0, 50_000)),
                p_reliable10=float(np.random.uniform(0.4, 0.95)),
                p_over10=float(np.random.uniform(0.01, 0.30)),
            )
            for i in range(500)
        ]
        df  = _make_df(rows)
        res = self.opt.run(df)
        assert res.n_customers == 500
        assert res.n_stp + res.n_manual + res.n_refer == 500
        assert res.combined["confidence_multiplier"].notna().all()
        assert res.combined["offer_amount_recommended"].notna().all()


# ── Monotonicity invariant ────────────────────────────────────────────────────

class TestMonotonicity:
    """
    Verify the end-to-end monotonicity guarantee:
    higher income reliability → equal or higher offer amount.
    """

    def test_end_to_end_higher_reliability_higher_offer(self):
        opt = ActionOptimizer()

        low_rel  = _default_df(persona="L1", adsc=15_000.0, p_reliable10=0.40, p_over10=0.40)
        high_rel = _default_df(persona="L1", adsc=15_000.0, p_reliable10=0.90, p_over10=0.02)

        res_low  = opt.run(low_rel)
        res_high = opt.run(high_rel)

        offer_low  = res_low.combined["offer_amount_recommended"].iloc[0]
        offer_high = res_high.combined["offer_amount_recommended"].iloc[0]

        assert offer_high >= offer_low, (
            f"Monotonicity violated: high_rel offer {offer_high} < low_rel offer {offer_low}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
