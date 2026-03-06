"""
Persona Stability Smoother
───────────────────────────
Prevents persona churn — rapid month-to-month switching between
L0 / L1 / L2 that would make downstream credit limits volatile.

Design
──────
  Stateless exponential smoothing of persona probabilities:

    smoothed_probs_t = α × new_probs_t + (1 − α) × smoothed_probs_{t-1}

  Switch persona only if:
    P(new_persona) − P(current_persona) > stability_switch_delta

  This ensures:
    • Customers near a persona boundary don't flip every month.
    • Large, genuine income structure changes (e.g. job loss → gig) are
      captured within 2–3 months.
    • Target: < 5% of customers switch persona in any given month.

Stateless design
────────────────
  The smoother does NOT maintain internal state between calls.
  It expects the caller to persist smoothed_probs from the previous
  scoring run and pass it in at the next run.  This keeps the class
  simple, testable, and compatible with both batch and streaming scoring.

Parameters (from config.yaml → advanced_modeling.persona_stability)
────────────────────────────────────────────────────────────────────
  smoothing_alpha     : 0.50   (weight on new observation)
  switch_delta        : 0.15   (min probability margin to accept a switch)
  target_switch_rate  : 0.05   (monitoring target, not enforced)

Usage
─────
  smoother = PersonaStabilitySmoother()

  # First run (no history)
  result = smoother.smooth(new_probs_df, current_personas=None, prev_smoothed_probs=None)

  # Subsequent runs (pass in previous smoothed probs + current persona)
  result = smoother.smooth(new_probs_df,
                           current_personas=prev_result["persona"],
                           prev_smoothed_probs=prev_result[["L0_prob","L1_prob","L2_prob"]])

  Returns a DataFrame with:
    persona          — stable persona label
    L0_prob, L1_prob, L2_prob — smoothed probabilities
    persona_switched — bool: True if persona changed this run
"""

import numpy as np
import pandas as pd
import logging
import yaml
from typing import Optional

logger = logging.getLogger(__name__)

_PERSONA_COLS: list = ["L0_prob", "L1_prob", "L2_prob"]
_PERSONAS: list = ["L0", "L1", "L2"]


class PersonaStabilitySmoother:
    """
    Stateless EMA smoother for persona probabilities.

    Parameters
    ----------
    smoothing_alpha : float
        Weight on new observation (0–1). Higher → reacts faster to changes.
        Default 0.50 (equal weight old/new).
    switch_delta : float
        Minimum probability margin P(new) − P(current) to accept a persona switch.
        Default 0.15.
    config_path : str, optional
        Path to config.yaml. If provided, overrides smoothing_alpha and switch_delta
        with values from advanced_modeling.persona_stability.
    """

    def __init__(
        self,
        smoothing_alpha: float = 0.50,
        switch_delta: float = 0.15,
        config_path: Optional[str] = None,
    ):
        if config_path is not None:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            stability_cfg = cfg.get("advanced_modeling", {}).get("persona_stability", {})
            self.smoothing_alpha = stability_cfg.get("smoothing_alpha", smoothing_alpha)
            self.switch_delta    = stability_cfg.get("switch_delta", switch_delta)
            self.target_switch_rate = stability_cfg.get("target_switch_rate", 0.05)
        else:
            self.smoothing_alpha = smoothing_alpha
            self.switch_delta    = switch_delta
            self.target_switch_rate = 0.05

    def smooth(
        self,
        new_probs: pd.DataFrame,
        current_personas: Optional[pd.Series] = None,
        prev_smoothed_probs: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Apply EMA smoothing and return stable persona assignments.

        Parameters
        ----------
        new_probs : pd.DataFrame
            Fresh persona probabilities from PersonaRouter for this run.
            Columns: L0_prob, L1_prob, L2_prob.
            Index: customer IDs.
            Rows for PAYROLL / THIN customers should be excluded before calling;
            those personas bypass the smoother.
        current_personas : pd.Series, optional
            Currently assigned personas (from previous scoring run).
            If None, raw argmax of new_probs is used (first-run behaviour).
        prev_smoothed_probs : pd.DataFrame, optional
            Smoothed probabilities from the previous run (L0_prob, L1_prob, L2_prob).
            If None, new_probs are used as-is (cold start).

        Returns
        -------
        pd.DataFrame with columns:
          persona          : str — stable assigned persona (L0 / L1 / L2)
          L0_prob          : float — smoothed probability
          L1_prob          : float — smoothed probability
          L2_prob          : float — smoothed probability
          persona_switched : bool  — True if persona changed vs current_personas
        """
        idx = new_probs.index
        n = len(idx)

        # ── Validate & extract new probs ────────────────────────────────────
        missing = [c for c in _PERSONA_COLS if c not in new_probs.columns]
        if missing:
            raise ValueError(f"PersonaStabilitySmoother: missing columns {missing}")

        P_new = new_probs[_PERSONA_COLS].reindex(idx).fillna(1 / 3).values  # (n, 3)

        # ── EMA smoothing ────────────────────────────────────────────────────
        if prev_smoothed_probs is not None:
            P_prev = prev_smoothed_probs[_PERSONA_COLS].reindex(idx).fillna(1 / 3).values
            P_smooth = self.smoothing_alpha * P_new + (1 - self.smoothing_alpha) * P_prev
        else:
            # Cold start: no history → use new probs directly
            P_smooth = P_new.copy()

        # Renormalise rows (guard against floating-point drift)
        row_sums = P_smooth.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        P_smooth = P_smooth / row_sums

        # ── Persona switching logic ──────────────────────────────────────────
        # argmax of smoothed probs → candidate new persona
        candidate_idx = np.argmax(P_smooth, axis=1)  # (n,)
        candidates = [_PERSONAS[i] for i in candidate_idx]

        if current_personas is not None:
            # Map current persona to column index
            current_arr = current_personas.reindex(idx).values
            stable_personas = list(current_arr)

            for i, cust_idx in enumerate(idx):
                curr = current_arr[i]
                cand = candidates[i]

                if curr == cand:
                    # No change proposed
                    continue

                if curr not in _PERSONAS:
                    # Current persona is PAYROLL/THIN/unknown — accept new assignment
                    stable_personas[i] = cand
                    continue

                curr_col = _PERSONAS.index(curr)
                # Switch only if smoothed P(candidate) exceeds P(current) by switch_delta
                delta = P_smooth[i, candidate_idx[i]] - P_smooth[i, curr_col]
                if delta >= self.switch_delta:
                    stable_personas[i] = cand
                # else keep current persona despite different argmax
        else:
            # First run: no current persona to protect
            stable_personas = candidates

        stable_personas = pd.Series(stable_personas, index=idx, name="persona")

        # ── Build output ─────────────────────────────────────────────────────
        switched = pd.Series(False, index=idx, name="persona_switched")
        if current_personas is not None:
            curr_aligned = current_personas.reindex(idx)
            switched = (stable_personas != curr_aligned).rename("persona_switched")

        result = pd.DataFrame(
            P_smooth,
            index=idx,
            columns=_PERSONA_COLS,
        )
        result["persona"] = stable_personas.values
        result["persona_switched"] = switched.values

        # ── Monitoring log ───────────────────────────────────────────────────
        switch_rate = switched.mean()
        logger.info(
            f"PersonaStabilitySmoother: {n} customers  "
            f"switch_rate={switch_rate:.3f}  "
            f"target={self.target_switch_rate:.2f}  "
            f"{'⚠ above target' if switch_rate > self.target_switch_rate else 'OK'}"
        )
        dist = stable_personas.value_counts()
        for p in _PERSONAS:
            cnt = dist.get(p, 0)
            logger.info(f"  {p}: {cnt:,}  ({cnt / n * 100:.1f}%)")

        return result

    def switch_rate(self, result: pd.DataFrame) -> float:
        """Fraction of customers who switched persona this run."""
        return result["persona_switched"].mean()

    def stability_report(
        self,
        result: pd.DataFrame,
        prev_personas: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Summarise stability metrics per persona.

        Returns a DataFrame indexed by persona with columns:
          count, pct, switched_in, switched_out, avg_confidence
        """
        n = len(result)
        rows = []
        for p in _PERSONAS:
            mask_now = result["persona"] == p
            cnt = mask_now.sum()
            avg_conf = result.loc[mask_now, f"{p}_prob"].mean() if cnt > 0 else np.nan

            switched_out = (
                (prev_personas == p) & result["persona_switched"]
            ).sum() if prev_personas is not None else np.nan

            switched_in = (
                (result["persona"] == p) & result["persona_switched"]
            ).sum() if prev_personas is not None else np.nan

            rows.append({
                "persona":     p,
                "count":       cnt,
                "pct":         round(cnt / n * 100, 2),
                "switched_in":  switched_in,
                "switched_out": switched_out,
                "avg_smoothed_prob": round(avg_conf, 3) if not np.isnan(avg_conf) else None,
            })
        return pd.DataFrame(rows)
