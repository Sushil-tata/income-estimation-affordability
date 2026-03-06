# Mental Model — Income Estimation & Affordability Framework
## For Data Scientists and Auditors

**Version:** v5.0 | **Audience:** DS team, Model Risk, Internal Audit
**Companion to:** `v5.0_frozen_architecture.md`

---

## The Big Picture in One Paragraph

> We look at a customer's bank account history (2–6 months of monthly data) and ask:
> *"How much income does this person reliably have, and how much debt can they safely take on?"*
> We answer this in a pipeline of 7 steps. Each step produces a number or a label that feeds the
> next. At the end, every customer gets an income estimate, a confidence score, and a credit
> decision. Nothing is a black box — every output field is defined, every number has a source.

---

## The Assembly Line Analogy

Think of it as a **7-station assembly line**. A customer record enters at Station 1 and exits at
Station 7 with a complete decision package. No station is skipped. No station can produce an
undefined output.

```
[Raw data]
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STATION 1 │ Data Intake      │ How much data do we have?        │
├─────────────────────────────────────────────────────────────────┤
│ STATION 2 │ SPARSE Check     │ Is there ENOUGH to estimate?     │
├─────────────────────────────────────────────────────────────────┤
│ STATION 3 │ Persona          │ What TYPE of income earner?      │
├─────────────────────────────────────────────────────────────────┤
│ STATION 4 │ Income Model     │ What is their income estimate?   │
├─────────────────────────────────────────────────────────────────┤
│ STATION 5 │ Reliability      │ How ACCURATE is that estimate?   │
├─────────────────────────────────────────────────────────────────┤
│ STATION 6 │ BCI              │ How CONFIDENT are we overall?    │
├─────────────────────────────────────────────────────────────────┤
│ STATION 7 │ Affordability    │ How much debt can they handle?   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
[Decision: STP_APPROVE / APPROVE_WITH_CHECKS / REFER / DECLINE]
```

---

## Station 1 — Data Intake: "How much data do we have?"

### What happens here
We take the customer's monthly aggregated bank data (credit totals, debit totals, transaction
counts) and immediately ask two questions:
- **Depth:** How many months do we have? (We need at least 2, ideally 6.)
- **Density:** How active is the account per month? (Sparse accounts = less reliable.)

We combine these into a single number called **OQS (Observation Quality Score)**, between 0 and 1.

```
OQS = (months we have / 6 months ideal) × weight
    + (avg transactions per month / 10 baseline) × weight
```

### What OQS is NOT
OQS is **not a gate**. A low OQS does not automatically reject the customer or route them to
SPARSE. OQS is just a number that feeds into every downstream model as a feature — like age
or income — so the models can adjust their outputs based on data quality.

### Feature masking
Some features need more months of data to be meaningful. If we don't have enough months:
- We **mask** those features (set to population average, not zero)
- We count how many features were masked (`feature_masking_count`)
- This count also feeds downstream models so they know how masked the record is

### Key output from this station
`oqs_score` (0–1), `feature_masking_count` (integer)

---

## Station 2 — SPARSE Check: "Is there ENOUGH to estimate income at all?"

### The fundamental question
Before we try to estimate income, we ask: *"Can we even produce a reliable income estimate
for this person given what we know?"* This is a yes/no question answered by a trained model
called the **data-sufficiency classifier**.

### THIN vs SPARSE — the most important distinction to understand

| | THIN | SPARSE |
|---|---|---|
| **What it means** | Low income or minimal activity, but we CAN estimate | Data is so poor we CANNOT estimate reliably |
| **Is it estimable?** | YES | NO |
| **What do we produce?** | A policy-floor income (BOT minimum) | A "REFER" decision with no income |
| **Is it a segment?** | YES — THIN is one of 6 income segments | NO — SPARSE is an estimation-feasibility flag |
| **Analogy** | A simple blood test with few markers: result is low but valid | A blood test where the sample was contaminated: no result |

**Rule of thumb:** THIN = "we know, and it's low." SPARSE = "we can't know."

### Key output from this station
If SPARSE: full output record with `decision = REFER`, no income, no BCI.
If not SPARSE: customer continues to Station 3.

---

## Station 3 — Persona: "What type of income earner is this customer?"

### Why persona matters
Different income types behave completely differently in bank data. A salaried employee gets
one large credit on the 25th every month. A freelancer gets lumpy irregular payments.
A small business owner has business income mixed in. A pass-through account sees large credits
that aren't income at all.

Using a single model for everyone would be like using one shoe size for all customers.
We need **persona-specific models**.

### The 6 personas

| Persona | Who they are | How income is estimated |
|---|---|---|
| **PAYROLL** | Regular salary employees — payroll credits visible | Direct observation (no model needed) |
| **L0** | Stable non-salary (e.g. professional fee, regular contract) | Quantile model on median credits |
| **L1** | Structured but irregular (e.g. consultant, overtime-heavy) | Quantile model on retained inflow proxy |
| **L2** | Informal / volatile (e.g. gig, seasonal, informal business) | Conservative quantile model (P25) |
| **THIN** | Low income or very low activity | Policy floor — no ML model |
| **PT** | Pass-through account (funds flow through, not earned here) | Formula: credits × retention ratio |

### How persona is assigned
Two-stage classifier:
1. First: "Is this THIN?" (binary yes/no model)
2. If not THIN: "Which of PAYROLL / L0 / L1 / L2 / PT?" (multiclass model)

This is a **learned model**, not a rulebook. The model looks at patterns in the data — credit
regularity, coefficient of variation, merchant categories, retention ratios — and outputs
a probability for each persona. The one with the highest probability wins.

### Stability: preventing persona flip-flopping
If a customer's persona jumps around month-to-month (e.g. L1 → L0 → L1 → L0), downstream
systems get confused. We apply a **smoothing step**:
- Keep a running average of the persona probabilities over time
- Only switch persona if the new persona is clearly stronger than the current one (by a margin δ)
- Target: fewer than 5% of customers switch persona in any given month

### Blending when uncertain (Mixture of Experts)
If the model is not confident which persona wins (e.g. 55% L1, 45% L0), instead of forcing
a hard choice, we **blend** the two income models in proportion to their probabilities.
This only applies to L0/L1/L2. THIN and PT always route to their own dedicated logic.

### Key output from this station
`persona` (one of 6), `persona_probabilities` (vector), smoothed probabilities

---

## Station 4 — Income Model: "What is the income estimate?"

### The core output: PolicyIncome
Every non-SPARSE customer gets a **PolicyIncome** — our official income estimate used for
credit decisions. Think of it as "the income figure we are prepared to stand behind."

### How it's produced (by persona)

**PAYROLL:** We read the payroll credits directly. No model. PolicyIncome = median payroll credit.
This is the most reliable estimate in the system.

**L0 (stable non-payroll):** A quantile regression model trained on customers where we know
the true income. It outputs a range (P35 to P50). PolicyIncome = P50 (the midpoint estimate).

**L1 (irregular structured):** Similar to L0, but the training target is a "retained inflow proxy"
rather than raw credits. Why? Because L1 customers often have pass-through money mixed with
real income. The proxy removes the noise before training. PolicyIncome = P50.

**L2 (informal/volatile):** Trained to P25 — a conservative lower bound. We deliberately
underestimate rather than overestimate for the most unpredictable earners. PolicyIncome = P25.

**THIN:** Not modelled. PolicyIncome = BOT minimum income floor (e.g. 15,000 THB/month).

**PT:** A formula: `median_credit_6m × min(retention_6m, retention_3m)`. The "retention ratio"
is how much of incoming money stays in the account (rather than flowing out immediately).
We take the more conservative of 3M and 6M retention to avoid periods of unusual PT activity.

### Calibrating the uncertainty bounds (CQR)
For L0, L1, L2 we apply **Conformalized Quantile Regression (CQR)** — a technique that
calibrates the model's uncertainty bounds using a holdout set so that the stated "P35"
actually covers 35% of true incomes. This is done separately per persona so that
high-variance personas (L2) don't borrow confidence from low-variance ones (PAYROLL).

Fallback if not enough data to calibrate per persona:
PERSONA (100+ cases) → SHRINKAGE (30–99) → POOLED (<30) → NO_CALIBRATION

### Key output from this station
`policy_income`, `income_p35`, `income_p50`, `income_source`

---

## Station 5 — Reliability Engine: "How accurate is that estimate?"

### The problem
Knowing that PolicyIncome = 45,000 THB is not enough. We also need to know:
- *How likely is it that the true income is within 10% of 45,000?* → `p_reliable10`
- *How likely is it that we've overestimated?* → `p_over10`

These two numbers are produced by **separate calibrated models** — they are not derived from
the income model itself. They take the income model's output, the data quality metrics, and
the persona, and predict how much we can trust the income estimate.

### Error Tolerance Bands
We classify estimation error into 5 bands:

| Band | Error size | Interpretation |
|---|---|---|
| VL | ≤ 5% | Negligible — trust the estimate |
| L | 5–10% | Low — normal operating range |
| M | 10–15% | Moderate — flag for review |
| H | 15–20% | High — manual check recommended |
| VH | > 20% | Very high — do not approve on estimate alone |

`p_reliable10` = probability that the actual error is within Band L or below.

### Key output from this station
`p_reliable10` (0–1), `p_over10` (0–1), `error_band` (VL/L/M/H/VH)

---

## Station 6 — BCI: "How confident are we overall?"

### What BCI is
The **Business Confidence Index** (BCI, 0–100) is a single number that summarises our
confidence in everything: the data quality, the persona assignment, the income estimate,
and the customer's behavioural consistency. Think of it as a "how sure are we?" dial.

### What BCI does and does NOT do

**BCI does:**
- Tighten or loosen the Burden Ratio cap (how much debt we allow relative to income)
- Influence the final credit decision category (STP / review / decline)

**BCI does NOT:**
- Change the income estimate (PolicyIncome is never multiplied or reduced by BCI)
- Override the income model

> **Key audit point:** In earlier designs, a low BCI would apply a "haircut" to income
> (e.g. multiply by 0.80). This was removed. BCI now acts only on the Burden Ratio cap.
> AdjustedIncome always equals PolicyIncome.

### How BCI is calculated
Five components, weighted:

| Component | Weight | What it measures |
|---|---|---|
| Income stability | 30% | How stable are the monthly credit patterns? |
| Data richness | 20% | OQS + how many features were masked |
| Persona clarity | 20% | How confident was the persona model? |
| Model uncertainty | 20% | 1 − p_reliable10 (lower reliability → lower BCI) |
| Behavioural consistency | 10% | Does spending behaviour match the income persona? |

### BCI Band → Burden Ratio Cap

| BCI Band | Score | Burden Ratio Cap | Meaning |
|---|---|---|---|
| HIGH | 80–100 | 0.45 | Strong confidence — allow up to 45% of income to debt |
| MEDIUM | 60–79 | 0.40 | Normal — 40% cap |
| LOW | 40–59 | 0.35 | Uncertain — tighten to 35%, flag for review |
| VERY_LOW | 0–39 | 0.30 | Poor confidence — 30% cap, likely decline |

### Persona BCI ceilings
A PAYROLL customer can score up to 100 BCI (we observe income directly).
A PT customer can score at most 50 BCI (bootstrap formula, not verified income).
A THIN customer caps at 30 BCI (minimal data, policy floor).
This prevents the BCI from overstating confidence where the underlying data is inherently limited.

### Key output from this station
`bci_score` (0–100), `bci_band` (HIGH/MEDIUM/LOW/VERY_LOW)

---

## Station 7 — Affordability Engine: "How much debt can they handle?"

### The core formula

```
Burden Ratio = Obligations / PolicyIncome
```

If `Burden Ratio > burden_cap_used` → the customer cannot afford the proposed obligation.

Note on terminology:
- **Burden Ratio** = Obligations ÷ Income (what fraction of income goes to debt). Lower is better.
- Some systems call the inverse (Income ÷ Obligations) "DSCR". We use Burden Ratio throughout
  to avoid confusion in audit reports.

### How the cap is determined

```
burden_cap_used = min(persona_burden_cap, bci_band_burden_cap)
```

Both gates must be satisfied. If the persona says 0.40 but the BCI says 0.35, we use 0.35.

**Persona caps — monotone rule:**
Higher uncertainty = tighter cap. Always.

| Persona | Cap |
|---|---|
| PAYROLL | 0.45 — most certain |
| L0 | 0.40 |
| L1 | 0.37 |
| L2 | 0.33 |
| PT | 0.33 — bootstrap formula, same floor |
| THIN | 0.35 — policy floor |

This monotonicity is enforced by architecture. It can never be violated by configuration.
A PAYROLL customer cannot get a tighter cap than an L2 customer for the same BCI.

### Max eligible obligation

```
max_eligible_obligation = PolicyIncome × burden_cap_used
```

This is the maximum total monthly obligation the customer can carry.
Compare to their proposed obligation to determine affordability.

### Key output from this station
`burden_cap_used`, `burden_ratio`, `max_eligible_obligation`, `decision`

---

## The Final Decision

Every customer gets exactly one of four decisions. No undefined states.

| Decision | Meaning |
|---|---|
| **STP_APPROVE** | Straight-through process — approve automatically |
| **APPROVE_WITH_CHECKS** | Approve but flag for secondary review |
| **REFER_FOR_INCOME_VERIFICATION** | Send to analyst for manual income check |
| **DECLINE** | Decline — confidence or affordability insufficient |

The decision depends on three things acting together:
1. **p_reliable10** — Is the income estimate reliable enough? (threshold: ≥ 0.70)
2. **p_over10** — Are we at low risk of overestimating? (threshold: ≤ 0.20)
3. **Burden Ratio** — Is the debt load within the cap?

> These thresholds (0.70, 0.20) are **policy operating thresholds** — not statistical constants.
> They will be calibrated against actual verification outcomes and portfolio performance.

---

## What Every Output Record Contains

Think of the output record as a **full audit trail in one row**. For every customer you can
trace exactly how the decision was made:

```
Who?       → customer_id, run_date, pipeline_version
What data? → oqs_score, feature_masking_count, sparse_reason_code
Who are    → persona, persona_probabilities
they?
Income?    → policy_income, income_p35, income_p50, income_source
How sure?  → p_reliable10, p_over10, error_band
Confidence?→ bci_score, bci_band
Affordable?→ persona_burden_cap, bci_band_burden_cap, burden_cap_used,
             burden_ratio, max_eligible_obligation
Decision?  → decision
Audit?     → income_haircut (always 1.0), adjusted_income (= policy_income),
             provisional_flags (which items are still provisional)
```

If an auditor asks "why was this customer declined?" — every answer is in this one record.

---

## Provisional vs Frozen — What This Means in Practice

**Frozen:** Defined by architecture. Cannot change without MRC approval. Examples:
- The pipeline structure (7 layers)
- The Burden Ratio formula
- The monotonicity constraint on persona caps
- The output schema field names

**Provisional:** Bootstrap values that will be replaced by calibrated values from data.
They are labelled `[PROVISIONAL]` throughout the architecture document and tracked in the
**Provisional Item Registry** (12 items, P-01 to P-12).
Examples: BCI component weights, CQR N thresholds, PT assignment threshold, EMA parameters.

**When does a provisional item get resolved?** When there is enough verified outcome data
(typically a 6M vintage) to calibrate the parameter properly. A calibrated value replaces
the provisional one via a standard model change request — not a full architecture change.

> **For auditors:** Every output record carries a `provisional_flags` field listing which
> P-items were active for that customer. This means the audit trail is fully self-documenting.

---

## Common Questions from Audit

**Q: Does BCI reduce the income estimate?**
No. AdjustedIncome always equals PolicyIncome. BCI only tightens the Burden Ratio cap.
The `income_haircut` field will always be 1.0. This is enforced in code, not policy.

**Q: Can a customer with very sparse data get approved?**
If predicted as SPARSE → always REFER, no approval. If THIN → policy floor income,
REFER decision. The system cannot approve a customer without a PolicyIncome.

**Q: What if the persona model is wrong?**
Two mitigations: (1) MoE blends top-2 personas when confidence is low, (2) the reliability
engine's `p_over10` catches overestimation risk regardless of which persona was assigned.

**Q: Why is there no 12-month data?**
The system is natively 2–6M. Requiring 12 months would exclude large portions of the
applicant pool. All features are designed for this window.

**Q: Why do PAYROLL customers bypass the income model?**
Their income is directly observable from payroll credits. Running it through a model would
add unnecessary uncertainty. Direct observation is always more reliable than model estimation.

**Q: What is the PT persona?**
Pass-Through accounts are accounts where money flows in and out quickly — the credits visible
in the account are not income, they are payments being routed through. The PT income formula
uses the **retention ratio** (how much stays after outflows) to estimate the portion that is
actually usable income, not just transit money.

**Q: What does `provisional_flags: ["P-07"]` mean in an output record?**
It means the PT income formula (P-07) was used for this customer's income estimate.
P-07 is a provisional bootstrap formula pending a proper learned model. The auditor knows
this estimate carries more uncertainty than a fully calibrated model would.

---

## One-Page Cheat Sheet

```
┌────────────────────────────────────────────────────────────────────┐
│                    PIPELINE CHEAT SHEET v5.0                       │
├────────────────────────────────────────────────────────────────────┤
│ DATA IN: 2–6 months of monthly aggregates                          │
│                                                                    │
│ OQS = data depth + data density (0–1). A model INPUT, not a gate. │
│                                                                    │
│ SPARSE = can't estimate. REFER. No income produced.               │
│ THIN   = low income but estimable. Policy floor. REFER.           │
│ (THIN ≠ SPARSE. Easily confused. Don't confuse them.)            │
│                                                                    │
│ PERSONAS: PAYROLL > L0 > L1 > L2 / PT / THIN / SPARSE            │
│ (ordered by income certainty, left = most certain)                 │
│                                                                    │
│ INCOME: PolicyIncome = what we stand behind                       │
│ • PAYROLL: read directly from account                             │
│ • L0/L1/L2: quantile model (P50 or P25)                          │
│ • THIN: policy floor (15,000 THB)                                 │
│ • PT: median_credit × retention_ratio                             │
│                                                                    │
│ RELIABILITY: p_reliable10 = P(error ≤ 10%)                        │
│              p_over10 = P(we overestimated by >10%)               │
│                                                                    │
│ BCI (0–100): overall confidence. HIGH/MEDIUM/LOW/VERY_LOW         │
│ BCI DOES NOT change income. It tightens the Burden Ratio cap.     │
│                                                                    │
│ BURDEN RATIO = Obligations / PolicyIncome                         │
│ CAP = min(persona cap, BCI band cap)                              │
│ Monotone: PAYROLL(0.45) ≥ L0(0.40) ≥ L1(0.37) ≥ L2≈PT(0.33)   │
│                                                                    │
│ DECISION:                                                          │
│  p_reliable10 ≥ 0.70 AND p_over10 ≤ 0.20 AND ratio ≤ cap        │
│  → STP_APPROVE                                                     │
│  Same reliability but ratio > cap → APPROVE_WITH_CHECKS           │
│  BCI LOW → REFER_FOR_INCOME_VERIFICATION                          │
│  BCI VERY_LOW → DECLINE                                            │
│                                                                    │
│ AUDIT: income_haircut always = 1.0 (no haircut ever applied)      │
│        adjusted_income = policy_income                             │
│        provisional_flags tells you what was a bootstrap estimate   │
└────────────────────────────────────────────────────────────────────┘
```
