# CLAUDE.md — Income Estimation & Affordability Framework (SCB/CardX Thailand)

## Project Overview

This repo implements a **gross income estimation and affordability assessment framework** for SCB/CardX Thailand credit decisions. The core problem: estimate a customer's income purely from transaction behavior when no declared income or bureau income data is available.

- **GitHub repo**: https://github.com/Sushil-tata/income-estimation-affordability
- **Institution**: Standard Chartered Bank (SCB) / CardX Thailand
- **Customer base**: 8 million SCB consent customers
- **Deployment environment**: **GDZ (Governed Data Zone)** — data cannot leave the zone
- **Stack**: Python, LightGBM, scikit-learn (Pandas-based, NOT Spark — GDZ compute constraint)
- **Status**: Framework built, needs testing with real GDZ data

## The Problem

CardX acquires many customers without declared income (e.g., supplementary cards, digital onboarding, informal workers). The Bank of Thailand (BOT) requires DSCR-based affordability assessment before credit limit assignment. Without income, DSCR cannot be computed.

**Solution**: Infer gross income band from behavioral signals in transaction history (spend patterns, inflow regularity, merchant category mix, timing patterns) using a two-stage ML model.

**Key constraint**: All computation must run within GDZ. No data export. LightGBM chosen over XGBoost/CatBoost due to GDZ compute limits and faster inference.

---

## Architecture

```
Customer
    │
    ▼
[PAYROLL check]
    │
    ├─ YES → PAYROLL BYPASS: income known, skip estimation
    │                         go directly to Affordability
    │
    └─ NO  → Behavioral Segmentation
                    │
                    ▼
             [5 Behavioral Segments]
                    │
                    ▼
             Two-Stage Income Estimation
                    │
                    ├── Stage 1: LightGBM Band Classifier
                    │           → assigns BAND_1 to BAND_5
                    │
                    └── Stage 2: Quantile Regression per band
                                → outputs Q25 / Q50 / Q75
                    │
                    ▼
             Business Confidence Index (BCI)
                    │
                    ▼
             Affordability Assessment (BOT DSCR norms)
                    │
                    ▼
             Policy Decision
                    │
                    ├── STP_APPROVE
                    ├── STP_DECLINE
                    ├── REFER_INCOME_VERIFY
                    └── MANUAL_REVIEW
```

---

## Behavioral Segmentation

Before income estimation, customers are segmented by behavioral pattern. Each segment has distinct income estimation model parameters.

| Segment | Profile | Key Signals |
|---|---|---|
| **SALARY_LIKE** | Regular monthly inflows, stable spend | End-of-month inflow spikes, consistent merchant patterns |
| **SME** | Business owner / self-employed | Irregular but large inflows, B2B merchants, high variance |
| **GIG_FREELANCE** | Gig worker / freelancer | Multiple small inflows, platform merchants (Grab, Lazada), irregular |
| **PASSIVE_INVESTOR** | Investment income | Inflows from securities/fund merchants, low spend volatility |
| **THIN** | Insufficient history | <3 months data or <5 transactions/month |

THIN customers route to REFER_INCOME_VERIFY by default (insufficient behavioral signal for estimation).

---

## Two-Stage Income Estimation Model

### Stage 1: LightGBM Band Classifier
- **Task**: Multi-class classification → assign customer to income band
- **Bands**:
  | Band | Monthly Income (THB) |
  |---|---|
  | BAND_1 | < 15,000 |
  | BAND_2 | 15,000 – 30,000 |
  | BAND_3 | 30,000 – 60,000 |
  | BAND_4 | 60,000 – 120,000 |
  | BAND_5 | > 120,000 |
- **Training data**: 160,000 verified lending records (customers where income was declared/verified)
- **Features**: 80+ behavioral features (see Feature Engineering section)
- **Why LightGBM**: GDZ compute limits favor fast gradient boosting; also interpretable via SHAP

### Stage 2: Quantile Regression (per band)
- Separate quantile regressor trained for each BAND
- Outputs three estimates: **Q25** (conservative), **Q50** (central), **Q75** (optimistic)
- Q50 used as point estimate; Q25/Q75 used for BCI computation and policy routing

---

## Feature Engineering

Features are computed from transaction history in GDZ. Key feature families:

| Family | Examples |
|---|---|
| **Inflow patterns** | Total monthly inflow, inflow count, inflow regularity score, max single inflow |
| **Outflow patterns** | Spend-to-inflow ratio, merchant category spend shares |
| **Payroll signals** | Salary-like inflow flag, day-of-month inflow concentration |
| **Merchant behavior** | MCG diversity, essential vs. discretionary split, e-commerce ratio |
| **Temporal** | Weekday/weekend spend ratio, late-night ratio, payday spike index |
| **Volatility** | Coefficient of variation (monthly spend/inflow), trend slope |
| **Thin signals** | Transaction count per month, months with any activity |

All features are PIT-safe. Feature selection module at `src/feature_selection/`.

---

## Business Confidence Index (BCI)

BCI is a 0–100 composite score measuring how much to trust the income estimate. It drives routing decisions (STP vs. manual review).

### 5 BCI Components

| Component | Weight | What It Measures |
|---|---|---|
| **Stability** | 30% | Consistency of behavioral signals over time |
| **Segment Clarity** | 20% | How cleanly the customer fits one behavioral segment |
| **Data Richness** | 20% | Volume and recency of transaction history |
| **Model Confidence** | 20% | LightGBM predicted probability for the assigned band |
| **Behavioral Consistency** | 10% | Agreement between Stage 1 band and Stage 2 point estimate |

### BCI → Routing

| BCI Range | Routing Decision |
|---|---|
| BCI >= 75 | STP path (straight-through processing) |
| BCI 50–74 | Soft STP with policy guardrails |
| BCI 30–49 | REFER_INCOME_VERIFY |
| BCI < 30 | MANUAL_REVIEW |

Implementation: `src/bci/`

---

## Affordability Assessment

BOT (Bank of Thailand) norms applied after income estimation:

- **Minimum income threshold**: >= 15,000 THB/month gross income (BAND_1 customers fail this check)
- **DSCR (Debt Service Coverage Ratio)**: Total monthly debt obligations / Monthly income <= 40%
- **Existing obligations**: pulled from NCB bureau (if available) or estimated from transaction outflows

Implementation: `src/affordability/`

---

## Policy Outcomes

| Outcome | Condition |
|---|---|
| **STP_APPROVE** | BCI >= 75, income >= 15K THB, DSCR <= 40%, no adverse flags |
| **STP_DECLINE** | BCI >= 75, income < 15K THB or DSCR > 40% |
| **REFER_INCOME_VERIFY** | BCI 30–74, or THIN segment |
| **MANUAL_REVIEW** | BCI < 30, conflicting signals, or policy override triggers |

Implementation: `src/offer/` (offer/policy assignment)

---

## Repository Structure

```
src/
  segmentation/         — behavioral segment classifier
  income_estimation/    — two-stage model (band classifier + quantile regressor)
  feature_selection/    — feature importance, stability filters
  bci/                  — Business Confidence Index computation
  affordability/        — BOT DSCR norms, affordability scoring
  modeling/             — model training utilities, cross-validation
  monitoring/           — model drift, PSI, performance tracking
  inference_pipeline.py — end-to-end inference orchestration
  offer/                — policy decision and offer assignment
  utils/                — shared utilities
config/                 — environment configs
data/                   — data schemas, sample data
notebooks/              — EDA, model development notebooks
scripts/                — training and batch inference scripts
tests/                  — unit + integration tests
docs/                   — architecture diagrams, BOT compliance notes
```

---

## Key Design Decisions

1. **LightGBM only (no XGBoost/CatBoost)**: GDZ has strict compute limits. LightGBM is faster to train and score in constrained environments. Decision is intentional and should not be changed without re-evaluating GDZ constraints.
2. **Two-stage design**: Band classifier first reduces variance of the quantile regressor — each band's regressor is calibrated on a homogeneous sub-population.
3. **BCI as routing lever**: Rather than hard thresholds, BCI allows the business to tune STP rate vs. manual review trade-off without retraining models.
4. **PAYROLL bypass**: Payroll customers are excluded from estimation — their income is known from bank transaction data and does not need ML inference.
5. **160K training records**: Verified lending records (not self-declared income) are used as ground truth — reduces self-declaration bias.
6. **Quantile outputs (Q25/Q50/Q75)**: Enables risk-adjusted credit limit assignment (e.g., use Q25 for higher-risk segments).

---

## Current Status

**Framework built. Needs testing with real GDZ data.**

| Component | Status |
|---|---|
| Behavioral segmentation | Built |
| Feature engineering | Built |
| Two-stage income model | Built (trained on 160K records) |
| BCI computation | Built |
| Affordability assessment | Built |
| Policy routing | Built |
| Inference pipeline (end-to-end) | Built |
| Monitoring hooks | Built (scaffolding) |
| Real GDZ data testing | NOT YET DONE |

---

## What's Next (Picking Up Where Left Off)

1. **Connect to real GDZ data**: Mount GDZ data sources; validate transaction table schema matches feature engineering assumptions; check for Thai-specific edge cases (Buddhist calendar dates, THB decimal handling)
2. **Run OOT validation**: Hold out 3 months of labeled data (customers with verified income); test band classifier accuracy and quantile calibration
3. **Calibrate BCI thresholds**: Initial BCI thresholds (75/50/30) are heuristic — calibrate against manual review outcomes and STP approval rates on real data
4. **Tune DSCR affordability**: Validate that 40% DSCR threshold matches BOT circular requirements; check for product-specific exceptions (secured vs. unsecured)
5. **Deploy inference pipeline**: Package `inference_pipeline.py` for batch scoring in GDZ; set up scheduled runs

---

## Important Context for Next Session

- **Line Assignment is a SEPARATE sub-agent** — it is NOT part of this repo. After affordability assessment outputs a policy decision, Line Assignment determines the specific credit limit. Do not conflate the two.
- **GDZ constraint is hard**: Data cannot leave the zone. No external API calls, no cloud model serving outside GDZ. All inference must run inside GDZ using local compute.
- **Buddhist calendar**: Thailand uses the Buddhist Era (BE) calendar — year is CE + 543. Transaction timestamps from Thai systems may use BE dates. Feature engineering must handle this.
- **THB amounts**: All monetary features are in Thai Baht (THB). Do not normalize against USD without explicit business requirement.
- **THIN segment handling**: THIN customers (< 3 months history) are a significant fraction of new-to-bank customers. The framework routes them to REFER_INCOME_VERIFY — do not try to force-score them through the ML model.
- **8M customer scope**: The 8M figure is SCB consent customers eligible for behavioral analysis — not all are CardX cardholders. The income estimation applies to the subset applying for CardX products.
