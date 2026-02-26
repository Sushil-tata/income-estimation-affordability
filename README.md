# Income Estimation & Affordability Framework
**CardX / SCB Group | GDZ Environment**

---

## Overview

A modular Python framework for estimating customer gross income from transaction data and computing credit affordability — without relying on declared income or bureau data.

Designed for deployment in the **Governed Data Zone (GDZ)** with compute and data export constraints.

---

## Three Outputs

| Output | Description |
|---|---|
| **Estimated Income** | Gross monthly income (THB) inferred from deposits + transaction behavior |
| **Affordability** | ADSC computation against BOT DSCR norms (≥15K income, ≤40% DSCR) |
| **Business Confidence Index (BCI)** | 0–100 score indicating reliability of the income estimate → drives STP vs manual review |

> Line Assignment is a **separate sub-agent** and is not part of this repository.

---

## Architecture

```
Customer Base (8M SCB consent)
│
├── PAYROLL  ──────────────────────────────────► Income Known (bypass estimation)
│
└── NON-PAYROLL
      │
      ▼
  [1] Behavioral Segmentation
      SALARY_LIKE | SME | GIG_FREELANCE | PASSIVE_INVESTOR | THIN
      │
      ▼
  [2] Income Estimation (Two-Stage)
      Stage 1: LightGBM Band Classifier  (BAND_1 ... BAND_5)
      Stage 2: Quantile Regression       (Q25 | Q50 | Q75)
      Training: 160K verified lending records
      │
      ▼
  [3] Business Confidence Index (BCI)
      Stability x Segment Clarity x Data Richness x Model Confidence x Behavioral Consistency
      │
      ▼
  [4] Affordability Engine
      ADSC = (Adjusted Income x 40%) - Existing Obligations
      │
      ▼
  [5] Policy Engine
      STP_APPROVE | STP_DECLINE | REFER_INCOME_VERIFY | MANUAL_REVIEW
```

---

## Repository Structure

```
income-estimation-affordability/
├── config/
│   ├── config.yaml           # BOT norms, BCI weights, model params, thresholds
│   └── feature_config.yaml   # P&L feature taxonomy (commitments, recurring, lifestyle)
│
├── src/
│   ├── segmentation/         # Behavioral segmentation pipeline
│   │   ├── rules.py          # Rule-based: PAYROLL, SALARY_LIKE
│   │   ├── clustering.py     # Behavioral: SME, GIG, PASSIVE, THIN
│   │   └── pipeline.py       # Orchestration
│   │
│   ├── income_estimation/    # Two-stage income estimation
│   │   ├── features.py       # Feature engineering (monthly P&L aggregates)
│   │   ├── band_model.py     # Stage 1: LightGBM band classifier
│   │   ├── regression.py     # Stage 2: Quantile regression within bands
│   │   └── pipeline.py       # Orchestration + payroll bypass
│   │
│   ├── bci/                  # Business Confidence Index
│   │   ├── components.py     # Five component scorers
│   │   └── scorer.py         # Weighted aggregation + segment caps + policy
│   │
│   ├── affordability/        # Affordability engine + policy
│   │   ├── engine.py         # DSCR calculation, ADSC, stress testing
│   │   └── policy.py         # Final decision matrix (STP / Refer / Decline)
│   │
│   └── utils/
│       ├── validation.py     # Data quality checks
│       └── logging.py        # Logging setup
│
├── notebooks/
│   ├── 01_segmentation.ipynb
│   ├── 02_income_estimation.ipynb
│   ├── 03_bci.ipynb
│   └── 04_affordability.ipynb
│
├── data/
│   ├── raw/          # GDZ data - gitignored
│   ├── processed/    # Processed features - gitignored
│   └── sample/       # Synthetic sample data for development
│
└── requirements.txt
```

---

## Income Estimation — Approach

The two-stage architecture addresses the **high-variance Tweedie regression problem**:

| Problem | Solution |
|---|---|
| Open-ended regression: high variance | Bounded prediction within income bands |
| Mean unstable for skewed income | Predict **median (Q50)** as point estimate |
| No confidence signal | Q25-Q75 interval feeds BCI |
| Single model for all segments | Segment-specific models (SALARY_LIKE, SME, GIG) |

---

## BCI — Component Weights

| Component | Weight | Description |
|---|---|---|
| Income Stability | 30% | CV of credits, zero-credit months, interval width |
| Segment Clarity | 20% | Fit to behavioral segment |
| Data Richness | 20% | History depth + transaction density |
| Model Confidence | 20% | Band classifier margin |
| Behavioral Consistency | 10% | Spend/balance match predicted income |

**Segment BCI Caps:** PAYROLL=100, SALARY_LIKE=90, SME=70, GIG=65, THIN=30

---

## Constraints

- **GDZ environment**: Data does not leave the zone; all modeling in-zone
- **Compute**: LightGBM chosen for speed and memory efficiency
- **Bureau data**: Available but not used as primary feature; validation only
- **Primary signal**: Transaction behavior from deposits + CardX data only

---

## Quick Start

```python
from src.segmentation import SegmentationPipeline
from src.income_estimation import IncomeEstimationPipeline
from src.bci import BCIScorer
from src.affordability import AffordabilityEngine, PolicyEngine

# Step 1: Segment
seg_pipeline = SegmentationPipeline(config_path="config/config.yaml")
features_df = seg_pipeline.run(features_df)

# Step 2: Estimate income
income_pipeline = IncomeEstimationPipeline(config_path="config/config.yaml")
income_pipeline.fit(train_df, y_verified_income)
income_results = income_pipeline.predict(features_df)

# Step 3: BCI
scorer = BCIScorer(config_path="config/config.yaml")
bci_results = scorer.compute(features_df, income_results)

# Step 4: Affordability
engine = AffordabilityEngine(config_path="config/config.yaml")
affordability = engine.compute(bci_results, features_df)

# Step 5: Policy decision
policy = PolicyEngine(config_path="config/config.yaml")
final_output = policy.get_full_output(bci_results, affordability, income_results, features_df)
```
