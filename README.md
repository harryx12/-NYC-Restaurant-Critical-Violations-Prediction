# Predicting NYC Restaurant Critical Violations



---

## Overview

Can we predict whether a restaurant's next NYC health inspection will result in a critical violation?

The New York City Department of Health and Mental Hygiene (DOHMH) inspects 26,000+ active food service establishments annually across 5 boroughs. This project builds a machine learning pipeline that assigns each restaurant a **risk score** — the predicted probability that its next inspection will flag a critical violation — enabling DOHMH to prioritize re-inspection schedules toward the highest-risk establishments without increasing total inspection volume.

---

## Dataset

| Property | Details |
|---|---|
| **Source** | [NYC OpenData — DOHMH Restaurant Inspection Results](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j/about_data) |
| **Raw size** | 295,995 rows × 27 columns |
| **Unique restaurants** | 30,935 (by CAMIS ID) |
| **Data currency** | Through April 2, 2026 |
| **External data** | None — sole source, no third-party APIs |

**Key fields used:**

- `CAMIS` — Unique restaurant identifier
- `INSPECTION DATE` — Enables temporal ordering and lag features
- `CRITICAL FLAG` — Binary target (54% critical, 46% not critical)
- `SCORE` — Numeric violation severity (mean ≈ 25, capped at 99th percentile)
- `VIOLATION CODE` — 27 standardized codes used for flag features
- `CUISINE`, `BORO`, `GRADE` — Structural and geographic signals

---

## Project Structure

```
├── Harshit_Raj_Taneja_DATA602_Project_uploads.ipynb   # Main analysis notebook
├── Project_Data602_Harshit_Raj_Taneja.pptx            # Presentation slides
└── README.md
```

**Notebook sections:**

1. Environment Setup
2. Load Data
3. Data Cleaning & Preprocessing
4. Exploratory Data Analysis (EDA)
5. Aggregate to Inspection Level
6. Target Definition & Train/Val/Test Split
7. Feature Engineering
8. Model Training (4 model families with GridSearchCV)
9. Model Comparison
10. Final Test Evaluation
11. Feature Importance
12. Calibration Analysis
13. Risk Scorecard
14. Project Summary

---

## Data Cleaning

Steps applied before modeling:

- Parsed `INSPECTION DATE` and `GRADE DATE` as datetime
- Removed rows with sentinel date `1900-01-01` (DOHMH system placeholder)
- Filtered to **Cycle Inspections only** (Initial + Re-inspection); excluded Pre-permit and Administrative visits
- Standardized text columns (strip whitespace, title case)
- Removed unknown borough code `'0'`
- Capped `SCORE` at the 99th percentile to reduce outlier influence

After cleaning: ~218,000 inspection records retained.

---

## Exploratory Data Analysis

Four key findings from EDA:

1. **Critical Flag Distribution** — Critical violations (121,924) outnumber non-critical (96,513), making up the largest share of inspection records.
2. **Critical Rate Over Time** — Post-2015 stabilization at 54–60% demonstrates data consistency over a decade, providing a strong foundation for temporal modeling.
3. **Score by Borough** — Similar distributions across boroughs allow the model to generalize geographically without overfitting to any one area.
4. **Cuisine Type** — Critical violation rates vary meaningfully by cuisine, providing a stable categorical feature with genuine predictive power.

Additionally, **Violation Code 10F** dominates (~34K records), nearly 50% more frequent than the second-most-common code (08A) — a strong candidate for a predictive feature.

---

## Feature Engineering

| Feature Group | Description |
|---|---|
| **Lag features** | Prior visit score, critical flag, grade, days since last inspection |
| **Violation code history** | Binary indicators for the most predictive violation codes from prior visit |
| **Establishment count** | Total prior inspections logged — proxy for compliance maturity |
| **Rolling critical rate** | Mean of Critical Flag over prior 3 and 5 inspections per restaurant |
| **Inspection type** | Encoded as 4 categories: Cycle Initial, Re-inspection, Pre-permit, Compliance |
| **Cuisine type** | Smoothed target encoding (`TargetEncoder`, smoothing=10), fitted on training set only — no leakage |

---

## Target & Train/Val/Test Split

**Target definition:**
```python
TARGET = CURRENT_CRITICAL.shift(-1)
# Predicts whether the restaurant's NEXT inspection will be critical
```

Restaurants with only one inspection are excluded (no lag features possible). The last inspection per restaurant is dropped (no future outcome to predict).

**Strict time-based split:**

| Set | Date Range | Purpose |
|---|---|---|
| Train | Up to Dec 31, 2023 | Model fitting |
| Validation | Jan 1 – Dec 31, 2024 | Model selection |
| Test | Jan 1, 2025 onward | Final evaluation (used once) |

---

## Modeling

Four model families trained with **5-fold Stratified CV + GridSearchCV**:

| Model | Type | Key Hyperparameters Tuned |
|---|---|---|
| Logistic Regression | Linear baseline | C, penalty (L1/L2), solver |
| Random Forest | Bagging ensemble | n_estimators, max_depth, min_samples_split |
| XGBoost | Boosting ensemble | learning_rate, max_depth, subsample, n_estimators |
| SVM | Margin-based | C, kernel (RBF), gamma |

**Primary metric:** F1-score (balances precision and recall for near-equal class distribution)  
**Secondary:** AUC-ROC, precision, recall, accuracy

Note: LR and SVM use `StandardScaler`; tree-based models do not. k-NN was explicitly excluded as it memorizes geometric similarity rather than learning structural factors like prior violations and rolling critical counts.

---

## Results

### Model Comparison (Validation Set)

**Random Forest** was selected as the best model based on highest validation F1-score. It captures non-linear relationships without requiring manual feature interaction terms, showed less CV-to-validation F1 gap than XGBoost (better generalization), and natively supports feature importance extraction for the operational risk scorecard.

Logistic Regression's AUC of 0.691 — despite being the simplest model — validates that the engineered features carry strong, linearly-accessible signal.

### Final Test Set Results (Random Forest)

| Metric | Score |
|---|---|
| **F1 Score** | 0.9258 |
| **Recall** | ~99.9% |
| **AUC-ROC** | 0.7236 |

- **5,181 true positives** with only **6 false negatives** out of 5,187 actual critical cases
- The model generalizes well to truly unseen data (inspections after both training and validation periods)

---

## Feature Importance (Top 3)

1. **`DAYS_SINCE_LAST`** (0.1435) — Inspection recency independently discovered by the model; validates domain knowledge
2. **`SCORE_DELTA`** (0.1216) — Captures improvement or deterioration trends, effectively tracking restaurant momentum
3. **`CUISINE_ENC`** (0.1210) — Cuisine encoding serves as a proxy for kitchen complexity and operational patterns

The gradual decay across the remaining 17 features indicates no single variable dominates — the model uses a broad, stable evidence base.

---

## Risk Scorecard

Each restaurant in the test set receives a `RISK_SCORE` — the predicted probability that its next inspection will be critical. The output scorecard surfaces the top 25 highest-risk establishments with:

- Restaurant name, borough, and cuisine
- Prior inspection score and rolling critical rate (last 5 inspections)
- Predicted risk score
- Ground-truth validation (whether the next inspection actually was critical)

**Operational use:** DOHMH can sort this list and deploy inspectors to highest-risk establishments first — improving public health outcomes without increasing total inspection volume.

**Borough-level risk** ranges tightly from 88–90%, with the Bronx and Queens at 90.2%, giving inspectors a data-driven justification for geographic prioritization.

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn category_encoders xgboost
```

The notebook is designed to run on **Google Colab**. Update `RAW_PATH` to point to the downloaded DOHMH CSV file:

```python
RAW_PATH = '/content/DOHMH_New_York_City_Restaurant_Inspection_Results_20260404.csv'
```

---

## Key Takeaways

- A machine learning model trained entirely on historical DOHMH inspection records can predict future critical violations with **~99.9% recall**, catching nearly every high-risk restaurant before the next visit.
- The most informative predictors are **recency** (days since last inspection), **trend** (score delta), and **cuisine type** — intuitive signals that align with domain expertise.
- The model produces a deployable, prioritized **risk scorecard** that could directly inform DOHMH's inspection scheduling workflow.
