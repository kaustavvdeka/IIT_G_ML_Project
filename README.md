# 🏃‍♂️ PLAYHACK — End-to-End Sports Injury Prediction Pipeline

An enterprise-grade, leakage-safe, beginner-friendly machine learning pipeline for sports injury prediction, onset estimation, and recovery duration forecasting built on multi-modal wearable, biometric, and training session data.

---

## 📌 1. Problem Statement & Objectives

Sports injuries impact athlete health, team performance, and club economics. The goal of this project is to build an interpretable, leakage-safe machine learning system to evaluate future injury risk across three distinct tasks:

| Task | Prediction Target | Description | Bound / Range | Official Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Task A** | `injured_in_risk_window` | Binary classification of injury in 30-day risk window | 0 = No, 1 = Yes | **F1 Score** (threshold tuned) |
| **Task B** | `onset_day_offset` | Predicted day of injury inside the risk window | 1 – 30 days | **MAE** (on injured athletes) |
| **Task C** | `recovery_duration` | Expected recovery period in days | ≥ 0 days | **MAE** (on injured athletes) |

> [!NOTE]
> **Conditional Modeling**: Tasks B and C are trained *only* on athletes where `injured_in_risk_window == 1`, as onset and recovery duration are mathematically and clinically conditional on an injury occurring.

---

## 🏛️ 2. Pipeline Architecture

```text
                                PLAYHACK Pipeline Architecture
                                
       ┌────────────────────────────── Multi-Source Raw CSVs ──────────────────────────────┐
       │ dailyActivity | hourlyHeartrate | sleepDay | training_sessions | athlete_metadata │
       └─────────────────────────────────────────┬──────────────────────────────────────────┘
                                                 │
                                                 ▼
                                     [ Data Validation & QA ]
                                    (Missing, Duplicates, Types)
                                                 │
                                                 ▼
                                     [ Feature Engineering ]
                             (7d, 14d, 30d Aggregations, ACWR Trends)
                                                 │
                                                 ▼
                                      [ Leakage Guard Filter ]
                            (Strict Obs-Window Cutoff: Day 1-30 Only)
                                                 │
                                                 ▼
                                     [ Athlete-Level Dataset ]
                                      (outputs/features.parquet)
                                                 │
                                                 ▼
                                      [ 5-Fold GroupKFold ]
                                   (Grouped strictly by athlete_id)
                                                 │
               ┌─────────────────────────────────┼─────────────────────────────────┐
               ▼                                 ▼                                 ▼
      [ Task A: Classification ]        [ Task B: Onset (MAE) ]         [ Task C: Recovery (MAE) ]
      - XGBoost / LightGBM             - XGBoost Regressor             - XGBoost Regressor
      - Random Forest / ExtraTrees     - LightGBM Regressor            - LightGBM Regressor
      - HistGradientBoosting           - CatBoost Regressor            - CatBoost Regressor
      - Logistic Regression            - Random Forest Regressor       - Random Forest Regressor
               │                                 │                                 │
               ▼                                 ▼                                 ▼
      [ F1 Threshold Tuning ]           [ Bounds Clipping: 1-30 ]       [ Non-Negative Clip: >=0 ]
               │                                 │                                 │
               ▼                                 └────────────────┬────────────────┘
      [ Top-3 Model Ensemble ]                                    │
               │                                                  │
               └─────────────────────────┬────────────────────────┘
                                         ▼
                             [ Leaderboard & Artifacts ]
                             - outputs/benchmark_results.json
                             - outputs/feature_importance.csv
                             - outputs/submission.csv
                                         │
                                         ▼
                           [ Streamlit Web Application ]
```

---

## 🛡️ 3. Leakage Prevention & Temporal Boundaries

Temporal leakage is the most prevalent failure mode in wearable health modeling. This project employs a **structural leakage guard**:

```text
       Observation Window (Day 1 – 30)                Risk Window (Day 31 – 60)
 ├─── 2026-01-05 ─────────────── 2026-02-03 ───┤ ├─── 2026-02-04 ─────────────── 2026-03-05 ───┤
                    ▲                                              ▲
               FEATURE DATA                                 PREDICTION TARGET
             (Extract features)                             (Predict outcome)
```

1. **Explicit Time Anchors**: Every feature builder takes `obs_end` as its upper bound. Data strictly satisfies `date <= obs_end`.
2. **Column Inspection**: `leakage_guard()` scans for suspicious terms (`future`, `risk`, `outcome`, `post`, `onset`, `recovery`) and raises immediate exceptions if any unapproved column appears in the feature table.
3. **No Risk-Window Ingestion**: Risk window data is never loaded into feature extraction routines.

---

## 👥 4. GroupKFold Cross-Validation

Standard random K-Fold randomly splits rows, leaking biometrics from the same athlete across train and validation sets. 

We use **`GroupKFold(n_splits=5)`** grouped by `athlete_id`.
- For every fold, the pipeline asserts:
  $$\text{Train Athletes} \cap \text{Validation Athletes} = \emptyset$$
- If any overlap is detected, execution halts immediately with a descriptive error.

---

## 🎯 5. F1 Threshold Optimization

Binary classifiers default to a $0.5$ probability threshold. Under class imbalance (injury rate ~15–30%), this degrades the F1-Score.

Our pipeline sweeps thresholds from $0.05$ to $0.95$ on Out-of-Fold (OOF) validation predictions:
$$T^* = \arg\max_{T \in [0.05, 0.95]} \text{F1}(y_{\text{true}}, \mathbb{I}(P \ge T))$$

---

## 🧠 6. Candidate Model Zoo

| Tier | Classifiers (Task A) | Regressors (Tasks B & C) |
| :--- | :--- | :--- |
| **Tier 1 (Boosted Trees)** | XGBoost, LightGBM, CatBoost | XGBoost, LightGBM, CatBoost |
| **Tier 2 (Bagging/Ensembles)** | Random Forest, Extra Trees, HistGBM | Random Forest, Extra Trees, HistGBM |
| **Tier 3 (Linear Baselines)** | Logistic Regression | Ridge Regression |
| **Ensemble** | Top-3 OOF Probability Average | — |

*Optional dependencies (`xgboost`, `lightgbm`, `catboost`) are detected dynamically; if missing, the pipeline logs a notice and continues with available models.*

---

## 🚀 7. Installation & Quick Start

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run Feature Engineering
Extracts multi-window aggregations and creates `outputs/features.parquet`:
```bash
python feature_engineering.py --data_dir ./data --out outputs/features.parquet
```

### 3. Train & Evaluate Models
Trains all candidate models across GroupKFold, tunes F1 thresholds, and logs the benchmark:
```bash
python train_and_evaluate.py --features outputs/features.parquet --folds 5
```

### 4. Generate Predictions & Submission
Produces the final competition submission file:
```bash
python predict.py --features outputs/features.parquet --out outputs/submission.csv
```

### 5. Launch Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 🖥️ 8. Interactive Streamlit Web Application

The interactive web dashboard includes 10 comprehensive sections:
1. **📊 Dashboard**: High-level KPIs, athlete count, injury rate, and best CV scores.
2. **📂 Dataset Explorer**: Interactive viewer for raw tables and feature parquets with athlete search.
3. **👤 Athlete View**: Demographic profiles, longitudinal trend charts, Risk Speedometer, and interactive single-athlete SHAP Waterfall plots.
4. **🏅 Team Risk Ranking**: Sortable live leaderboard prioritizing athletes by their current calculated injury risk and expected onset.
5. **🎮 What-If Simulator**: Interactive sliders allowing coaches to tweak recent workload/sleep and instantly recalculate the athlete's future injury probability.
6. **🔬 Feature Explorer**: Descriptive statistical summaries, distribution histograms, and an interactive Feature Correlation Matrix.
7. **🏆 Model Benchmark**: Complete leaderboard table and CV performance comparison charts.
8. **🎯 Classification Analysis**: Interactive Out-Of-Fold Confusion Matrix, precision-recall metrics, and F1 threshold curves.
9. **🌲 Feature Importance**: Top 20 predictive features and explainability analysis.
10. **🔮 Prediction Tool**: Live athlete injury probability estimator.

---

## 📁 9. Project Directory Layout

```text
playhack-injury-prediction/
├── data/                      # Raw datasets & test directory
│   └── README.md
├── src/                       # Core modular source code
│   ├── __init__.py
│   ├── config.py              # Configuration, aliases, paths, parameters
│   ├── utils.py               # Logger, column finder, date parser, serializer
│   ├── data_loader.py         # Standardized CSV readers
│   ├── data_validation.py     # Schema checking, missing values, anomaly audit
│   ├── leakage_guard.py       # Zero-leakage temporal boundary enforcement
│   ├── feature_engineering.py # Multi-granular rolling aggregations & ACWR
│   ├── models.py              # Model zoo and tabular preprocessors
│   ├── evaluation.py          # Metrics, GroupKFold validator, threshold optimizer
│   ├── training.py            # CV orchestrator, top-3 ensemble, model persistence
│   ├── prediction.py          # Inference engine & submission builder
│   └── shap_analysis.py       # SHAP attributions & hypothesis tests
├── app/                       # Streamlit web application
│   ├── streamlit_app.py       # Multi-tab dashboard
│   └── components.py          # Visual chart generators
├── outputs/                   # Generated artifacts
│   ├── features.parquet       # Extracted feature dataset
│   ├── benchmark_results.json # Comprehensive CV benchmark
│   ├── feature_importance.csv # Ranked feature importances
│   ├── predictions.csv        # Out-of-fold validation predictions
│   ├── submission.csv         # Formatted competition submission
│   ├── models/                # Serialized model pickles
│   └── plots/                 # Saved visual figures
├── notebooks/
│   └── exploration.ipynb      # EDA and pipeline demonstration notebook
├── feature_engineering.py     # CLI feature extraction entrypoint
├── train_and_evaluate.py      # CLI model training entrypoint
├── predict.py                 # CLI inference entrypoint
├── requirements.txt           # Python package requirements
├── README.md                  # Project documentation
└── .gitignore                 # Cache and artifact ignore rules
```

---

## 🔬 10. Important Scientific Limitations

1. **Correlation vs. Causation**: High importance of metrics such as `training_load_change_7d_vs_30d` indicates statistical association with future injury occurrence, not definitive biomedical causation.
2. **Observational Data Quality**: Wearable sensor records can contain unmeasured confounders (e.g. nutrition, psychological stress, travel schedules, surface hardness).
3. **Generalizability**: Models trained on this dataset reflect specific sporting cohorts and should be externally validated before clinical adoption.
4. **Medical Disclaimer**: *All predictions generated by this pipeline are statistical risk estimates and should not be used as clinical diagnoses or medical advice.*
