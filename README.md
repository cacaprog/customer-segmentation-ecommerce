# 🛍️ Intelligent Customer Segmentation & Revenue Optimization

> **An end-to-end data science portfolio project** transforming raw e-commerce transactions into actionable customer intelligence — from data engineering to behavioral clustering, predictive modeling, and a live BI dashboard.

---

## 📌 Project Overview

This project applies advanced machine learning and statistical analysis to the **UCI Online Retail II Dataset** — over 1 million transactions from a UK-based B2B wholesale retailer spanning 24 months (Dec 2009 – Dec 2011).

The goal: move beyond descriptive analytics and deliver **production-ready customer intelligence** that marketing and commercial teams can act on immediately.

| Metric | Value |
|---|---|
| **Total Customers Analyzed** | 5,878 |
| **Total Revenue Represented** | £17.7 million |
| **Average Customer Value** | £3,019 |
| **Features Engineered** | 52 |
| **RFM Segments** | 11 |
| **ML Clusters** | 9 (6 RFM-enhanced + 3 behavioral) |
| **Customers ML-Scored** | 5,878 (CLV + Churn + Purchase Window) |
| **Churn Recall @ deployment threshold** | 99.5% |
| **CLV Model R²** | 0.882 (leakage-free) |

---

## 🎯 Business Objectives

1. **Customer Segmentation** — Identify actionable customer groups based on purchasing behavior
2. **Churn Prevention** — Flag high-value customers at risk of disengagement
3. **Revenue Optimization** — Surface cross-sell opportunities via market basket analysis
4. **Campaign Targeting** — Generate segment-specific marketing strategies with concrete action plans
5. **Predictive Intelligence** — Score every customer by future value, churn probability, and next purchase window

---

## 📁 Project Structure

```
customer-segmentation/
│
├── data/
│   ├── raw/
│   │   ├── online_retail.csv               # Original UCI dataset (not modified)
│   │   └── data_sample.csv
│   ├── processed/
│   │   ├── data_cleaned.csv
│   │   ├── data_sample_cleaned.csv
│   │   └── rfm_customer_scores.csv
│   └── features/
│       ├── features_master.csv             # 52 features × 5,878 customers
│       ├── features_with_rfm_clusters.csv
│       └── features_with_behavioral_clusters.csv
│
├── notebooks/
│   ├── 00_data_sample_generator.ipynb
│   ├── 01_eda_data_foundation.ipynb
│   ├── 02_rfm_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_track1_rfm_enhanced_clustering.ipynb
│   ├── 04_track2_behavioral_clustering.ipynb
│   ├── 05_temporal_pattern_analysis.ipynb
│   └── 06_market_basket_analysis.ipynb
│
├── src/
│   ├── models/
│   │   └── 08_predictive_models.py         # CLV + Churn + Purchase Window
│   └── utils/
│       ├── 07_insight_engine.py            # Automated alerts & anomaly detection
│       └── 09_dashboard.py                 # BI dashboard
│
├── outputs/
│   ├── campaigns/
│   │   ├── ml_scored_customers.csv         # 5,878 customers with all ML scores
│   │   ├── segment_champions.csv
│   │   ├── segment_cant_lose.csv
│   │   ├── segment_at_risk.csv
│   │   ├── segment_potential_loyalists.csv
│   │   ├── segment_daily_patterns.csv
│   │   └── segment_hourly_patterns.csv
│   ├── figures/
│   │   ├── customer_segments.png
│   │   ├── day_of_week_patterns.png
│   │   ├── fig6_1_category_performance.png
│   │   ├── fig6_2_copurchase_matrix.png
│   │   ├── fig6_4_category_pairs.png
│   │   ├── fig6_5_segment_category_heatmap.png
│   │   ├── fig8_1_clv_model.png            # Actual vs predicted, residuals, SHAP
│   │   ├── fig8_2_churn_model.png          # Reliability diagram, threshold sweep, SHAP
│   │   ├── fig8_3_purchase_window.png      # Confusion matrix, segment distribution, SHAP
│   │   └── fig8_4_campaign_dashboard.png   # CLV × churn scatter, priority matrix
│   ├── models/
│   │   ├── behavioral_kmeans.pkl
│   │   ├── behavioral_scaler.pkl
│   │   ├── behavioral_outlier_ids.pkl
│   │   ├── behavioral_feature_names.pkl
│   │   ├── rfm_enhanced_kmeans.pkl
│   │   ├── rfm_enhanced_scaler.pkl
│   │   ├── rfm_enhanced_outlier_ids.pkl
│   │   └── rfm_enhanced_feature_names.pkl
│   └── reports/
│       ├── eda_summary.json
│       ├── rfm_executive_summary.json
│       ├── rfm_segment_profiles.csv
│       ├── rfm_enhanced_cluster_profiles.csv
│       ├── behavioral_cluster_profiles.csv
│       ├── hourly_purchase_patterns.csv
│       ├── daily_purchase_patterns.csv
│       ├── customer_timing_personas.csv
│       ├── insights_report.json
│       └── dashboard_20260218.html
│
├── project/
│   ├── decision_final.md                   # 34 documented technical decisions
│   ├── project_plan.md
│   └── project_considerations.md
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔬 Methodology

### Phase 1 — Data Foundation & EDA
- Processed **1M+ raw transactions** across the full 24-month period
- Resolved data quality issues: ~25% missing CustomerIDs, negative quantities (returns), £0 price entries, bulk order edge cases
- Built cohort analysis, revenue trend decomposition, and geographic distribution maps
- **Key finding:** UK accounts for ~85% of revenue; significant seasonality peaks in Q4

### Phase 2 — RFM Analysis & Customer Scoring
- Computed individual Recency, Frequency, and Monetary scores (quintile-based, 1–5 scale)
- Assigned 5,878 identified customers to **11 named business segments**
- Generated CRM-ready CSV exports for direct campaign targeting
- **Key finding:** 8% of customers (Champions) generate 47% of total revenue

### Phase 3 — Advanced Feature Engineering
- Engineered **52 features** across 4 dimensions:

| Category | Features |
|---|---|
| **Behavioral** | avg_basket_size, product_diversity, return_rate, is_bulk_buyer, unique_products |
| **Temporal** | preferred_hour, weekend_ratio, purchase_velocity, quarter_concentration |
| **Monetary** | CLV, AOV, spending_cv, revenue_growth, price_range |
| **Engagement** | loyalty_index, churn_risk_score, activity_rate, engagement_consistency |

### Phase 4 — Multi-Level Clustering
- **Track 1 (RFM-Enhanced):** 6 clusters combining RFM scores with behavioral signals — validated via Silhouette Score and Davies-Bouldin Index
- **Track 2 (Behavioral):** 3 clusters focusing on basket patterns, product diversity, and purchasing cadence
- Applied StandardScaler normalization; K-Means with elbow method + business validation; serialized all models to `.pkl` for production reuse

### Phase 5 — Temporal Pattern Analysis
- Mapped purchase behavior by hour-of-day and day-of-week across all segments
- Derived **customer timing personas** (Morning Planner, Business Hours Buyer, etc.)
- **Key finding:** 78% of revenue concentrates in Mon–Thu, 10am–3pm window — optimal campaign deployment times

### Phase 6 — Market Basket Analysis
- Analyzed category co-purchase patterns across 5,878 customers
- Built co-purchase frequency matrix for top 10 product categories
- Identified highest-affinity category pairs for bundling and cross-sell strategy
- **Key finding:** HOME DECOR & OTHER + METAL SIGN is the strongest co-purchase pair (20,047 co-occurrences)

### Phase 7 — Automated Insight Engine
- Built threshold-based alert system for segment migrations, churn warnings, and revenue anomalies
- Applied Isolation Forest for unusual customer behavior detection and Mann-Kendall test for category-level trend signals
- Output: `insights_report.json` with structured, actionable alerts ready for downstream consumption

### Phase 8 — Predictive Models

Three production-grade LightGBM models trained on explicitly leakage-free feature sets:

**Customer Lifetime Value (Regression)**
- Target: log-transformed historical CLV
- Feature set: 27 behavioral features — monetary-derived variables excluded after detecting perfect correlation (r = 1.000) with the target
- Split: tenure-based temporal proxy (75/25), not random — mimics inference on newer customers at deployment time
- The model predicts revenue from *how* a customer shops, not from how much they have already spent. R² drops from 0.991 → 0.882 after leakage removal. This is the honest number.

**Churn Prediction (Binary Classification)**
- Target: `is_at_risk` binary flag
- Feature set: 35 features — recency-derived leakage variables excluded (`churn_risk_score`, `days_overdue`, `days_since_last`, `is_at_risk`)
- Threshold tuned to 0.22 (vs. LightGBM default 0.50): business logic is to maximize recall while maintaining precision ≥ 0.35
- 1,551 customers flagged for intervention at deployment threshold

**Next Purchase Window (3-class Classification)**
- Classes: Active (<30 days), Warming (31–90 days), Dormant (>90 days)
- Maps to three campaign archetypes: routine reorder, re-engagement nudge, win-back
- Stratified random split — appropriate for a current-state classifier, not a future predictor. ROC-AUC = 1.0 is expected and documented: the target is derived from Recency, which is included as a feature.

### Phase 9 — BI Dashboard
- Interactive dashboard (`dashboard_20260218.html`) covering executive KPIs, segment deep-dives, temporal heatmaps, and ML-scored customer action center
- Deployable via `09_dashboard.py`

---

## 📊 Key Results

### Segment Distribution

| Segment | Customers | Revenue Share | Avg. Monetary | Action |
|---|---|---|---|---|
| **Champions** | 471 (8%) | **46.96%** | £17,692 | Retain & reward |
| **Loyal Customers** | 979 (17%) | 22.02% | £3,991 | Upsell & deepen |
| **Lost** | 797 (14%) | 13.37% | £2,977 | Win-back campaigns |
| **Can't Lose Them** | 227 (4%) | 5.74% | £4,488 | Urgent reactivation |
| **Potential Loyalists** | 1,280 (22%) | 3.80% | £526 | Frequency programs |
| **At Risk** | 502 (9%) | 2.69% | £952 | Churn intervention |
| **About To Sleep** | 606 (10%) | 2.72% | £795 | Re-engagement |

### Predictive Model Performance

| Model | Metric | Value | Notes |
|---|---|---|---|
| **CLV** | R² (test set) | 0.882 | Leakage-free — no monetary features |
| **CLV** | CV R² (5-fold) | 0.918 ± 0.013 | |
| **CLV** | Median APE | 24.8% | Log-scale target |
| **CLV** | MAE / RMSE | £659 / £4,613 | Driven by extreme-value customers |
| **Churn** | ROC-AUC | 0.959 | |
| **Churn** | PR-AUC | 0.727 | More reliable than ROC for imbalanced classes |
| **Churn** | Brier Score | 0.082 | Skill = 0.545 vs. naive baseline |
| **Churn** | Recall @ θ=0.22 | **0.995** | Primary business metric |
| **Purchase Window** | ROC-AUC (weighted) | 1.000 | Expected — see decision log |

### Market Basket Highlights

- **Strongest cross-sell pair:** HOME DECOR & OTHER ↔ METAL SIGN (20,047 co-purchases)
- **Broadest reach category:** HOME DECOR & OTHER appears in baskets of 5,200+ unique customers
- **Segment differentiation:** At Risk customers over-index on HOME DECOR & OTHER (37.3%) — category-specific win-back offers are highest priority
- **Untapped opportunity:** BAG category reaches ~4,100 customers but is under-represented in Potential Loyalists (5.7%) vs. Champions (10.6%)

---

## 💡 Business Recommendations

### Immediate Actions (0–30 days)
- **"Can't Lose Them" reactivation** — 227 customers, avg. £4,488 spend, 340+ days inactive; ~£1M in recoverable revenue
- **Champions VIP Program** — 471 customers generating £8.3M; exclusive access and relationship management
- **ML churn list activation** — 1,551 flagged customers (99.5% recall); prioritize the subset with CLV score in top quartile

### Medium-Term (30–90 days)
- **Send-time optimization** — Deploy campaigns Mon–Thu, 10am–2pm based on confirmed purchase peaks
- **Potential Loyalists frequency program** — 1,280 customers at avg. 2 purchases; tiered loyalty mechanics to accelerate progression
- **Cross-sell bundles** — HOME DECOR + METAL SIGN, HOME DECOR + STORAGE, HEART/LOVE + HOME DECOR; all validated by co-purchase frequency

### Strategic (90+ days)
- **CLV-based budget allocation** — Re-weight marketing spend by predicted lifetime value, not last-purchase recency
- **Purchase window targeting** — Use Active / Warming / Dormant labels to select campaign type per customer
- **Quarterly model refresh** — Re-score all customers every 90 days; monitor segment migration rates as leading revenue indicators

---

## ⚠️ Documented Limitations

**Cold-start (churn model):** Single-transaction customers (27.6% of base, n=1,623) receive near-uniform churn probabilities (~0.105). Five cadence features are undefined for one-time buyers, making them indistinguishable. This is correct model behavior — a customer who has bought once has not yet demonstrated a pattern. A production pipeline should add a `prediction_confidence` tier (New → Cold Start → Emerging → Established) gating model output before it reaches campaign systems.

**Temporal proxy split:** Train/test uses tenure as a proxy for time, not a true temporal split re-engineering features at two cut-off dates from raw transactions. Results should be interpreted with this constraint in mind.

---

## 🛠️ Tech Stack

```python
# Core
pandas, numpy, scipy

# Machine Learning
scikit-learn          # Clustering, preprocessing, validation
lightgbm              # CLV, churn, and purchase window models
mlxtend               # Market basket / association rules

# Explainability
shap                  # Feature importance for all three predictive models

# Visualization
matplotlib, seaborn   # Static charts
plotly                # Interactive dashboard

# Dimensionality Reduction
umap-learn            # Cluster visualization
```

---

## ⚙️ Reproducibility

```python
RANDOM_SEED = 42
REFERENCE_DATE = "2011-12-10"          # RFM recency anchor (last date + 1 day)
CLV_TRAIN_TENURE_THRESHOLD = 540       # Days — temporal proxy split boundary
CHURN_DECISION_THRESHOLD = 0.22        # Tuned for Recall ≥ 0.99, Precision ≥ 0.35
```

All notebooks run end-to-end on the UCI Online Retail II dataset at [UCI ML Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii).

```bash
pip install -r requirements.txt

# Run notebooks sequentially
jupyter nbconvert --to notebook --execute notebooks/01_eda_data_foundation.ipynb
# ... through 06_market_basket_analysis.ipynb

# Run predictive models
python src/models/08_predictive_models.py

# Launch dashboard
python src/utils/09_dashboard.py
```

---

## 📐 Design Principles

**1. Business-First** — Every technical decision is documented with a business rationale. Threshold tuning, feature exclusions, and split strategy are all driven by what the output needs to do downstream.

**2. Transparent Decisions** — 34 methodological decisions are logged in `DECISIONS.md` with explicit options considered, trade-offs accepted, and reversibility assessments — including decisions to *not* do things (collaborative filtering dropped in favor of basket analysis; probability calibration dropped after it degraded Brier score 2.5×).

**3. Leakage-Free Modeling** — Monetary-derived features removed from the CLV model after detecting perfect correlation with the target (R² dropped from 0.991 → 0.882 — the honest number). Recency-derived features removed from the churn model. Both exclusions are documented with full rationale.

**4. Production-Ready** — Serialized models, scored customer CSVs, a deployed insight engine, and a live HTML dashboard — not just notebooks with charts.

---

## 📈 Project Status

| Phase | Deliverable | Status |
|---|---|---|
| 1 — EDA | Data quality report + cohort analysis | ✅ Complete |
| 2 — RFM | 11-segment scoring system + CRM exports | ✅ Complete |
| 3 — Feature Engineering | 52 behavioral/temporal features | ✅ Complete |
| 4 — Clustering | Multi-layer segmentation (6+3 clusters) + serialized models | ✅ Complete |
| 5 — Temporal Analysis | Timing personas + send-time optimization | ✅ Complete |
| 6 — Market Basket | Category co-purchase matrix + cross-sell rules | ✅ Complete |
| 7 — Insight Engine | Automated alerts + anomaly detection | ✅ Complete |
| 8 — Predictive Layer | CLV + Churn + Purchase Window; 5,878 customers scored | ✅ Complete |
| 9 — Dashboard | Interactive BI dashboard (HTML + deployable script) | ✅ Complete |

---

## 📬 About

This project demonstrates applied ML for business value generation, prioritizing interpretable models, rigorous leakage prevention, and executive-facing outputs over technical complexity for its own sake. Every analytical choice has a documented business rationale.

**Dataset:** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) — Chen, D., Sain, S.L., & Guo, K. (2012). Data mining for the online retail industry. *Journal of Database Marketing and Customer Strategy Management.*
