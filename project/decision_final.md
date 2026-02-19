# 📊 PROJECT DECISIONS LOG
## Intelligent Customer Segmentation & Revenue Optimization

---

### 📋 **PROJECT METADATA**

| Attribute | Value |
|-----------|-------|
| **Project Name** | Online Retail Intelligence System |
| **Dataset** | UCI Online Retail II Dataset |
| **Business Domain** | B2B Wholesale Retail (UK-based) |
| **Analysis Period** | Dec 2009 - Dec 2011 (24 months) |
| **Primary Stakeholder** | CMO / Head of Marketing |
| **Success Metric** | Revenue lift from targeted campaigns >15% |
| **Project Start Date** | [YYYY-MM-DD] |
| **Last Updated** | 2026-02-18 |

---

## 🎯 **BUSINESS OBJECTIVES**

### Primary Goals
1. **Customer Segmentation**: Identify 4-8 actionable customer segments based on purchasing behavior
2. **Churn Prevention**: Predict which high-value customers are at risk of churning
3. **Revenue Optimization**: Identify cross-sell/upsell opportunities through market basket analysis
4. **Campaign Targeting**: Generate automated recommendations for personalized marketing

### Success Criteria
- [ ] Each segment must have >500 customers (minimum viable campaign size)
- [ ] Inter-segment CLV variance >40% (meaningful differentiation)
- [ ] Recommendation system achieves >15% lift vs. random baseline
- [ ] Insights are actionable within 2 weeks (no complex implementation required)

### Out of Scope (for this phase)
- ❌ Real-time recommendation engine (batch processing only)
- ❌ Integration with CRM/email platforms
- ❌ Pricing optimization
- ❌ Inventory forecasting

---

## 📊 **DATA UNDERSTANDING**

### Dataset Characteristics
| Metric | Value | Notes |
|--------|-------|-------|
| **Total Rows** | [TBD after EDA] | |
| **Unique Customers** | [TBD] | |
| **Date Range** | 2009-12-01 to 2011-12-09 | 24 months |
| **Countries** | [TBD] | UK-centric? |
| **Product SKUs** | [TBD] | |
| **Missing CustomerID** | [TBD]% | **CRITICAL METRIC** |

### Known Data Quality Issues
⚠️ **Issue**: ~25% of transactions lack CustomerID  
📝 **Decision**: [PENDING - options: exclude, impute using heuristics, or analyze separately]  
💼 **Business Impact**: Cannot segment ~25% of revenue  
✅ **Mitigation**: Focus on identified customers; report coverage metrics to stakeholders

⚠️ **Issue**: Negative quantities indicate returns/cancellations  
📝 **Decision**: [PENDING]  
💼 **Business Impact**: [TBD]  
✅ **Mitigation**: [TBD]

⚠️ **Issue**: Some unit prices ≤ £0  
📝 **Decision**: [PENDING]  
💼 **Business Impact**: [TBD]  
✅ **Mitigation**: [TBD]

---

## 🔧 **TECHNICAL DECISIONS**

### 1️⃣ **DATA PREPROCESSING**

#### Missing Value Strategy
| Column | Missing % | Strategy | Rationale |
|--------|-----------|----------|-----------|
| CustomerID | [TBD]% | [PENDING] | Cannot segment without ID |
| Description | [TBD]% | [PENDING] | Needed for basket analysis |
| [Other cols] | | | |

**DECISION TEMPLATE:**
```
DECISION #001: Handling Missing CustomerID
├─ Options Considered:
│  ├─ A) Exclude all rows without CustomerID
│  ├─ B) Create "Guest" segment for anonymous transactions
│  └─ C) Impute using IP/session heuristics (requires additional data)
├─ Choice: [A/B/C]
├─ Rationale: [Explain why]
├─ Trade-offs: [What we're sacrificing]
└─ Validation: [How we'll verify this was correct]
```

#### Outlier Treatment
**DECISION #002: Extreme Unit Prices**
- **Threshold**: [TBD - e.g., >99.9th percentile or >£1000]
- **Action**: [Keep / Cap / Investigate manually]
- **Rationale**: [Could be luxury items (legitimate) vs. data errors]
- **Validation**: Manual review of top 50 highest-priced items

**DECISION #003: Bulk Purchases (Quantity >100)**
- **Action**: [TBD]
- **Rationale**: B2B context makes bulk orders normal, not outliers
- **Feature Engineering**: Create `is_bulk_buyer` flag instead of removing

#### Return/Cancellation Handling
**DECISION #004: Negative Quantities**
- **Options**:
  - A) Exclude entirely (simplest)
  - B) Create separate "return_rate" feature
  - C) Net out returns from original purchases
- **Choice**: [PENDING]
- **Impact on Metrics**: [How this affects CLV, frequency calculations]

---

### 2️⃣ **FEATURE ENGINEERING**

#### RFM Calculation
**DECISION #005: RFM Reference Date**
- **Date Chosen**: [Last date in dataset + 1 day, e.g., 2011-12-10]
- **Rationale**: Standard practice for historical analysis
- **Alternative Considered**: Use dynamic "today" (rejected - not reproducible)

**DECISION #006: RFM Binning Strategy**
- **Method**: Quintiles (5 bins per metric) vs. Business-rule thresholds
- **Choice**: [PENDING after EDA]
- **Adaptation for B2B**:
  ```python
  # B2B typically has:
  # - Longer purchase cycles (adjust Recency thresholds)
  # - Lower frequency (monthly vs. weekly)
  # - Higher monetary values (bulk orders)
  
  # Proposed thresholds (to be validated):
  recency_bins = [0, 30, 90, 180, 365, inf]  # days
  frequency_bins = [1, 3, 6, 12, 24, inf]    # purchases
  monetary_bins = [0, 500, 2000, 5000, 20000, inf]  # GBP
  ```

#### Temporal Features
**DECISION #007: Weekend Shopper Definition**
- **Threshold**: >50% of purchases on Sat/Sun = True
- **Rationale**: Clear behavioral split for targeting
- **Alternative**: Use continuous ratio (rejected - harder to action)

**DECISION #008: Seasonality Detection**
- **Method**: [Monthly dummies vs. Fourier features vs. STL decomposition]
- **Choice**: [PENDING]
- **Validation**: Visual inspection + business calendar alignment (Christmas, etc.)

#### Behavioral Features
**DECISION #009: Product Diversity Metric**
- **Formula**: `unique_categories / total_purchases` (concentration index)
- **Why not count**: Normalizes for purchase frequency
- **Business Use**: Identify upsell candidates (low diversity = opportunity)

**DECISION #010: Churn Risk Definition**
- **Churn Threshold**: No purchase in [60/90/120] days after expected next purchase
- **Calculation Method**: [TBD - based on avg. purchase cycle per customer]
- **Edge Case**: Seasonal customers (only buy in Q4) - need special handling

---

### 3️⃣ **MODELING DECISIONS**

#### Segmentation Approach
**DECISION #011: Number of Clusters**
- **Range to Test**: 4-10 clusters
- **Selection Criteria** (in priority order):
  1. Business actionability (can marketing create distinct campaigns?)
  2. Segment size (each >500 customers)
  3. Silhouette Score (>0.4 acceptable)
  4. Within-cluster variance (<30% of total variance)
- **Method**: Elbow + Silhouette + **manual business review**

**DECISION #012: Clustering Algorithm**
- **Primary**: K-Means (interpretable, fast, good for RFM data)
- **Validation**: Hierarchical clustering to verify natural groupings
- **Why not DBSCAN**: Requires density assumptions that may not hold; harder to explain

**DECISION #013: Feature Scaling**
- **Method**: StandardScaler (z-score normalization)
- **Why not MinMax**: RFM components have different distributions; want equal variance
- **Applied to**: All numeric features before clustering

#### Predictive Models
**DECISION #014: Churn Model Algorithm**
- **Options**: Logistic Regression, Random Forest, XGBoost
- **Selection Criteria**: 
  - Interpretability: HIGH (need to explain to marketing)
  - Performance: Recall >70% (false negatives are costly)
  - Calibration: Probabilities must be reliable (for scoring)
- **Choice**: [PENDING after baseline tests]

**DECISION #015: Train/Test Split Strategy**
- **Method**: Temporal split (NOT random)
- **Train**: [Months 1-18]
- **Validation**: [Months 19-21]
- **Test**: [Months 22-24]
- **Why temporal**: Prevents data leakage; mimics production scenario

**DECISION #016: Class Imbalance Handling** (for churn model)
- **Expected Ratio**: ~15% churn (minority class)
- **Options**: SMOTE, class weights, threshold tuning
- **Choice**: [PENDING]
- **Validation Metric**: F1-score (balances precision/recall)

#### Market Basket Analysis
**DECISION #017: Association Rule Thresholds**
- **Min Support**: [0.5%] (appears in 0.5% of baskets)
- **Min Confidence**: [30%] (30% of A buyers also buy B)
- **Min Lift**: [1.5] (50% more likely than random)
- **Rationale**: Balance between finding patterns and avoiding noise
- **Post-processing**: Manual review of top 50 rules for business sense

---

### 4️⃣ **VALIDATION & EVALUATION**

#### Model Performance Metrics
**DECISION #018: Primary Metrics by Model**
| Model Type | Primary Metric | Secondary Metrics | Business Threshold |
|------------|----------------|-------------------|-------------------|
| Segmentation | Silhouette Score | Davies-Bouldin, Segment Size | >0.4, each segment >500 |
| Churn Prediction | Recall | Precision, F1, AUC | Recall >70% |
| CLV Prediction | RMSE, MAPE | R², MAE | MAPE <20% |
| Recommendations | Lift | Click-through rate (if A/B tested) | Lift >1.5x |

**DECISION #019: Business Validation Process**
1. Statistical validation (metrics above)
2. Segment profile review (do personas make sense?)
3. Stakeholder walkthrough (can marketing use this?)
4. Pilot campaign on 1 segment (measure actual lift)

#### Robustness Checks
**DECISION #020: Sensitivity Analysis**
- **Test**: Re-run clustering with ±10% threshold changes
- **Expected**: Segment membership should be >80% stable
- **If not**: Indicates over-segmentation or weak signals

**DECISION #021: Holdout Validation**
- **Reserved**: Last 3 months of data (never touched during development)
- **Purpose**: Final reality check before deployment
- **Fail Condition**: If holdout performance drops >15%, revisit feature engineering

---

## ⚙️ **INFRASTRUCTURE & REPRODUCIBILITY**

#### Environment Setup
**DECISION #022: Python Version & Key Libraries**
```python
python==3.10.x
pandas==2.0.3
scikit-learn==1.3.0
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
mlxtend==0.22.0  # for market basket
```
**Why these versions**: Stability + compatibility tested

**DECISION #023: Random Seed Management**
```python
RANDOM_SEED = 42  # Set globally for reproducibility
np.random.seed(RANDOM_SEED)
# Applied to: train/test split, K-Means initialization
```

#### Code Organization
**DECISION #024: Project Structure**
```
online-retail-intelligence/
├── data/
│   ├── raw/                 # Never modified
│   ├── processed/           # After cleaning
│   └── features/            # Engineered features
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_rfm_analysis.ipynb
│   └── [sequential numbering]
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── models/
│   └── utils/
├── outputs/
│   ├── figures/
│   ├── models/              # Serialized models
│   └── reports/
├── DECISIONS.md             # This file
└── requirements.txt
```

**DECISION #025: Notebook vs. Scripts**
- **Notebooks**: Exploration, visualization, reporting
- **Scripts**: Reusable functions, production pipelines
- **Rule**: If used >2x, move to `src/`

---

## 🚨 **RISK REGISTER**

### Data Risks
| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| **>30% missing CustomerID** | High | High | Focus on identified customers; report coverage | [Name] |
| **Insufficient data for seasonal patterns** | Medium | Medium | Use 24 months; if still weak, exclude seasonality features | [Name] |
| **Dataset not representative of current behavior** | Medium | High | Validate with stakeholder; consider recent data pull | [Name] |

### Model Risks
| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| **Segments not actionable** | Medium | Critical | Involve marketing in segment naming/validation | [Name] |
| **Overfitting due to small dataset** | Medium | High | Use cross-validation; simpler models; regularization | [Name] |
| **Model degrades over time** | High | Medium | Set up monitoring; retrain quarterly | [Name] |

### Business Risks
| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| **Insights not implemented** | High | Critical | Deliver actionable playbook, not just models | [Name] |
| **Stakeholder expectations misaligned** | Medium | High | Weekly check-ins; demo early and often | [Name] |

---

## 📝 **ASSUMPTIONS & LIMITATIONS**

### Assumptions
1. **Stationarity**: Customer behavior patterns from 2009-2011 are still relevant
   - *Validation*: Compare with any available recent data
   - *Risk*: E-commerce trends may have shifted significantly

2. **Data Completeness**: Transactions in dataset represent all business activity
   - *Validation*: Check for gaps in date range
   - *Risk*: Missing channels (in-store, phone orders?)

3. **CustomerID Consistency**: Same ID = same customer across time
   - *Validation*: Check for impossible patterns (multiple countries same day)
   - *Risk*: Shared accounts or data entry errors

4. **B2B Context**: Majority of customers are resellers/wholesalers
   - *Validation*: Analyze quantity distributions, top buyers
   - *Risk*: Mix of B2B/B2C changes dynamics

### Known Limitations
1. **No Customer Demographics**: Age, gender, company size unknown
   - *Impact*: Cannot create demographic personas
   - *Workaround*: Use behavioral proxies (product preferences)

2. **No Marketing Attribution**: Don't know which campaigns drove purchases
   - *Impact*: Cannot measure campaign ROI retrospectively
   - *Workaround*: Focus on future campaign targeting

3. **Limited Geographic Data**: Only country-level location
   - *Impact*: Cannot do local market analysis
   - *Workaround*: UK vs. International segmentation

4. **Snapshot in Time**: Historical data, not real-time
   - *Impact*: Recommendations are batch, not live
   - *Workaround*: Set expectation for weekly/monthly refresh

---

## 🎯 **DECISION REVIEW CHECKPOINTS**

### After EDA (Week 1)
- [ ] Review all data quality decisions
- [ ] Finalize preprocessing pipeline
- [ ] Update risk register with findings
- [ ] Validate business objectives are still achievable

### After Feature Engineering (Week 2)
- [ ] Confirm RFM thresholds make business sense
- [ ] Review feature correlation matrix (multicollinearity?)
- [ ] Stakeholder review of feature definitions

### After Modeling (Week 3)
- [x] Compare all model performance to baselines
- [x] Business validation of segments
- [ ] Sensitivity analysis passed?
- [x] Document any pivot decisions

### Pre-Deployment (Week 4)
- [ ] Holdout set validation
- [ ] Code review & refactoring
- [ ] Documentation complete
- [ ] Stakeholder sign-off on insights

---

## 📊 **APPENDIX: DECISION TEMPLATES**

### Template for New Decisions
```markdown
**DECISION #XXX: [Decision Title]**
Date: [YYYY-MM-DD]
Decision Maker: [Name/Role]

Options Considered:
├─ A) [Option A description]
│  ├─ Pros: [List]
│  └─ Cons: [List]
├─ B) [Option B description]
│  ├─ Pros: [List]
│  └─ Cons: [List]

Choice: [A/B]

Rationale:
[Explain why this option was selected]

Trade-offs Accepted:
[What we're giving up with this choice]

Success Criteria:
[How we'll know this was the right decision]

Reversibility:
[Easy/Medium/Hard - can we change this later?]

Dependencies:
[What else depends on this decision?]

Related Decisions: #001, #005
```

---

## 🔄 **CHANGE LOG**

| Date | Decision # | Change | Reason | Approved By |
|------|------------|--------|--------|-------------|
| [YYYY-MM-DD] | #003 | Changed outlier threshold from 99th to 99.9th percentile | Too many valid luxury items excluded | [Name] |
| 2026-02-18 | #026–#033 | Phase 8 ML Predictive Layer decisions added | Phase completion | Senior DS |

---

## ✅ **SIGN-OFF**

**Technical Lead**: ________________________  Date: __________  
**Business Stakeholder**: __________________  Date: __________  
**Data Governance**: _______________________  Date: __________

---

**Document Control**
- Version: 1.1
- Status: Living Document (Updated Throughout Project)
- Next Review: [Date]
- Owner: [Senior Data Scientist Name]

---

---

# 🤖 PHASE 8: ML PREDICTIVE LAYER — DECISIONS LOG
**Date**: 2026-02-18  
**Phase**: 8 — ML Predictive Layer  
**Script**: `08_predictive_models_v2.py`  
**Outputs**: `phase8_ml_scored_customers.csv`, `fig8_1` through `fig8_4`

---

## PHASE 8 OVERVIEW

Three predictive models were built on top of the feature set from Phase 3 and cluster labels from Phase 4:

| Model | Type | Algorithm | Primary Metric | Result |
|-------|------|-----------|----------------|--------|
| CLV Prediction | Regression | LightGBM | R² (log scale) | 0.8825 |
| Churn Prediction | Binary Classification | LightGBM | ROC-AUC | 0.9593 |
| Next Purchase Window | 3-class Classification | LightGBM | ROC-AUC (weighted) | 1.000* |

*Expected — see Decision #031.

**Output**: 5,878 customers fully scored with `predicted_clv`, `churn_probability`, `purchase_window_label`, and a composite `campaign_priority_score` ready for CRM import.

---

## SCOPE CHANGES VS. ORIGINAL PLAN

**DECISION #026: Product Recommendation System — Dropped**
- **Original plan**: Collaborative filtering or content-based recommendation system
- **Choice**: Dropped entirely
- **Rationale**: Phase 6 (Market Basket Analysis) already delivers cross-sell intelligence via co-purchase matrix and category pair analysis. Adding collaborative filtering on the same transaction data would be redundant and would not meaningfully add to the deliverable. The dataset is also not in a session/interaction format suitable for collaborative filtering without significant re-engineering.
- **Trade-off**: Portfolio shows one fewer model type
- **Mitigation**: Phase 6 basket analysis already covers this business need with more interpretable outputs (specific product pairs and lift scores vs. opaque recommendation vectors)
- **Reversibility**: Medium — could be added in a future phase with proper session data

**DECISION #027: Next Purchase Date (Regression) → Next Purchase Window (3-class)**
- **Original plan**: Time-to-event regression or survival analysis to predict exact next purchase date
- **Choice**: 3-class classification — Active (<30d), Warming (31–90d), Dormant (>90d)
- **Rationale**: Continuous date prediction is too noisy on B2B data with irregular purchase cycles. A 3-class window maps directly to three campaign archetypes: routine reorder, re-engagement nudge, win-back. Marketing can act on a window classification; they cannot act on "this customer will buy in 47.3 days."
- **Trade-off**: Less precise than a continuous prediction
- **Business Value**: Higher — each class maps to a concrete campaign type with distinct budget and messaging
- **Reversibility**: Easy — survival analysis could be layered on later

---

## FEATURE ENGINEERING DECISIONS (LEAKAGE PREVENTION)

**DECISION #028: CLV Feature Set — Monetary-Derived Features Excluded**
- **Issue discovered**: `avg_basket_value` = `Monetary / Frequency` → correlation with CLV target = **1.000**. Same identity holds for `aov`, `aov_std`, `aov_max`, `aov_min`. Including these features means the model is shown the answer.
- **Features removed from CLV model**: `avg_basket_value`, `aov`, `aov_std`, `aov_max`, `aov_min`, `avg_item_price`, `avg_basket_size`, `avg_price`, `Monetary` (the target itself)
- **Also excluded**: `loyalty_index` (partially encodes monetary magnitude)
- **Impact**: CV R² drops from 0.991 → 0.918. This is the honest number.
- **Interpretation**: 88% of CLV variance is explained by purely behavioral signals — purchase cadence, product diversity, temporal preferences, return patterns. This is the stronger portfolio claim: the model predicts revenue from *how* a customer shops, not from how much they have already spent.
- **Final CLV feature count**: 27 features
- **Reversibility**: N/A — this was a correction, not a choice

**DECISION #029: Churn Feature Set — Recency-Derived Features Excluded**  
- **Excluded from Churn model**: `is_at_risk`, `churn_risk_score`, `days_overdue`, `days_since_last`, `Recency`
- **Rationale**: `is_at_risk` is the binary target — including it as a feature is direct target leakage. `churn_risk_score` and `days_overdue` were engineered from recency rules, making them proxies of `is_at_risk`. `Recency` and `days_since_last` encode the same information as the target (both capture "how long since last purchase").
- **What this means**: The model must infer churn risk from behavioral patterns (purchase cadence, inter-purchase gaps, velocity) rather than from raw time elapsed. This is the production-valid approach — at inference time, Recency would be computed fresh anyway.
- **Also excluded**: `loyalty_index` (partially encodes recency signals), `purchase_frequency` (duplicate of `Frequency`)
- **Final Churn feature count**: 35 features
- **Reversibility**: N/A — correction

**DECISION #030: Churn features include monetary signals (unlike CLV model)**
- **Rationale**: For churn prediction, `Monetary`, `avg_basket_value`, `aov` are legitimate predictors. High-value customers with a sudden drop in order size is a churn signal. The target is a binary label, not Monetary itself, so there is no leakage.
- **Contrast with CLV**: In the CLV model, these features ARE the target (Monetary = CLV). In the Churn model, they are correlated predictors of a distinct binary outcome.

---

## MODELING DECISIONS

**DECISION #031: Algorithm — LightGBM for all three models**
- **Options considered**: Logistic Regression, Random Forest, XGBoost, LightGBM
- **Choice**: LightGBM across all three
- **Rationale**: Handles mixed feature types natively (no need to one-hot encode `preferred_day`, `preferred_quarter`), built-in early stopping prevents overfitting, fast enough for iterative experimentation, `scale_pos_weight` handles class imbalance cleanly, SHAP integration is native.
- **Why not XGBoost**: LightGBM is faster and achieves equivalent or better results on tabular data at this scale
- **Why not Random Forest**: No early stopping, slower, and LightGBM consistently outperformed it in initial baseline tests
- **Reversibility**: Easy — models are interchangeable behind the same feature pipeline

**DECISION #032: Train/Test Split — Temporal Proxy for CLV and Churn; Stratified Random for NPW**

*CLV and Churn (temporal proxy split):*
- **Method**: Customers with `tenure_days >= 540` (≈18 months) form the pool of "established" customers. 75% of them are sampled for training; the remaining 25% plus all customers with `tenure_days < 540` form the test set.
- **Rationale**: Mimics inference on newer customers at deployment time. Established customers with long histories train the model; it is tested on shorter-tenure customers who joined the dataset later.
- **Train**: 2,161 customers | median tenure 669d
- **Test**: 3,717 customers | median tenure 396d
- **Limitation**: This is a proxy — the ideal approach would re-engineer all features at two separate temporal cut-off dates from raw transactions. That was out of scope for this phase.

*Next Purchase Window (stratified random split):*
- **Method**: Standard `train_test_split` with `stratify=y`, 75/25
- **Rationale**: Purchase window is a **current-state classifier**, not a future predictor. The target (Active / Warming / Dormant) describes what state the customer is in today based on their Recency. A temporal split creates artificial distribution shift: older-tenure customers are structurally more Dormant, so a model trained on them fails on the Active-heavy test set. This is not a leakage problem — it is an inappropriate split type for the task.
- **Why AUC = 1.0 is expected and correct**: The target is derived from Recency, and Recency is included as a feature. The model learns a near-deterministic mapping. The business value is not in the AUC but in (a) the probability scores at class boundaries (customers with Recency 28–35 days who could go either way) and (b) the campaign window labels used downstream.

**DECISION #033: Probability Calibration — Dropped; Raw LightGBM Probabilities Used**
- **Approach tested**: `CalibratedClassifierCV` with both sigmoid and isotonic methods, cv=3 and cv=5
- **Result**: All configurations degraded Brier score from 0.082 (raw) to ~0.22 (calibrated) — 2.5x worse
- **Root cause**: The wrapper runs its own internal k-fold that ignores our temporal proxy split. The calibration mapping is learned on a distribution of older-tenure customers and does not transfer to the test set's mixture of newer customers. This is a known failure mode of `CalibratedClassifierCV` when the outer split is non-random.
- **Decision**: Use raw LightGBM probabilities
- **Validation**: Brier skill score = 0.545 vs. naive baseline (always predict class prior). The reliability diagram (fig8_2) shows the raw model is already well-calibrated without a wrapper.
- **Reversibility**: Easy — if a future retraining uses a different split strategy, calibration could be re-evaluated

---

## THRESHOLD DECISION (CHURN MODEL)

**DECISION #034: Churn Decision Threshold = 0.22**
- **Default threshold**: 0.50 (LightGBM default)
- **Chosen threshold**: 0.22
- **Selection logic**: Sweep thresholds from 0.10 to 0.90. Among all thresholds where Precision ≥ 0.35, select the one that maximises Recall.
- **Business rationale**: A missed at-risk customer (false negative) is more costly than a wasted campaign touch (false positive). The 0.35 precision floor ensures the flag list is not so long as to be operationally unmanageable for a marketing team.
- **Result at θ = 0.22**: Recall = 0.995, Precision = 0.43
- **Customers flagged**: 1,551 of 5,878 (26.4%)
- **Reversibility**: Easy — threshold is applied post-inference and can be changed without retraining

---

## COLD-START LIMITATION (DOCUMENTED FOR PRODUCTION AWARENESS)

Single-transaction customers (27.6% of the dataset, n=1,623) receive near-identical churn probabilities (~0.105, the model minimum). This is because five cadence features — `avg_inter_purchase_days`, `std_inter_purchase_days`, `spending_cv`, `revenue_growth`, `purchase_velocity` — are zero-filled for customers with only one transaction, making them indistinguishable from each other.

This is **correct model behavior**, not a bug. A customer who has bought once has not yet demonstrated a purchase pattern. The model correctly abstains from a high-confidence churn score.

**Production implication**: A `prediction_confidence` tier should be added to the output in a future production pipeline:

| Customer state | Transactions | Confidence |
|---|---|---|
| New | 0 | No prediction |
| Cold start | 1–2 | Low |
| Emerging | 3–5 | Medium |
| Established | 6+ | Full |

This was not implemented in Phase 8 as it is a production pipeline concern, not a modeling concern.

---

## PHASE 8 FINAL METRICS

| Model | Metric | Value | Notes |
|-------|--------|-------|-------|
| CLV | R² (log, test set) | 0.8825 | Honest — no monetary features |
| CLV | CV R² (5-fold) | 0.9179 ± 0.013 | |
| CLV | Median APE | 24.8% | £ predictions on log-scale target |
| CLV | MAE / RMSE | £659 / £4,613 | Driven by extreme-value customers |
| Churn | ROC-AUC | 0.9593 | |
| Churn | PR-AUC | 0.7266 | More reliable than ROC for imbalanced data |
| Churn | Brier score | 0.0818 | Skill = 0.545 vs. naive baseline |
| Churn | Recall @ θ=0.22 | 0.995 | Primary business metric |
| NPW | ROC-AUC (weighted) | 1.000 | Expected — see Decision #032 |

---

## OUTPUTS PRODUCED

| File | Description |
|------|-------------|
| `08_predictive_models_v2.py` | Full modeling script with all decisions documented inline |
| `phase8_ml_scored_customers.csv` | 5,878 customers scored across all three models |
| `fig8_1_clv_model.png` | Actual vs predicted, residuals, SHAP importance |
| `fig8_2_churn_model.png` | Reliability diagram, threshold tuning, SHAP importance |
| `fig8_3_purchase_window.png` | Confusion matrix, segment distribution, SHAP importance |
| `fig8_4_campaign_dashboard.png` | CLV vs churn scatter, priority by segment, CLV × window heatmap |
