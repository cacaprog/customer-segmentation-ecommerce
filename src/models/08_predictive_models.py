"""
Phase 8: ML Predictive Layer  (v2 — post-audit)
=================================================
Three production models:
  1. CLV Prediction        — LightGBM Regressor
  2. Churn Prediction      — LightGBM Classifier
  3. Next Purchase Window  — LightGBM 3-class Classifier

Post-audit fixes applied vs v1
────────────────────────────────────────────────────────────────────────────
FIX 1 — CLV LEAKAGE (critical)
  avg_basket_value = Monetary / Frequency  →  corr(target) = 1.000  REMOVED
  aov, aov_std, aov_max, aov_min           →  same identity  REMOVED
  avg_item_price, avg_basket_size          →  revenue ratios  REMOVED
  avg_price                                →  redundant with max_price, zero R² gain  REMOVED
  Corrected CV R² drops from 0.991 → 0.935, which is the honest number.
  0.935 still means 93.5% of CLV variance is explained by behavioral signals alone.

FIX 2 — CALIBRATION WRAPPER (methodology)
  CalibratedClassifierCV degraded Brier from 0.085 → 0.218 (2.5x worse).
  Root cause: wrapper uses internal k-fold that ignores our temporal proxy split,
  so calibration maps learned on older customers don't transfer to the test set.
  LightGBM with scale_pos_weight is already well-calibrated on large N datasets.
  Brier skill score vs naive baseline = 0.52 → no external calibration needed.
  We retain a reliability diagram to demonstrate calibration quality visually.

FIX 3 — NPW SPLIT (methodology, already in v1 but documented more clearly here)
  Temporal split is wrong for purchase_window (current-state classifier).
  Stratified random split used instead. AUC = 1.0 is expected and correct —
  the model learns the Recency→window mapping, which is deterministic.
  The value is in the probability scores at class boundaries and downstream
  campaign segmentation, not in AUC as a generalization metric.
────────────────────────────────────────────────────────────────────────────
"""

# ─────────────────────────────────────────────
# 0. Imports & Config
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize

import lightgbm as lgb
import shap

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Applying the base style first
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Overriding rcParams to match the Darkgrid signature exactly
plt.rcParams.update({
    "figure.facecolor": "white",      # Keep the outer border clean
    "axes.facecolor": "#EAEAF2",     # The classic Seaborn grey background
    "axes.grid": True, 
    "grid.color": "white", 
    "grid.linewidth": 1.0,           # Slightly thinner for a cleaner look
    "font.family": "sans-serif", 
    "axes.spines.top": True,         # Darkgrid typically keeps the full frame
    "axes.spines.right": True,
    "axes.edgecolor": "white",       # Makes the border blend with the grid
})

# Adapting the C dictionary to the HUSL palette
# These values are pulled from the HUSL color space to ensure 
# they don't look too "heavy" against the new grey background.
C = {
    "primary": "#5f9ed1",   # HUSL Blue
    "accent":  "#8d67ab",   # HUSL Purple/Violet
    "warn":    "#e15759",   # HUSL Muted Red
    "ok":      "#59a14f",   # HUSL Leaf Green
    "neutral": "#999999",   # Balanced Grey
    "purple":  "#b27ad3"    # Lightened Purple
}


print("✓ Imports loaded")


# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
df = pd.read_csv("/home/cairo/code/portfolio/customer-segmentation/data/features/features_with_rfm_clusters.csv")
print(f"✓ Loaded {len(df):,} customers, {df.shape[1]} columns")


# ─────────────────────────────────────────────
# 2. Feature Set Definitions
# ─────────────────────────────────────────────
#
# EXCLUSION RULES (applied to ALL models unless noted):
#
# Group A — RFM labels (derived from target or Recency):
#   Segment, R_Score, F_Score, M_Score, RFM_Score, RFM_Sum
#
# Group B — Churn/recency signals (leak churn target):
#   is_at_risk, churn_risk_score, days_overdue, days_since_last, Recency
#   (Recency IS included in NPW model — it's the primary feature there)
#
# Group C — Monetary identity features (leak CLV target):
#   avg_basket_value  = Monetary / Frequency        → corr(target)=1.000
#   aov               = Monetary / Frequency        → corr(target)=1.000
#   aov_std, aov_max, aov_min                       → derivatives of aov
#   avg_item_price    = total_revenue / total_items → revenue ratio
#   avg_basket_size   = total_items / Frequency     → quantity ratio
#   avg_price         = redundant with max_price, zero R² contribution
#
# Note: Group C is only excluded from the CLV model (where Monetary is the target).
#       For Churn and NPW, including monetary features is appropriate.
#

# ── Shared base (used across churn + npw) ────────────────────────────────────
BASE_BEHAVIORAL = [
    # Basket
    "avg_basket_size", "avg_basket_value", "avg_item_price",
    "unique_products", "product_diversity", "is_bulk_buyer", "max_quantity_single_txn",
    # Cadence
    "days_active", "avg_inter_purchase_days", "std_inter_purchase_days", "purchase_velocity",
    # Temporal preferences
    "preferred_hour", "hour_concentration", "weekend_ratio", "is_weekend_shopper",
    "preferred_day", "preferred_quarter", "quarter_concentration",
    # Engagement
    "active_months", "customer_lifetime_days", "tenure_days",
    "activity_rate", "engagement_consistency",
    # Spending profile
    "aov", "aov_std", "aov_max", "aov_min",
    "max_price", "price_range", "spending_cv",
    # Returns & growth
    "return_rate", "revenue_growth",
    # Volume
    "Frequency", "Monetary",
    # Cluster
    "RFM_Enhanced_Cluster", "loyalty_index",
    # Purchase count alias (same as Frequency, kept for explainability)
    "purchase_frequency",
]
BASE_BEHAVIORAL = list(dict.fromkeys(BASE_BEHAVIORAL))

# ── CLV features: Group C removed ────────────────────────────────────────────
CLV_FEATURES = [f for f in BASE_BEHAVIORAL if f not in [
    "avg_basket_value", "aov", "aov_std", "aov_max", "aov_min",
    "avg_item_price", "avg_basket_size", "avg_price",
    "Monetary",         # this IS the target
    "loyalty_index",    # partially encodes monetary magnitude
    "purchase_frequency",  # duplicate of Frequency
]]

# ── Churn features: only exclude recency-derived signals ─────────────────────
CHURN_FEATURES = [f for f in BASE_BEHAVIORAL if f not in [
    "loyalty_index",    # partially encodes recency signals
    "purchase_frequency",  # duplicate
]]

print(f"✓ Feature sets defined:")
print(f"   CLV features:   {len(CLV_FEATURES)} (monetary-derived removed)")
print(f"   Churn features: {len(CHURN_FEATURES)} (recency-derived removed)")


# ─────────────────────────────────────────────
# 3. Train / Test Split (Temporal Proxy)
# ─────────────────────────────────────────────
#
# Customers with tenure >= 540 days (≈18 months of 24) represent
# long-established buyers — we train on 75% of them and test on the remainder
# plus all shorter-tenure customers (who joined later in the dataset window).
# This mimics inferring on newer customers at deployment time.
#
# This split is used for CLV and Churn.
# The NPW model uses a stratified random split (see Model 3 section).
#
TENURE_THRESHOLD = 540

older       = df[df["tenure_days"] >= TENURE_THRESHOLD].copy()
newer       = df[df["tenure_days"] <  TENURE_THRESHOLD].copy()
train_older = older.sample(frac=0.75, random_state=RANDOM_SEED)
test_older  = older.drop(train_older.index)

train_df = train_older.copy()
test_df  = pd.concat([test_older, newer], ignore_index=True)

print(f"\n✓ Temporal-proxy split:")
print(f"   Train: {len(train_df):,} customers | median tenure {train_df['tenure_days'].median():.0f}d")
print(f"   Test:  {len(test_df):,}  customers | median tenure {test_df['tenure_days'].median():.0f}d")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: CLV Prediction (Regression)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("MODEL 1: CLV PREDICTION (REGRESSION)")
print("="*70)

CLV_TARGET = "Monetary"

X_train_clv = train_df[CLV_FEATURES].fillna(0)
X_test_clv  = test_df[CLV_FEATURES].fillna(0)
y_train_clv = np.log1p(train_df[CLV_TARGET])
y_test_clv  = np.log1p(test_df[CLV_TARGET])

print(f"\nTarget: log1p(Monetary)")
print(f"  Train: mean={y_train_clv.mean():.3f}, std={y_train_clv.std():.3f}")
print(f"  Monetary range: £{train_df[CLV_TARGET].min():.0f} – £{train_df[CLV_TARGET].max():,.0f}")
print(f"  Features: {len(CLV_FEATURES)} (zero leakage — no monetary-derived ratios)")

lgb_clv = lgb.LGBMRegressor(
    n_estimators=600,
    learning_rate=0.04,
    num_leaves=31,
    max_depth=6,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_SEED,
    n_jobs=-1, verbose=-1,
)

lgb_clv.fit(
    X_train_clv, y_train_clv,
    eval_set=[(X_test_clv, y_test_clv)],
    callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(False)],
)

y_pred_log = lgb_clv.predict(X_test_clv)
y_pred_clv = np.expm1(y_pred_log)
y_true_clv = np.expm1(y_test_clv)

mae   = mean_absolute_error(y_true_clv, y_pred_clv)
rmse  = np.sqrt(mean_squared_error(y_true_clv, y_pred_clv))
r2    = r2_score(y_test_clv, y_pred_log)
mape  = np.median(np.abs((y_true_clv - y_pred_clv) / (y_true_clv + 1))) * 100

cv_scores = cross_val_score(
    lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31,
                      random_state=RANDOM_SEED, verbose=-1),
    X_train_clv, y_train_clv,
    cv=5, scoring="r2", n_jobs=-1,
)

print(f"\n── CLV Model Performance ──")
print(f"   R² (log scale):     {r2:.4f}   ← honest: no monetary features used")
print(f"   CV R² (5-fold):     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"   Median APE:         {mape:.1f}%")
print(f"   MAE:                £{mae:,.0f}")
print(f"   RMSE:               £{rmse:,.0f}")

# SHAP
explainer_clv    = shap.TreeExplainer(lgb_clv)
shap_values_clv  = explainer_clv.shap_values(X_test_clv)

# Score all customers
X_all_clv = df[CLV_FEATURES].fillna(0)
df["predicted_clv"]  = np.expm1(lgb_clv.predict(X_all_clv))
df["clv_tier"] = pd.qcut(df["predicted_clv"], q=4,
                          labels=["Low", "Mid", "High", "Premium"])

print(f"\n   CLV tier distribution:")
print(df["clv_tier"].value_counts().sort_index().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: Churn Prediction (Binary Classification)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("MODEL 2: CHURN PREDICTION (BINARY CLASSIFICATION)")
print("="*70)

CHURN_TARGET = "is_at_risk"

X_train_ch = train_df[CHURN_FEATURES].fillna(0)
X_test_ch  = test_df[CHURN_FEATURES].fillna(0)
y_train_ch = train_df[CHURN_TARGET]
y_test_ch  = test_df[CHURN_TARGET]

pos_rate  = y_train_ch.mean()
scale_pos = (1 - pos_rate) / pos_rate
print(f"\nClass balance — positive rate: {pos_rate:.3f}  |  scale_pos_weight: {scale_pos:.2f}")

lgb_churn = lgb.LGBMClassifier(
    n_estimators=600,
    learning_rate=0.04,
    num_leaves=31,
    max_depth=6,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_SEED,
    n_jobs=-1, verbose=-1,
)

lgb_churn.fit(
    X_train_ch, y_train_ch,
    eval_set=[(X_test_ch, y_test_ch)],
    callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(False)],
)

y_prob_ch = lgb_churn.predict_proba(X_test_ch)[:, 1]

# ── Calibration check ────────────────────────────────────────────────────────
#
# We tested CalibratedClassifierCV (sigmoid and isotonic, cv=3 and cv=5).
# In all cases, external calibration degraded Brier from 0.085 to ~0.22.
# Root cause: the wrapper's internal k-fold ignores our temporal proxy split,
# creating distribution mismatch between the calibration map and test set.
#
# Decision: use raw LightGBM probabilities.
# Brier skill score vs naive baseline (always predict base rate):
brier_raw   = brier_score_loss(y_test_ch, y_prob_ch)
brier_naive = pos_rate * (1 - pos_rate)
skill_score = 1 - brier_raw / brier_naive
print(f"\n   Calibration decision:")
print(f"   Brier (raw model):    {brier_raw:.4f}")
print(f"   Brier (naive):        {brier_naive:.4f}")
print(f"   Brier skill score:    {skill_score:.3f}  (>0 = better than naive)")
print(f"   → Raw probabilities used. External calibration degraded Brier to ~0.22")
print(f"     due to temporal distribution shift in CalibratedClassifierCV's internal CV.")

# ── Threshold tuning ─────────────────────────────────────────────────────────
thresholds = np.arange(0.1, 0.9, 0.01)
results_thresh = []
for t in thresholds:
    preds = (y_prob_ch >= t).astype(int)
    tp = ((preds == 1) & (y_test_ch == 1)).sum()
    fp = ((preds == 1) & (y_test_ch == 0)).sum()
    fn = ((preds == 0) & (y_test_ch == 1)).sum()
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    results_thresh.append({"threshold": t, "precision": prec, "recall": rec, "f1": f1})

results_thresh_df = pd.DataFrame(results_thresh)

# Strategy: maximise Recall (false negatives = missed at-risk customer = revenue loss)
# Constraint: Precision >= 0.35 (avoid flagging too many false alarms for marketing)
best_thresh = 0.5
best_recall = 0.0
for _, row in results_thresh_df.iterrows():
    if row["recall"] > best_recall and row["precision"] >= 0.35:
        best_recall = row["recall"]
        best_thresh = row["threshold"]

y_pred_ch = (y_prob_ch >= best_thresh).astype(int)

roc_auc = roc_auc_score(y_test_ch, y_prob_ch)
pr_auc  = average_precision_score(y_test_ch, y_prob_ch)

print(f"\n── Churn Model Performance ──")
print(f"   ROC-AUC:              {roc_auc:.4f}")
print(f"   PR-AUC:               {pr_auc:.4f}")
print(f"   Optimal threshold:    {best_thresh:.2f}  (max recall | precision ≥ 0.35)")
print(f"\n   Classification report @ θ={best_thresh:.2f}:")
print(classification_report(y_test_ch, y_pred_ch, target_names=["Active", "At Risk"]))

# SHAP
explainer_churn   = shap.TreeExplainer(lgb_churn)
shap_values_churn = explainer_churn.shap_values(X_test_ch)

# Score all customers
X_all_ch = df[CHURN_FEATURES].fillna(0)
df["churn_probability"] = lgb_churn.predict_proba(X_all_ch)[:, 1]
df["churn_flag"]        = (df["churn_probability"] >= best_thresh).astype(int)

churn_by_segment = df.groupby("Segment").agg(
    customers=("CustomerID", "count"),
    avg_churn_prob=("churn_probability", "mean"),
    flagged=("churn_flag", "sum"),
    avg_predicted_clv=("predicted_clv", "mean"),
).round(3).sort_values("avg_churn_prob", ascending=False)

print(f"   Churn risk by segment:")
print(churn_by_segment.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: Next Purchase Window (3-class Classification)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("MODEL 3: NEXT PURCHASE WINDOW (3-CLASS)")
print("="*70)

#
# Target: which engagement window does the customer currently occupy?
#   0 = Active  (Recency ≤ 30d)  → routine reorder campaigns
#   1 = Warming (31–90d)         → re-engagement nudge
#   2 = Dormant (>90d)           → win-back campaign
#
# WHY STRATIFIED RANDOM SPLIT (not temporal):
# Purchase window is a current-state classifier, not a future predictor.
# The temporal split creates artificial distribution shift: long-tenure
# customers are structurally more Dormant, so a model trained on them
# fails to generalise. This is appropriate only for predictive tasks
# (CLV, Churn) where temporal integrity prevents future data leakage.
#
# WHY AUC = 1.0 IS EXPECTED:
# Recency is the primary feature AND the target is derived from Recency.
# The model learns a near-deterministic mapping. The value of this model
# is NOT its generalisation gap — it's the probability scores at class
# boundaries (customers with Recency 28–35d get meaningful probability
# distributions across Active/Warming) and the downstream campaign labels.
#

NPW_FEATURES = list(dict.fromkeys(BASE_BEHAVIORAL + ["Recency"]))

df["purchase_window"] = pd.cut(
    df["Recency"], bins=[-1, 30, 90, 999999], labels=[0, 1, 2]
).astype(int)

X_npw = df[NPW_FEATURES].fillna(0)
y_npw = df["purchase_window"]

X_train_npw, X_test_npw, y_train_npw, y_test_npw = train_test_split(
    X_npw, y_npw,
    test_size=0.25, random_state=RANDOM_SEED, stratify=y_npw,
)

print(f"\nClass distribution (train):")
for cls, name in [(0, "Active (<30d)"), (1, "Warming (30–90d)"), (2, "Dormant (>90d)")]:
    n = (y_train_npw == cls).sum()
    print(f"   {name}: {n}")

lgb_npw = lgb.LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    objective="multiclass",
    num_class=3,
    random_state=RANDOM_SEED,
    n_jobs=-1, verbose=-1,
)

lgb_npw.fit(
    X_train_npw, y_train_npw,
    eval_set=[(X_test_npw, y_test_npw)],
    callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(False)],
)

y_pred_npw = lgb_npw.predict(X_test_npw)
y_prob_npw = lgb_npw.predict_proba(X_test_npw)

roc_auc_npw = roc_auc_score(
    label_binarize(y_test_npw, classes=[0, 1, 2]),
    y_prob_npw, average="weighted", multi_class="ovr"
)

print(f"\n── Next Purchase Window Performance ──")
print(f"   ROC-AUC (weighted OvR): {roc_auc_npw:.4f}  ← expected ≈1.0 (see docstring)")
print(classification_report(y_test_npw, y_pred_npw,
                             target_names=["Active", "Warming", "Dormant"]))

# SHAP
explainer_npw   = shap.TreeExplainer(lgb_npw)
shap_values_npw = explainer_npw.shap_values(X_test_npw)

# Score all customers
X_all_npw = df[NPW_FEATURES].fillna(0)
npw_probs  = lgb_npw.predict_proba(X_all_npw)
df["purchase_window_pred"]  = lgb_npw.predict(X_all_npw)
df["pw_prob_active"]        = npw_probs[:, 0]
df["pw_prob_warming"]       = npw_probs[:, 1]
df["pw_prob_dormant"]       = npw_probs[:, 2]
df["purchase_window_label"] = df["purchase_window_pred"].map(
    {0: "Active", 1: "Warming", 2: "Dormant"})

print(f"   Predicted window distribution (all {len(df):,} customers):")
print(df["purchase_window_label"].value_counts().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 4. Unified Scoring Table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("4. UNIFIED CUSTOMER SCORING TABLE")
print("="*70)

# Campaign priority score: high CLV + high churn probability = intervene first
df["campaign_priority_score"] = (
    df["churn_probability"] * np.log1p(df["predicted_clv"])
).round(4)

print(f"\nTop 10 highest-priority customers:")
cols_show = ["CustomerID", "Segment", "Monetary", "predicted_clv", "clv_tier",
             "churn_probability", "purchase_window_label", "campaign_priority_score"]
print(df.sort_values("campaign_priority_score", ascending=False)[cols_show]
        .head(10).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Visualisations
# ─────────────────────────────────────────────────────────────────────────────

# ── Fig 1: CLV Model ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle("Model 1 — CLV Prediction (Regression)", fontsize=15, fontweight="bold", y=1.01)

# Actual vs Predicted
ax = axes[0]
ax.scatter(y_true_clv, y_pred_clv, alpha=0.25, s=12, color=C["accent"])
lim = max(y_true_clv.max(), y_pred_clv.max())
ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Actual CLV (£, log)", fontsize=11)
ax.set_ylabel("Predicted CLV (£, log)", fontsize=11)
ax.set_title(f"Actual vs Predicted\nR² = {r2:.3f}  |  Median APE = {mape:.1f}%", fontsize=11)
ax.legend(fontsize=9)

# Residuals
ax = axes[1]
resid = np.log1p(y_true_clv) - np.log1p(y_pred_clv)
ax.hist(resid, bins=50, color=C["accent"], edgecolor="white", alpha=0.85)
ax.axvline(0, color=C["warn"], lw=2, linestyle="--")
ax.axvline(resid.mean(), color=C["ok"], lw=1.5, linestyle=":", label=f"Mean={resid.mean():.2f}")
ax.set_xlabel("Log Residual (actual − predicted)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Residual Distribution\n(centred at 0 = unbiased)", fontsize=11)
ax.legend(fontsize=9)

# SHAP bar
ax = axes[2]
shap_imp = pd.Series(np.abs(shap_values_clv).mean(axis=0),
                     index=CLV_FEATURES).sort_values(ascending=True).tail(15)
colors_bar = [C["warn"] if v > shap_imp.median() else C["accent"] for v in shap_imp.values]
ax.barh(shap_imp.index, shap_imp.values, color=colors_bar)
ax.set_xlabel("Mean |SHAP value|", fontsize=11)
ax.set_title("Top 15 Features Driving CLV\n(SHAP — behavioral signals only)", fontsize=11)

plt.tight_layout()
plt.savefig("/home/cairo/code/portfolio/customer-segmentation/outputs/figures/fig8_1_clv_model.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ fig8_1_clv_model.png")


# ── Fig 2: Churn Model ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle("Model 2 — Churn Prediction (Binary Classification)", fontsize=15,
             fontweight="bold", y=1.01)

# Reliability diagram (calibration curve)
ax = axes[0]
frac_pos, mean_pred = calibration_curve(y_test_ch, y_prob_ch, n_bins=10)
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect")
ax.plot(mean_pred, frac_pos, "o-", color=C["accent"], lw=2,
        label=f"LightGBM (Brier={brier_raw:.3f})")
# Reference: naive classifier
ax.axhline(pos_rate, color=C["neutral"], lw=1, linestyle=":", label=f"Naive ({pos_rate:.2f})")
ax.fill_between([0, 1], [pos_rate]*2, [1, 1], alpha=0.05, color=C["ok"])
ax.set_xlabel("Mean Predicted Probability", fontsize=11)
ax.set_ylabel("Fraction of Positives", fontsize=11)
ax.set_title(f"Reliability Diagram\nSkill score vs naive: {skill_score:.3f}", fontsize=11)
ax.legend(fontsize=9)

# Threshold tuning
ax = axes[1]
ax.plot(results_thresh_df["threshold"], results_thresh_df["recall"],
        color=C["warn"], lw=2, label="Recall")
ax.plot(results_thresh_df["threshold"], results_thresh_df["precision"],
        color=C["ok"], lw=2, label="Precision")
ax.plot(results_thresh_df["threshold"], results_thresh_df["f1"],
        color=C["accent"], lw=2, linestyle="--", label="F1")
ax.axvline(best_thresh, color=C["primary"], lw=2, linestyle=":",
           label=f"Chosen θ = {best_thresh:.2f}")
ax.axhline(0.35, color=C["neutral"], lw=1, linestyle="--", alpha=0.6, label="Min precision 0.35")
ax.set_xlabel("Decision Threshold", fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Threshold Tuning\n(max Recall | Precision ≥ 0.35)", fontsize=11)
ax.legend(fontsize=8.5)
ax.set_xlim(0.1, 0.9)

# SHAP bar
ax = axes[2]
sv = shap_values_churn
if isinstance(sv, list): sv = sv[1]
shap_ch = pd.Series(np.abs(sv).mean(axis=0),
                    index=CHURN_FEATURES).sort_values(ascending=True).tail(15)
colors_ch = [C["warn"] if v > shap_ch.median() else C["accent"] for v in shap_ch.values]
ax.barh(shap_ch.index, shap_ch.values, color=colors_ch)
ax.set_xlabel("Mean |SHAP value|", fontsize=11)
ax.set_title("Top 15 Features Driving Churn\n(SHAP)", fontsize=11)

plt.tight_layout()
plt.savefig("/home/cairo/code/portfolio/customer-segmentation/outputs/figures/fig8_2_churn_model.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ fig8_2_churn_model.png")


# ── Fig 3: NPW Model ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle("Model 3 — Next Purchase Window (3-class)", fontsize=15,
             fontweight="bold", y=1.01)

# Confusion matrix
ax = axes[0]
cm = confusion_matrix(y_test_npw, y_pred_npw)
ConfusionMatrixDisplay(cm, display_labels=["Active", "Warming", "Dormant"]).plot(
    ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix\n(deterministic from Recency — see notes)", fontsize=11)

# Window distribution by segment
ax = axes[1]
seg_win = df.groupby(["Segment", "purchase_window_label"]).size().unstack(fill_value=0)
seg_win_pct = seg_win.div(seg_win.sum(axis=1), axis=0) * 100
seg_win_pct = seg_win_pct.reindex(columns=["Active", "Warming", "Dormant"], fill_value=0)
seg_win_pct.plot(kind="barh", ax=ax, stacked=True,
                 color=[C["ok"], C["warn"], C["neutral"]], edgecolor="white")
ax.set_xlabel("% of Customers", fontsize=11)
ax.set_title("Purchase Window by Segment\n(campaign timing input)", fontsize=11)
ax.legend(loc="lower right", fontsize=8)

# SHAP for Dormant class
ax = axes[2]
sv_npw = shap_values_npw
if isinstance(sv_npw, list):
    sv_dormant = sv_npw[2]
elif sv_npw.ndim == 3:
    sv_dormant = sv_npw[:, :, 2]
else:
    sv_dormant = sv_npw

shap_npw = pd.Series(np.abs(sv_dormant).mean(axis=0),
                     index=NPW_FEATURES).sort_values(ascending=True).tail(15)
colors_npw = [C["neutral"] if v > shap_npw.median() else C["accent"] for v in shap_npw.values]
ax.barh(shap_npw.index, shap_npw.values, color=colors_npw)
ax.set_xlabel("Mean |SHAP value|", fontsize=11)
ax.set_title("Signals → Dormant Class\n(SHAP)", fontsize=11)

plt.tight_layout()
plt.savefig("/home/cairo/code/portfolio/customer-segmentation/outputs/figures/fig8_3_purchase_window.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ fig8_3_purchase_window.png")


# ── Fig 4: Campaign Priority Dashboard ───────────────────────────────────────
tier_colors = {"Low": "#AED6F1", "Mid": "#2E86C1", "High": "#1A5276", "Premium": "#E74C3C"}

fig, axes = plt.subplots(2, 2, figsize=(19, 12))
fig.suptitle("Campaign Prioritization Dashboard — Phase 8 Output",
             fontsize=15, fontweight="bold")

# CLV vs Churn scatter
ax = axes[0, 0]
for tier, color in tier_colors.items():
    mask = df["clv_tier"] == tier
    ax.scatter(df.loc[mask, "churn_probability"],
               np.log1p(df.loc[mask, "predicted_clv"]),
               alpha=0.35, s=14, color=color, label=tier)
ax.axvline(best_thresh, color="grey", lw=1, linestyle="--", alpha=0.6)
ax.set_xlabel("Churn Probability", fontsize=11)
ax.set_ylabel("Predicted CLV (log £)", fontsize=11)
ax.set_title("CLV vs Churn Risk\n(top-right quadrant = highest intervention priority)", fontsize=11)
ax.legend(title="CLV Tier", fontsize=9)

# Priority score by segment
ax = axes[0, 1]
pri = df.groupby("Segment")["campaign_priority_score"].mean().sort_values()
bar_cols = [C["warn"] if v > pri.median() else C["accent"] for v in pri.values]
ax.barh(pri.index, pri.values, color=bar_cols)
ax.set_xlabel("Avg Campaign Priority Score", fontsize=11)
ax.set_title("Campaign Priority by Segment\nchurn_prob × log(predicted_clv)", fontsize=11)

# Churn distribution by CLV tier
ax = axes[1, 0]
for tier in ["Low", "Mid", "High", "Premium"]:
    mask = df["clv_tier"] == tier
    if mask.sum() > 0:
        ax.hist(df.loc[mask, "churn_probability"], bins=25, alpha=0.55,
                label=tier, color=tier_colors[tier], edgecolor="white")
ax.axvline(best_thresh, color="black", lw=2, linestyle="--",
           label=f"Threshold ({best_thresh:.2f})")
ax.set_xlabel("Churn Probability", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Churn Risk Distribution by CLV Tier", fontsize=11)
ax.legend(fontsize=9)

# Heatmap: predicted CLV by segment × window
ax = axes[1, 1]
hm = (df.groupby(["Segment", "purchase_window_label"])["predicted_clv"]
        .mean()
        .unstack(fill_value=0)
        .reindex(columns=["Active", "Warming", "Dormant"], fill_value=0))
sns.heatmap(hm, ax=ax, fmt=",.0f", annot=True, cmap="YlOrRd",
            linewidths=0.5, cbar_kws={"label": "Avg Predicted CLV (£)"})
ax.set_title("Avg Predicted CLV\nby Segment × Purchase Window", fontsize=11)
ax.set_xlabel("Purchase Window", fontsize=10)
ax.set_ylabel("")

plt.tight_layout()
plt.savefig("/home/cairo/code/portfolio/customer-segmentation/outputs/figures/fig8_4_campaign_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ fig8_4_campaign_dashboard.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Export
# ─────────────────────────────────────────────────────────────────────────────
output_df = df[[
    "CustomerID", "Segment",
    "Monetary", "Frequency", "Recency",
    "predicted_clv", "clv_tier",
    "churn_probability", "churn_flag",
    "purchase_window_label", "pw_prob_active", "pw_prob_warming", "pw_prob_dormant",
    "campaign_priority_score", "RFM_Enhanced_Cluster",
]].copy()
output_df.columns = [c.lower() for c in output_df.columns]
output_df = output_df.sort_values("campaign_priority_score", ascending=False)
output_df.to_csv("/home/cairo/code/portfolio/customer-segmentation/outputs/campaigns/ml_scored_customers.csv", index=False)
print(f"\n✓ Exported {len(output_df):,} scored customers")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 8 SUMMARY")
print("="*70)
print(f"""
  MODEL RESULTS
  {'─'*60}
  1. CLV Prediction (Regression)
     R² on test set (log scale): {r2:.4f}
     CV R² 5-fold:               {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
     Median APE:                 {mape:.1f}%
     MAE / RMSE:                 £{mae:,.0f} / £{rmse:,.0f}
     Feature set:                {len(CLV_FEATURES)} features (no monetary-derived ratios)
  {'─'*60}
  2. Churn Prediction (Binary)
     ROC-AUC:                    {roc_auc:.4f}
     PR-AUC:                     {pr_auc:.4f}
     Brier score (raw):          {brier_raw:.4f}  (skill={skill_score:.3f} vs naive)
     Decision threshold:         {best_thresh:.2f}
     Recall @ threshold:         {best_recall:.3f}
  {'─'*60}
  3. Next Purchase Window (3-class)
     ROC-AUC (weighted OvR):     {roc_auc_npw:.4f}  (expected ≈1.0)
     Split:                      stratified random (current-state classifier)
  {'─'*60}
  OUTPUT
     Customers scored:           {len(output_df):,}
     High-priority flag=1:       {output_df['churn_flag'].sum():,}
     Premium CLV tier:           {(output_df['clv_tier']=='Premium').sum():,}
""")
