"""
Phase 7: Automated Insight Engine
==================================
Intelligent Customer Segmentation & Revenue Optimization Project
UCI Online Retail II Dataset

Generates automated, prioritized business insights from pre-computed
customer analytics outputs (RFM, behavioral clusters, temporal patterns).

Usage:
    engine = InsightEngine()
    engine.load_data()
    report = engine.generate_full_report()
    engine.export_json("insights_report.json")
    engine.print_executive_summary()
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class Insight:
    """A single business insight with priority and recommended action."""
    id: str
    category: str           # "churn_risk" | "growth" | "opportunity" | "timing" | "product"
    priority: str           # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
    title: str
    headline: str           # One-sentence business impact statement
    detail: str             # 2-3 sentence explanation
    metric_name: str        # Key supporting metric
    metric_value: str       # Formatted value
    revenue_impact: float   # Estimated £ impact (0 if not applicable)
    customers_affected: int
    recommended_action: str
    campaign_timing: str    # When to execute
    effort_level: str       # "Quick Win" | "Medium" | "Strategic"

    def to_dict(self):
        return asdict(self)


@dataclass
class Alert:
    """Threshold-based alert for immediate attention."""
    alert_id: str
    severity: str           # "RED" | "AMBER" | "GREEN"
    segment: str
    message: str
    metric: str
    value: float
    threshold: float
    action_required: str


# ─────────────────────────────────────────────────────────────
# THRESHOLDS & CONFIGURATION
# ─────────────────────────────────────────────────────────────

THRESHOLDS = {
    # Churn risk score (0-100 scale; higher = more risk)
    "churn_risk_critical": 70,
    "churn_risk_high": 50,

    # Recency thresholds (days)
    "recency_critical": 365,
    "recency_warning": 180,

    # Revenue concentration
    "revenue_concentration_alert": 50,     # % revenue from top segment

    # Engagement consistency (0-1 scale)
    "engagement_low": 0.3,
    "engagement_high": 0.7,

    # Product diversity (unique products)
    "diversity_low": 10,                   # Low diversity = cross-sell opportunity

    # High-value customer threshold (£ CLV)
    "high_value_clv": 5000,

    # Bulk buyer definition
    "bulk_buyer_qty": 100,

    # Timing window for campaign execution (days)
    "campaign_window_immediate": 7,
    "campaign_window_short": 30,
    "campaign_window_medium": 90,
}

DATA_PATHS = {
    "features": "/home/cairo/code/portfolio/customer-segmentation/data/features/features_master.csv",
    "rfm_scores": "/home/cairo/code/portfolio/customer-segmentation/data/processed/rfm_customer_scores.csv",
    "rfm_segment_profiles": "/home/cairo/code/portfolio/customer-segmentation/outputs/reports/rfm_segment_profiles.csv",
    "rfm_enhanced_clusters": "/home/cairo/code/portfolio/customer-segmentation/outputs/reports/rfm_enhanced_cluster_profiles.csv",
    "behavioral_clusters": "/home/cairo/code/portfolio/customer-segmentation/outputs/reports/behavioral_cluster_profiles.csv",
    "timing_personas": "/home/cairo/code/portfolio/customer-segmentation/outputs/reports/customer_timing_personas.csv",
    "daily_patterns": "/home/cairo/code/portfolio/customer-segmentation/outputs/reports/daily_purchase_patterns.csv",
    "hourly_patterns": "/home/cairo/code/portfolio/customer-segmentation/outputs/reports/hourly_purchase_patterns.csv",
    "segment_hourly": "/home/cairo/code/portfolio/customer-segmentation/outputs/reports/segment_hourly_patterns.csv",
    "segment_daily": "/home/cairo/code/portfolio/customer-segmentation/outputs/reports/segment_daily_patterns.csv",
    "executive_summary": "/home/cairo/code/portfolio/customer-segmentation/outputs/reports/rfm_executive_summary.json",
    # Segment CSVs
    "champions": "/home/cairo/code/portfolio/customer-segmentation/outputs/campaigns/segment_champions.csv",
    "cant_lose": "/home/cairo/code/portfolio/customer-segmentation/outputs/campaigns/segment_cant_lose.csv",
    "at_risk": "/home/cairo/code/portfolio/customer-segmentation/outputs/campaigns/segment_at_risk.csv",
    "potential_loyalists": "/home/cairo/code/portfolio/customer-segmentation/outputs/campaigns/segment_potential_loyalists.csv",
}


# ─────────────────────────────────────────────────────────────
# INSIGHT ENGINE
# ─────────────────────────────────────────────────────────────

class InsightEngine:
    """
    Automated insight generation engine for customer segmentation analytics.

    Produces three output types:
    1. Threshold-based Alerts (immediate attention required)
    2. Prioritized Insights (strategic recommendations)
    3. Opportunity Scores (ranked cross-sell/upsell opportunities)
    """

    def __init__(self, data_paths: Dict = None, thresholds: Dict = None):
        self.data_paths = data_paths or DATA_PATHS
        self.thresholds = thresholds or THRESHOLDS
        self.data = {}
        self.insights: List[Insight] = []
        self.alerts: List[Alert] = []
        self.opportunities: List[Dict] = []
        self.generated_at = datetime.now().isoformat()

    # ──────────────────────────────────────────
    # DATA LOADING
    # ──────────────────────────────────────────

    def load_data(self) -> "InsightEngine":
        """Load all required datasets. Returns self for chaining."""
        print("📊 Loading analytics datasets...")

        # Tabular data
        for key in [
            "features", "rfm_scores", "rfm_segment_profiles",
            "rfm_enhanced_clusters", "behavioral_clusters",
            "timing_personas", "daily_patterns", "hourly_patterns",
            "segment_hourly", "segment_daily",
            "champions", "cant_lose", "at_risk", "potential_loyalists"
        ]:
            path = self.data_paths.get(key)
            if path and Path(path).exists():
                self.data[key] = pd.read_csv(path)
            else:
                print(f"  ⚠️  {key} not found at {path}")

        # JSON summary
        with open(self.data_paths["executive_summary"]) as f:
            self.data["executive_summary"] = json.load(f)

        print(f"  ✅ Loaded {len(self.data)} datasets")
        return self

    # ──────────────────────────────────────────
    # ALERT GENERATION
    # ──────────────────────────────────────────

    def _generate_alerts(self) -> List[Alert]:
        """Check key metrics against thresholds and generate alerts."""
        alerts = []
        features = self.data["features"]
        exec_summary = self.data["executive_summary"]

        # ── ALERT 1: Revenue Concentration
        champ_rev_pct = exec_summary["segments"]["Champions"]["Revenue_Pct"]
        alerts.append(Alert(
            alert_id="ALERT_001",
            severity="RED" if champ_rev_pct > 50 else "AMBER",
            segment="Champions",
            message=(
                f"Champions (8% of customers) drive {champ_rev_pct:.1f}% of total revenue. "
                f"Extreme concentration — any churn in this segment is existential."
            ),
            metric="Revenue Concentration",
            value=champ_rev_pct,
            threshold=self.thresholds["revenue_concentration_alert"],
            action_required="Implement VIP retention programme immediately. Monthly check-ins, early access, dedicated account managers."
        ))

        # ── ALERT 2: Can't Lose Them — high recency
        cant_lose = self.data["cant_lose"]
        cant_lose_avg_recency = cant_lose["Recency"].mean()
        cant_lose_revenue = exec_summary["segments"]["Can't Lose Them"]["Total_Revenue"]
        alerts.append(Alert(
            alert_id="ALERT_002",
            severity="RED",
            segment="Can't Lose Them",
            message=(
                f"227 high-value customers haven't purchased in {cant_lose_avg_recency:.0f} days on average. "
                f"£{cant_lose_revenue:,.0f} in annual revenue at immediate risk."
            ),
            metric="Avg Days Since Last Purchase",
            value=cant_lose_avg_recency,
            threshold=self.thresholds["recency_critical"],
            action_required="Launch emergency win-back campaign within 7 days. Personal outreach, exclusive offers, account review call."
        ))

        # ── ALERT 3: High churn risk segment (RFM Enhanced Cluster 3)
        rfm_enh = self.data["rfm_enhanced_clusters"]
        high_churn = rfm_enh[rfm_enh["churn_risk_score_mean"] > self.thresholds["churn_risk_high"]]
        if not high_churn.empty:
            worst = high_churn.sort_values("churn_risk_score_mean", ascending=False).iloc[0]
            alerts.append(Alert(
                alert_id="ALERT_003",
                severity="RED",
                segment=f"ML Cluster {int(worst['RFM_Enhanced_Cluster'])}",
                message=(
                    f"Cluster {int(worst['RFM_Enhanced_Cluster'])} has avg churn risk score "
                    f"{worst['churn_risk_score_mean']:.1f}/100 with {int(worst['Customer_Count'])} customers "
                    f"and £{worst['clv_mean']*worst['Customer_Count']:,.0f} CLV at stake."
                ),
                metric="Churn Risk Score",
                value=worst["churn_risk_score_mean"],
                threshold=self.thresholds["churn_risk_high"],
                action_required="Deploy automated churn prevention drip campaign. 30-60-90 day touchpoint cadence."
            ))

        # ── ALERT 4: At Risk customers previously high-value
        at_risk = self.data["at_risk"]
        features = self.data["features"]
        at_risk_high_val = features[
            (features["is_at_risk"] == 1) &
            (features["clv"] > self.thresholds["high_value_clv"])
        ]
        if len(at_risk_high_val) > 0:
            alerts.append(Alert(
                alert_id="ALERT_004",
                severity="AMBER",
                segment="At Risk — High Value",
                message=(
                    f"{len(at_risk_high_val)} previously high-value customers (CLV >£{self.thresholds['high_value_clv']:,}) "
                    f"are now classified as At Risk. "
                    f"Combined CLV: £{at_risk_high_val['clv'].sum():,.0f}."
                ),
                metric="High-Value At-Risk Count",
                value=len(at_risk_high_val),
                threshold=10,
                action_required="Priority list for sales team personal outreach. Investigate root cause of disengagement."
            ))

        # ── ALERT 5: Sunday revenue anomaly
        daily = self.data["daily_patterns"]
        sunday = daily[daily["DayName"] == "Sunday"].iloc[0]
        thursday = daily[daily["DayName"] == "Thursday"].iloc[0]
        ratio = sunday["Revenue_Pct"] / thursday["Revenue_Pct"]
        alerts.append(Alert(
            alert_id="ALERT_005",
            severity="GREEN",
            segment="All Segments",
            message=(
                f"Sunday generates only {sunday['Revenue_Pct']:.1f}% of weekly revenue vs. "
                f"{thursday['Revenue_Pct']:.1f}% on Thursdays — a {1/ratio:.1f}x gap. "
                f"Weekend activity is significantly undermonetised."
            ),
            metric="Sunday vs Thursday Revenue Ratio",
            value=ratio,
            threshold=0.5,
            action_required="Test weekend-specific promotions for the 652 Weekend Browser customers."
        ))

        self.alerts = alerts
        return alerts

    # ──────────────────────────────────────────
    # INSIGHT GENERATION
    # ──────────────────────────────────────────

    def _generate_churn_insights(self) -> List[Insight]:
        """Generate churn prevention insights."""
        insights = []
        exec_summary = self.data["executive_summary"]
        features = self.data["features"]
        cant_lose = self.data["cant_lose"]

        # ── INSIGHT 1: Can't Lose Them win-back
        cant_lose_revenue = exec_summary["segments"]["Can't Lose Them"]["Total_Revenue"]
        cant_lose_avg_value = exec_summary["segments"]["Can't Lose Them"]["Avg_Monetary"]
        insights.append(Insight(
            id="INS_CHU_001",
            category="churn_risk",
            priority="CRITICAL",
            title="Emergency Win-Back: Can't Lose Them Segment",
            headline=f"227 formerly loyal customers averaging £{cant_lose_avg_value:,.0f} each haven't purchased in nearly a year.",
            detail=(
                f"The 'Can't Lose Them' segment averages 341 days since last purchase yet maintains "
                f"strong historical frequency (8.93 purchases) and spend (£4,488 avg). "
                f"These customers chose to disengage — likely due to competitive offers, service issues, or changing needs. "
                f"Win-back cost is far lower than new customer acquisition."
            ),
            metric_name="Avg Days Inactive",
            metric_value="341 days",
            revenue_impact=cant_lose_revenue * 0.40,  # Conservative 40% recovery estimate
            customers_affected=227,
            recommended_action=(
                "Personal win-back campaign: (1) Personalised email from account manager with 'We miss you' subject. "
                "(2) 15% loyalty discount on next order. (3) Phone outreach for top 50 by historical spend. "
                "(4) Survey to understand reason for disengagement."
            ),
            campaign_timing="Immediate — within 7 days",
            effort_level="Quick Win"
        ))

        # ── INSIGHT 2: About To Sleep pre-emptive retention
        about_to_sleep = exec_summary["segments"]["About To Sleep"]
        insights.append(Insight(
            id="INS_CHU_002",
            category="churn_risk",
            priority="HIGH",
            title="Pre-emptive Retention: About To Sleep Segment",
            headline=f"606 customers averaging 313 days inactive — within 2 months of becoming 'At Risk'.",
            detail=(
                f"About To Sleep customers have average recency of 313 days but still show some engagement (F-score: 2.47). "
                f"They represent £481,964 in historical revenue. "
                f"Intervening now is 3-5x cheaper than attempting to recover them once fully churned."
            ),
            metric_name="Days Until At-Risk Threshold",
            metric_value="~52 days",
            revenue_impact=about_to_sleep["Total_Revenue"] * 0.30,
            customers_affected=606,
            recommended_action=(
                "Automated re-engagement drip: (1) T+0: 'We noticed you haven't ordered recently' email with bestsellers. "
                "(2) T+14: Category highlight email based on past purchase behaviour. "
                "(3) T+30: Limited-time 10% incentive. (4) T+45: Final personal outreach."
            ),
            campaign_timing="Launch within 14 days",
            effort_level="Medium"
        ))

        # ── INSIGHT 3: ML cluster high churn risk
        rfm_enh = self.data["rfm_enhanced_clusters"]
        cluster_3 = rfm_enh[rfm_enh["RFM_Enhanced_Cluster"] == 3].iloc[0]
        insights.append(Insight(
            id="INS_CHU_003",
            category="churn_risk",
            priority="HIGH",
            title="ML-Identified Churn Risk Cluster (Cluster 3)",
            headline=f"942 customers with 71/100 churn risk score despite having real purchase history.",
            detail=(
                f"ML Cluster 3 customers show an average churn risk score of 70.98 — the highest of any cluster. "
                f"They have purchased historically (avg 4.35 times, £1,778 spend) but last bought 354 days ago on average. "
                f"High bulk-buyer rate (97%) suggests B2B customers potentially switching to a competitor supplier."
            ),
            metric_name="Avg Churn Risk Score",
            metric_value="70.98 / 100",
            revenue_impact=cluster_3["clv_sum"] * 0.35,
            customers_affected=942,
            recommended_action=(
                "B2B-focused win-back: (1) Sales team review of top 100 by CLV. "
                "(2) Competitive benchmarking — survey on pricing/service gaps. "
                "(3) Volume discount offer for returning order. "
                "(4) Set up regular account review cadence for recovered customers."
            ),
            campaign_timing="Within 30 days",
            effort_level="Medium"
        ))

        return insights

    def _generate_growth_insights(self) -> List[Insight]:
        """Generate growth and expansion insights."""
        insights = []
        exec_summary = self.data["executive_summary"]

        # ── INSIGHT 4: Potential Loyalists upgrade opportunity
        pot_loy = exec_summary["segments"]["Potential Loyalists"]
        loyal_avg_monetary = exec_summary["segments"]["Loyal Customers"]["Avg_Monetary"]
        uplift_per_customer = loyal_avg_monetary - pot_loy["Avg_Monetary"]
        insights.append(Insight(
            id="INS_GRW_001",
            category="growth",
            priority="HIGH",
            title="Upgrade Pathway: Convert Potential Loyalists to Loyal",
            headline=f"1,280 Potential Loyalists average only £526 spend — vs £3,991 for Loyal Customers.",
            detail=(
                f"Potential Loyalists represent the largest customer segment (21.8%) but only 3.8% of revenue. "
                f"Converting just 10% of them to Loyal Customer status would generate "
                f"an additional £{uplift_per_customer * 1280 * 0.10:,.0f} in revenue. "
                f"They've purchased recently (avg 62 days) and are still engaged — timing is optimal."
            ),
            metric_name="Revenue Uplift Potential (10% conversion)",
            metric_value=f"£{uplift_per_customer * 1280 * 0.10:,.0f}",
            revenue_impact=uplift_per_customer * 1280 * 0.10,
            customers_affected=1280,
            recommended_action=(
                "Loyalty upgrade programme: (1) 'You're almost a VIP' milestone email at purchase #2. "
                "(2) Exclusive early access to new product ranges. "
                "(3) Personalised category recommendation based on first purchase. "
                "(4) Progress bar showing loyalty tier status."
            ),
            campaign_timing="Ongoing — launch within 30 days",
            effort_level="Strategic"
        ))

        # ── INSIGHT 5: Lost customers with high historical value — reactivation
        lost = exec_summary["segments"]["Lost"]
        insights.append(Insight(
            id="INS_GRW_002",
            category="growth",
            priority="MEDIUM",
            title="Selective Reactivation: High-Value Lost Customers",
            headline=f"797 'Lost' customers averaged £2,977 spend — worth selective reactivation investment.",
            detail=(
                f"Lost customers represent 13.4% of historical revenue (£2.37M). "
                f"While mass reactivation is inefficient, targeting the top quartile by Monetary value "
                f"with personalised outreach could recover significant revenue. "
                f"Average recency of 139 days suggests some are still reachable."
            ),
            metric_name="Potential Recovery Revenue",
            metric_value=f"£{lost['Total_Revenue'] * 0.20:,.0f}",
            revenue_impact=lost["Total_Revenue"] * 0.20,
            customers_affected=200,  # Top quartile
            recommended_action=(
                "Tiered reactivation: (1) Sort Lost customers by Monetary (desc). "
                "(2) Top 200 receive personalised email + phone call. "
                "(3) Next 300 receive targeted email campaign with 10% incentive. "
                "(4) Remainder receive low-cost automated drip only."
            ),
            campaign_timing="Within 60 days",
            effort_level="Medium"
        ))

        return insights

    def _generate_opportunity_insights(self) -> List[Insight]:
        """Generate cross-sell and product opportunity insights."""
        insights = []
        features = self.data["features"]
        exec_summary = self.data["executive_summary"]

        # ── INSIGHT 6: Cross-sell to low-diversity high-value customers
        low_div_high_val = features[
            (features["product_diversity"] < self.thresholds["diversity_low"]) &
            (features["Monetary"] > features["Monetary"].quantile(0.75))
        ]
        insights.append(Insight(
            id="INS_OPP_001",
            category="opportunity",
            priority="HIGH",
            title="Cross-Sell: High-Value Customers Buying Narrow Range",
            headline=f"550 high-value customers averaging £15,770 CLV buy <10 product categories each.",
            detail=(
                f"These customers have proven purchasing power but concentrated category exposure. "
                f"Category co-purchase analysis shows HOME DECOR + METAL SIGN (20,047 co-purchases) and "
                f"HOME DECOR + HEART/LOVE (19,240) are the strongest adjacencies. "
                f"Recommending just one adjacent category per customer could meaningfully lift basket size."
            ),
            metric_name="Avg CLV of Target Customers",
            metric_value="£15,770",
            revenue_impact=low_div_high_val["clv"].mean() * len(low_div_high_val) * 0.08,
            customers_affected=550,
            recommended_action=(
                "Category expansion campaign: (1) Pull each customer's purchase history. "
                "(2) Map to top 3 co-purchased categories they haven't bought. "
                "(3) Send 'Customers like you also love...' email with 3 category showcases. "
                "(4) Bundle offer: 5% off when adding new category to existing order."
            ),
            campaign_timing="Within 30 days",
            effort_level="Medium"
        ))

        # ── INSIGHT 7: BAG + HOME DECOR bundle opportunity
        insights.append(Insight(
            id="INS_OPP_002",
            category="opportunity",
            priority="MEDIUM",
            title="Bundle Opportunity: Bag + Home Decor Power Pair",
            headline="BAG and HOME DECOR are co-purchased in 15,762 basket combinations — prime for bundling.",
            detail=(
                "The Category Co-Purchase Matrix reveals Bag+Home Decor (15,762), Candle+Home Decor (16,157), "
                "and Vintage+Home Decor (16,891) as the strongest non-Metal Sign adjacencies. "
                "Creating curated bundles at a modest 5% discount would increase average order value "
                "while reducing per-category marketing cost."
            ),
            metric_name="Top Co-Purchase Pair Count",
            metric_value="20,047 (Home Decor + Metal Sign)",
            revenue_impact=500000,  # Conservative estimate
            customers_affected=3000,
            recommended_action=(
                "Bundle design: (1) Create 5 signature bundles based on top co-purchase pairs. "
                "(2) Feature on homepage and in order confirmation emails. "
                "(3) A/B test 5% vs 10% bundle discount. "
                "(4) Track incremental AOV vs margin impact monthly."
            ),
            campaign_timing="Within 45 days",
            effort_level="Medium"
        ))

        # ── INSIGHT 8: Champions cross-sell premium categories
        champ_avg_clv = exec_summary["segments"]["Champions"]["Avg_Monetary"]
        insights.append(Insight(
            id="INS_OPP_003",
            category="opportunity",
            priority="HIGH",
            title="Champions Upsell: Premium Category Expansion",
            headline=f"471 Champion customers averaging £17,692 each — highest receptivity to premium ranges.",
            detail=(
                f"Champions buy 29 times on average and have the highest product diversity (12.4 categories) but "
                f"still allocate 30% of spend to the commoditised HOME DECOR & OTHER category. "
                f"Introducing curated premium/exclusive ranges would increase AOV and strengthen lock-in. "
                f"Champions have an F-score of 5.0 — they are the most reliable feedback source for new products."
            ),
            metric_name="Champions Total Revenue",
            metric_value="£8,333,125",
            revenue_impact=8333125.12 * 0.05,  # 5% AOV lift
            customers_affected=471,
            recommended_action=(
                "VIP product previews: (1) Monthly 'First Look' email with new or seasonal products. "
                "(2) Invite top 50 Champions to annual product feedback panel. "
                "(3) Introduce exclusive SKU only available to Champions tier. "
                "(4) Track Champions AOV monthly for uplift measurement."
            ),
            campaign_timing="Within 14 days",
            effort_level="Quick Win"
        ))

        return insights

    def _generate_timing_insights(self) -> List[Insight]:
        """Generate insights from temporal pattern analysis."""
        insights = []
        daily = self.data["daily_patterns"]
        hourly = self.data["hourly_patterns"]
        timing = self.data["timing_personas"]

        # ── INSIGHT 9: Tuesday/Thursday peak — campaign timing
        thu = daily[daily["DayName"] == "Thursday"].iloc[0]
        sun = daily[daily["DayName"] == "Sunday"].iloc[0]
        peak_hours = hourly.sort_values("Revenue_Pct", ascending=False).head(3)
        peak_hrs_str = ", ".join([f"{int(h)}:00" for h in peak_hours["Hour"].tolist()])
        insights.append(Insight(
            id="INS_TMG_001",
            category="timing",
            priority="MEDIUM",
            title="Campaign Timing Optimisation: Tuesday-Thursday Peak",
            headline=f"Tuesday+Thursday generate 40.5% of weekly revenue — ideal campaign deployment window.",
            detail=(
                f"Revenue peaks Tuesday ({daily[daily['DayName']=='Tuesday'].iloc[0]['Revenue_Pct']:.1f}%) "
                f"and Thursday ({thu['Revenue_Pct']:.1f}%). "
                f"Peak hours are {peak_hrs_str} — aligning email sends and promotions to this window "
                f"maximises visibility during active purchasing intent."
            ),
            metric_name="Best Revenue Window",
            metric_value="Tue-Thu, 10:00–13:00",
            revenue_impact=0,
            customers_affected=5878,
            recommended_action=(
                "Email deployment schedule: (1) Move all campaign sends to Tuesday 10:00 or Thursday 11:00. "
                "(2) Schedule promotional pushes for Wednesday evening (pre-Thursday peak). "
                "(3) Avoid Friday afternoon and Saturday sends (lowest engagement). "
                "(4) Test Sunday morning sends for Weekend Browser segment specifically."
            ),
            campaign_timing="Implement immediately for all future campaigns",
            effort_level="Quick Win"
        ))

        # ── INSIGHT 10: Weekend Browsers untapped opportunity
        weekend_count = (timing["Timing_Persona"] == "Weekend Browser").sum()
        weekday_prof = (timing["Timing_Persona"] == "Weekday Professional").sum()
        insights.append(Insight(
            id="INS_TMG_002",
            category="timing",
            priority="LOW",
            title="Weekend Browser Micro-Segment: Untapped Weekend Channel",
            headline=f"652 Weekend Browsers are active when the business is essentially closed (0.05% of Saturday revenue).",
            detail=(
                f"652 customers (11.1%) predominantly shop on weekends, yet Saturday generates just 0.05% of revenue. "
                f"This could indicate stock/availability issues on weekends, or a customer segment whose needs "
                f"are fundamentally different from the weekday B2B majority. "
                f"Targeted weekend content could convert these browsers into higher-frequency buyers."
            ),
            metric_name="Weekend Browser Count",
            metric_value="652 customers",
            revenue_impact=652 * 500 * 0.20,  # Conservative new revenue estimate
            customers_affected=652,
            recommended_action=(
                "Weekend activation test: (1) Pull Weekend Browser customer list. "
                "(2) Send Saturday morning email with 'Weekend Special' offer. "
                "(3) Ensure weekend stock visibility and ordering functionality. "
                "(4) Measure open rate, CTR, and conversion vs weekday control group."
            ),
            campaign_timing="Test within 60 days",
            effort_level="Quick Win"
        ))

        return insights

    # ──────────────────────────────────────────
    # OPPORTUNITY SCORING
    # ──────────────────────────────────────────

    def _score_opportunities(self) -> List[Dict]:
        """Rank all opportunities by revenue impact x effort score."""
        opportunities = []
        for insight in self.insights:
            if insight.revenue_impact > 0:
                effort_score = {"Quick Win": 3, "Medium": 2, "Strategic": 1}.get(insight.effort_level, 2)
                priority_score = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(insight.priority, 2)
                composite = (insight.revenue_impact / 1000) * effort_score * priority_score
                opportunities.append({
                    "insight_id": insight.id,
                    "title": insight.title,
                    "revenue_impact_gbp": round(insight.revenue_impact, 0),
                    "customers_affected": insight.customers_affected,
                    "effort_level": insight.effort_level,
                    "priority": insight.priority,
                    "composite_score": round(composite, 0),
                    "campaign_timing": insight.campaign_timing,
                })

        opportunities.sort(key=lambda x: x["composite_score"], reverse=True)
        self.opportunities = opportunities
        return opportunities

    # ──────────────────────────────────────────
    # AGGREGATE METRICS
    # ──────────────────────────────────────────

    def _compute_portfolio_health(self) -> Dict:
        """Compute overall customer portfolio health score."""
        exec_summary = self.data["executive_summary"]
        features = self.data["features"]
        segs = exec_summary["segments"]

        # Healthy = recent + frequent + high-value segments
        healthy_revenue = segs["Champions"]["Revenue_Pct"] + segs["Loyal Customers"]["Revenue_Pct"]
        at_risk_revenue = (segs["Can't Lose Them"]["Revenue_Pct"] +
                          segs["At Risk"]["Revenue_Pct"] +
                          segs["Hibernating"]["Revenue_Pct"])

        # Churn rate proxy
        churn_prone_customers = (segs["Can't Lose Them"]["Customer_Count"] +
                                  segs["At Risk"]["Customer_Count"] +
                                  segs["Hibernating"]["Customer_Count"])
        total_customers = exec_summary["total_customers"]

        health_score = round(healthy_revenue - (at_risk_revenue * 0.5), 1)

        return {
            "portfolio_health_score": health_score,
            "healthy_revenue_pct": round(healthy_revenue, 1),
            "at_risk_revenue_pct": round(at_risk_revenue, 1),
            "at_risk_customer_pct": round(churn_prone_customers / total_customers * 100, 1),
            "total_customers": total_customers,
            "total_revenue_gbp": exec_summary["total_revenue"],
            "avg_customer_value_gbp": round(exec_summary["avg_customer_value"], 0),
            "revenue_concentration_risk": "HIGH" if segs["Champions"]["Revenue_Pct"] > 40 else "MEDIUM",
        }

    # ──────────────────────────────────────────
    # MAIN ORCHESTRATION
    # ──────────────────────────────────────────

    def generate_full_report(self) -> Dict:
        """Run all insight modules and compile the full report."""
        print("\n🔍 Generating automated insights...")

        # Run all insight generators
        all_insights = []
        all_insights.extend(self._generate_churn_insights())
        all_insights.extend(self._generate_growth_insights())
        all_insights.extend(self._generate_opportunity_insights())
        all_insights.extend(self._generate_timing_insights())
        self.insights = all_insights

        # Generate alerts
        print("  ⚡ Running threshold alerts...")
        self._generate_alerts()

        # Score opportunities
        print("  📈 Scoring opportunities...")
        self._score_opportunities()

        # Portfolio health
        print("  🏥 Computing portfolio health...")
        health = self._compute_portfolio_health()

        # Compile report
        report = {
            "meta": {
                "generated_at": self.generated_at,
                "project": "Intelligent Customer Segmentation & Revenue Optimization",
                "dataset": "UCI Online Retail II",
                "total_insights": len(self.insights),
                "total_alerts": len(self.alerts),
                "total_opportunities": len(self.opportunities),
            },
            "portfolio_health": health,
            "alerts": [
                {
                    "alert_id": a.alert_id,
                    "severity": a.severity,
                    "segment": a.segment,
                    "message": a.message,
                    "metric": a.metric,
                    "value": round(a.value, 2),
                    "threshold": a.threshold,
                    "action_required": a.action_required,
                }
                for a in self.alerts
            ],
            "insights": [ins.to_dict() for ins in self.insights],
            "opportunity_ranking": self.opportunities,
            "summary": {
                "total_revenue_at_risk": sum(
                    i.revenue_impact for i in self.insights
                    if i.category == "churn_risk"
                ),
                "total_revenue_opportunity": sum(
                    i.revenue_impact for i in self.insights
                    if i.category in ("growth", "opportunity")
                ),
                "critical_actions": [
                    i.title for i in self.insights if i.priority == "CRITICAL"
                ],
                "quick_wins": [
                    i.title for i in self.insights if i.effort_level == "Quick Win"
                ],
            },
        }

        print(f"\n✅ Report complete: {len(self.insights)} insights | {len(self.alerts)} alerts | {len(self.opportunities)} ranked opportunities")
        return report

    # ──────────────────────────────────────────
    # EXPORT
    # ──────────────────────────────────────────

    def export_json(self, filepath: str) -> None:
        """Export the full report to JSON."""
        report = self.generate_full_report() if not self.insights else {
            "meta": {"generated_at": self.generated_at},
            "alerts": [vars(a) for a in self.alerts],
            "insights": [ins.to_dict() for ins in self.insights],
            "opportunity_ranking": self.opportunities,
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        print(f"💾 Report exported to {filepath}")

    def print_executive_summary(self) -> None:
        """Print formatted executive summary to stdout."""
        print("\n" + "=" * 70)
        print("  AUTOMATED INSIGHT ENGINE — EXECUTIVE SUMMARY")
        print("=" * 70)

        print("\n🚨 ALERTS REQUIRING IMMEDIATE ATTENTION")
        print("-" * 50)
        for a in self.alerts:
            icon = "🔴" if a.severity == "RED" else "🟡" if a.severity == "AMBER" else "🟢"
            print(f"\n{icon} [{a.alert_id}] {a.segment}")
            print(f"   {a.message}")
            print(f"   → ACTION: {a.action_required}")

        print("\n\n📊 TOP INSIGHTS BY PRIORITY")
        print("-" * 50)
        sorted_insights = sorted(self.insights, key=lambda x: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}[x.priority])
        for ins in sorted_insights:
            print(f"\n[{ins.priority}] {ins.title}")
            print(f"   {ins.headline}")
            if ins.revenue_impact > 0:
                print(f"   💰 Revenue impact: £{ins.revenue_impact:,.0f}")
            print(f"   ⏰ Timing: {ins.campaign_timing}")
            print(f"   📋 {ins.recommended_action[:100]}...")

        print("\n\n🏆 OPPORTUNITY RANKING (Revenue Impact × Effort Score)")
        print("-" * 50)
        for i, opp in enumerate(self.opportunities[:5], 1):
            print(f"\n#{i}: {opp['title']}")
            print(f"     Revenue Impact: £{opp['revenue_impact_gbp']:,.0f} | Effort: {opp['effort_level']} | Score: {opp['composite_score']:.0f}")

        print("\n" + "=" * 70)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = InsightEngine()
    engine.load_data()
    report = engine.generate_full_report()
    engine.export_json("/home/cairo/code/portfolio/customer-segmentation/outputs/reports/insights_report.json")
    engine.print_executive_summary()
