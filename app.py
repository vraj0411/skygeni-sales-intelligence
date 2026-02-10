"""
SkyGeni Sales Intelligence Dashboard

Run with: streamlit run app.py
"""

from pathlib import Path
import os

# Load environment variables from .env file (if present)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        pass

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))

# Import LLM insights generator
try:
    from src.llm.insights_generator import get_insights_generator
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="SkyGeni Sales Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .high-risk { color: #e74c3c; }
    .medium-risk { color: #f39c12; }
    .low-risk { color: #27ae60; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load scored deals data."""
    data_path = Path("data/processed/scored_deals.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
        df["created_date"] = pd.to_datetime(df["created_date"])
        df["closed_date"] = pd.to_datetime(df["closed_date"])
        return df
    return None


@st.cache_data
def load_metrics():
    """Load saved metrics."""
    metrics_path = Path("data/processed/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


def main():
    # Title
    st.title("SkyGeni Sales Intelligence Dashboard")

    # Initialize LLM
    llm = None
    if LLM_AVAILABLE:
        try:
            llm = get_insights_generator()
        except Exception as e:
            st.sidebar.warning(f"LLM not available: {e}")

    # Load data
    df = load_data()
    metrics = load_metrics()

    if df is None:
        st.error("No data found. Please run `python train.py` first to generate scored deals.")
        st.code("python train.py")
        return

    # Sidebar filters
    st.sidebar.header("Filters")

    # Date range filter
    min_date = df["created_date"].min().date()
    max_date = df["created_date"].max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Region filter
    regions = ["All"] + sorted(df["region"].unique().tolist())
    selected_region = st.sidebar.selectbox("Region", regions)

    # Industry filter
    industries = ["All"] + sorted(df["industry"].unique().tolist())
    selected_industry = st.sidebar.selectbox("Industry", industries)

    # Risk filter
    risk_categories = ["All", "High Risk", "Medium Risk", "Low Risk"]
    selected_risk = st.sidebar.selectbox("Risk Category", risk_categories)

    # Apply filters
    filtered_df = df.copy()
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df["created_date"].dt.date >= date_range[0]) &
            (filtered_df["created_date"].dt.date <= date_range[1])
        ]
    if selected_region != "All":
        filtered_df = filtered_df[filtered_df["region"] == selected_region]
    if selected_industry != "All":
        filtered_df = filtered_df[filtered_df["industry"] == selected_industry]
    if selected_risk != "All":
        filtered_df = filtered_df[filtered_df["risk_category"] == selected_risk]

    # Main metrics row
    st.header("Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        win_rate = filtered_df["target"].mean() * 100
        st.metric("Win Rate", f"{win_rate:.1f}%")

    with col2:
        total_pipeline = filtered_df["deal_amount"].sum()
        st.metric("Total Pipeline", f"${total_pipeline:,.0f}")

    with col3:
        high_risk = (filtered_df["risk_category"] == "High Risk").sum()
        st.metric("High Risk Deals", high_risk)

    with col4:
        avg_risk = filtered_df["risk_score"].mean() * 100
        st.metric("Avg Risk Score", f"{avg_risk:.1f}%")

    with col5:
        at_risk_value = filtered_df[filtered_df["risk_category"] == "High Risk"]["deal_amount"].sum()
        st.metric("At-Risk Value", f"${at_risk_value:,.0f}")

    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Risk Overview",
        "Pipeline Analysis",
        "Rep Performance",
        "Deal Explorer",
        "Model Insights",
        "Alerts",
        "Forecast"
    ])

    # Tab 1: Risk Overview
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Risk distribution pie chart
            risk_dist = filtered_df["risk_category"].value_counts()
            fig_pie = px.pie(
                values=risk_dist.values,
                names=risk_dist.index,
                title="Deal Risk Distribution",
                color_discrete_map={
                    "High Risk": "#e74c3c",
                    "Medium Risk": "#f39c12",
                    "Low Risk": "#27ae60"
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Risk score histogram
            fig_hist = px.histogram(
                filtered_df,
                x="risk_score",
                nbins=30,
                title="Risk Score Distribution",
                labels={"risk_score": "Risk Score", "count": "Number of Deals"}
            )
            fig_hist.add_vline(x=0.6, line_dash="dash", line_color="red",
                             annotation_text="High Risk Threshold")
            st.plotly_chart(fig_hist, use_container_width=True)

        # Risk by segment
        st.subheader("Risk by Segment")
        col1, col2 = st.columns(2)

        with col1:
            # By region
            region_risk = filtered_df.groupby("region").agg({
                "risk_score": "mean",
                "deal_amount": "sum",
                "deal_id": "count"
            }).round(3)
            region_risk.columns = ["Avg Risk", "Total Value", "Deal Count"]

            fig_region = px.bar(
                region_risk.reset_index(),
                x="region",
                y="Avg Risk",
                color="Avg Risk",
                title="Average Risk by Region",
                color_continuous_scale="RdYlGn_r"
            )
            st.plotly_chart(fig_region, use_container_width=True)

        with col2:
            # By industry
            industry_risk = filtered_df.groupby("industry").agg({
                "risk_score": "mean",
                "deal_amount": "sum"
            }).round(3)

            fig_industry = px.bar(
                industry_risk.reset_index(),
                x="industry",
                y="risk_score",
                color="risk_score",
                title="Average Risk by Industry",
                color_continuous_scale="RdYlGn_r"
            )
            st.plotly_chart(fig_industry, use_container_width=True)

        # LLM Analysis for Risk Overview
        if llm:
            st.markdown("---")
            st.markdown("**Why This Risk Distribution?**")
            with st.spinner("Analyzing..."):
                try:
                    risk_data = {
                        "high_risk": (filtered_df["risk_category"] == "High Risk").sum(),
                        "high_risk_value": filtered_df[filtered_df["risk_category"] == "High Risk"]["deal_amount"].sum(),
                        "medium_risk": (filtered_df["risk_category"] == "Medium Risk").sum(),
                        "low_risk": (filtered_df["risk_category"] == "Low Risk").sum(),
                        "avg_risk": filtered_df["risk_score"].mean()
                    }
                    analysis = llm.analyze_risk_distribution(risk_data)
                    st.info(analysis)
                except Exception as e:
                    st.caption(f"Analysis unavailable: {e}")

    # Tab 2: Pipeline Analysis
    with tab2:
        # Win rate trend over time
        filtered_df["month"] = filtered_df["created_date"].dt.to_period("M").astype(str)
        monthly_metrics = filtered_df.groupby("month").agg({
            "target": "mean",
            "deal_amount": ["sum", "count"],
            "risk_score": "mean"
        }).round(3)
        monthly_metrics.columns = ["Win Rate", "Revenue", "Deal Count", "Avg Risk"]
        monthly_metrics = monthly_metrics.reset_index()

        # Create subplot
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=monthly_metrics["month"], y=monthly_metrics["Win Rate"],
                      name="Win Rate", line=dict(color="green")),
            secondary_y=False
        )
        fig.add_trace(
            go.Bar(x=monthly_metrics["month"], y=monthly_metrics["Deal Count"],
                  name="Deal Count", opacity=0.3),
            secondary_y=True
        )

        fig.update_layout(title="Win Rate Trend Over Time")
        fig.update_yaxes(title_text="Win Rate", secondary_y=False)
        fig.update_yaxes(title_text="Deal Count", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

        # Pipeline by stage
        col1, col2 = st.columns(2)

        with col1:
            stage_metrics = filtered_df.groupby("deal_stage").agg({
                "target": "mean",
                "deal_amount": "sum",
                "deal_id": "count"
            }).round(3)
            stage_metrics.columns = ["Win Rate", "Total Value", "Count"]

            fig_stage = px.funnel(
                stage_metrics.reset_index(),
                x="Count",
                y="deal_stage",
                title="Deal Funnel by Stage"
            )
            st.plotly_chart(fig_stage, use_container_width=True)

        with col2:
            # Lead source performance
            source_perf = filtered_df.groupby("lead_source").agg({
                "target": "mean",
                "deal_amount": "mean",
                "risk_score": "mean"
            }).round(3)
            source_perf.columns = ["Win Rate", "Avg Deal Size", "Avg Risk"]

            fig_source = px.scatter(
                source_perf.reset_index(),
                x="Avg Deal Size",
                y="Win Rate",
                size="Win Rate",
                color="lead_source",
                title="Lead Source Performance"
            )
            st.plotly_chart(fig_source, use_container_width=True)

        # LLM Analysis for Pipeline
        if llm:
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Why This Win Rate Trend?**")
                with st.spinner("Analyzing..."):
                    try:
                        quarterly_df = filtered_df.copy()
                        quarterly_df["quarter"] = quarterly_df["created_date"].dt.to_period("Q").astype(str)
                        trend_data = quarterly_df.groupby("quarter").agg({
                            "target": "mean",
                            "deal_id": "count"
                        }).round(3)
                        trend_data.columns = ["Win Rate", "Deal Count"]
                        analysis = llm.analyze_pipeline_trend(trend_data)
                        st.info(analysis)
                    except Exception as e:
                        st.caption(f"Analysis unavailable: {e}")

            with col2:
                st.markdown("**Why Lead Sources Differ?**")
                with st.spinner("Analyzing..."):
                    try:
                        analysis = llm.analyze_lead_source(source_perf)
                        st.info(analysis)
                    except Exception as e:
                        st.caption(f"Analysis unavailable: {e}")

    # Tab 3: Rep Performance
    with tab3:
        st.subheader("Sales Rep Performance")

        rep_metrics = filtered_df.groupby("sales_rep_id").agg({
            "target": "mean",
            "deal_amount": ["sum", "mean", "count"],
            "risk_score": "mean",
            "sales_cycle_days": "mean"
        }).round(3)
        rep_metrics.columns = ["Win Rate", "Total Revenue", "Avg Deal Size",
                               "Deal Count", "Avg Risk", "Avg Cycle Days"]
        rep_metrics = rep_metrics.sort_values("Total Revenue", ascending=False)

        # Rep leaderboard
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top Performers (by Revenue)**")
            top_reps = rep_metrics.head(10).reset_index()
            fig_top = px.bar(
                top_reps,
                x="sales_rep_id",
                y="Total Revenue",
                color="Win Rate",
                title="Top 10 Reps by Revenue"
            )
            st.plotly_chart(fig_top, use_container_width=True)

        with col2:
            st.markdown("**Rep Risk Profile**")
            fig_scatter = px.scatter(
                rep_metrics.reset_index(),
                x="Win Rate",
                y="Avg Risk",
                size="Deal Count",
                hover_name="sales_rep_id",
                title="Rep Win Rate vs Avg Risk"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Rep detail table
        st.markdown("**Rep Performance Table**")
        st.dataframe(
            rep_metrics.style.format({
                "Win Rate": "{:.1%}",
                "Total Revenue": "${:,.0f}",
                "Avg Deal Size": "${:,.0f}",
                "Avg Risk": "{:.1%}",
                "Avg Cycle Days": "{:.0f}"
            }),
            use_container_width=True
        )

        # LLM Analysis for Rep Performance
        if llm:
            st.markdown("---")
            st.markdown("**Why Rep Performance Varies?**")
            with st.spinner("Analyzing..."):
                try:
                    analysis = llm.analyze_rep_performance(rep_metrics.reset_index())
                    st.info(analysis)
                except Exception as e:
                    st.caption(f"Analysis unavailable: {e}")

    # Tab 4: Deal Explorer
    with tab4:
        st.subheader("Deal Explorer")

        # Sort options
        sort_by = st.selectbox(
            "Sort by",
            ["Risk Score (High to Low)", "Deal Amount (High to Low)",
             "Created Date (Recent)", "Sales Cycle (Longest)"]
        )

        if sort_by == "Risk Score (High to Low)":
            display_df = filtered_df.sort_values("risk_score", ascending=False)
        elif sort_by == "Deal Amount (High to Low)":
            display_df = filtered_df.sort_values("deal_amount", ascending=False)
        elif sort_by == "Created Date (Recent)":
            display_df = filtered_df.sort_values("created_date", ascending=False)
        else:
            display_df = filtered_df.sort_values("sales_cycle_days", ascending=False)

        # Display columns
        display_cols = [
            "deal_id", "risk_score", "risk_category", "win_probability",
            "deal_amount", "industry", "region", "sales_rep_id",
            "lead_source", "deal_stage", "sales_cycle_days", "outcome"
        ]

        st.dataframe(
            display_df[display_cols].head(100).style.format({
                "risk_score": "{:.1%}",
                "win_probability": "{:.1%}",
                "deal_amount": "${:,.0f}"
            }).apply(
                lambda x: ["background-color: #ffcccc" if v == "High Risk"
                          else "background-color: #ffffcc" if v == "Medium Risk"
                          else "background-color: #ccffcc" if v == "Low Risk"
                          else "" for v in x],
                subset=["risk_category"]
            ),
            use_container_width=True
        )

        # Deal detail view
        st.markdown("---")
        st.subheader("Deal Detail View")
        selected_deal = st.selectbox(
            "Select a deal to view details",
            display_df["deal_id"].head(50).tolist()
        )

        if selected_deal:
            deal = filtered_df[filtered_df["deal_id"] == selected_deal].iloc[0]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Deal Info**")
                st.write(f"Amount: ${deal['deal_amount']:,.0f}")
                st.write(f"Industry: {deal['industry']}")
                st.write(f"Region: {deal['region']}")
                st.write(f"Product: {deal['product_type']}")

            with col2:
                st.markdown("**Risk Assessment**")
                risk_color = "red" if deal["risk_score"] > 0.6 else "orange" if deal["risk_score"] > 0.3 else "green"
                st.markdown(f"Risk Score: <span style='color:{risk_color};font-size:24px;'>{deal['risk_score']:.1%}</span>",
                           unsafe_allow_html=True)
                st.write(f"Win Probability: {deal['win_probability']:.1%}")
                st.write(f"Category: {deal['risk_category']}")

            with col3:
                st.markdown("**Context**")
                st.write(f"Sales Rep: {deal['sales_rep_id']}")
                st.write(f"Lead Source: {deal['lead_source']}")
                st.write(f"Cycle Days: {deal['sales_cycle_days']}")
                st.write(f"Actual Outcome: {deal['outcome']}")

            # LLM Explanation for deal risk
            if llm and deal["risk_score"] > 0.3:
                st.markdown("---")
                st.markdown("**Why This Deal Is At Risk?**")
                with st.spinner("Analyzing deal..."):
                    try:
                        explanation = llm.explain_deal_risk(deal)
                        st.warning(explanation)
                    except Exception as e:
                        st.caption(f"Could not generate explanation: {e}")

    # Tab 5: Model Insights
    with tab5:
        st.subheader("Model Performance & Insights")

        if metrics:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Model Metrics**")
                model_metrics = metrics.get("model_metrics", {})
                auc_val = model_metrics.get('auc', 0)
                acc_val = model_metrics.get('accuracy', 0)
                prec_val = model_metrics.get('precision', 0)
                rec_val = model_metrics.get('recall', 0)
                f1_val = model_metrics.get('f1', 0)
                st.write(f"AUC: {auc_val:.4f}" if isinstance(auc_val, (int, float)) else "AUC: N/A")
                st.write(f"Accuracy: {acc_val:.4f}" if isinstance(acc_val, (int, float)) else "Accuracy: N/A")
                st.write(f"Precision: {prec_val:.4f}" if isinstance(prec_val, (int, float)) else "Precision: N/A")
                st.write(f"Recall: {rec_val:.4f}" if isinstance(rec_val, (int, float)) else "Recall: N/A")
                st.write(f"F1 Score: {f1_val:.4f}" if isinstance(f1_val, (int, float)) else "F1: N/A")

            with col2:
                st.markdown("**Cross-Validation**")
                cv_results = metrics.get("cv_results", {})
                mean_auc = cv_results.get('mean_auc', 0)
                std_auc = cv_results.get('std_auc', 0)
                st.write(f"Mean AUC: {mean_auc:.4f}" if isinstance(mean_auc, (int, float)) else "Mean AUC: N/A")
                st.write(f"Std AUC: {std_auc:.4f}" if isinstance(std_auc, (int, float)) else "Std AUC: N/A")

            # Feature importance
            st.markdown("---")
            st.markdown("**Feature Importance**")

            importance_path = Path("models/risk_scorer/feature_importance.csv")
            if importance_path.exists():
                importance_df = pd.read_csv(importance_path)

                fig_importance = px.bar(
                    importance_df.head(15),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top 15 Features by Importance"
                )
                fig_importance.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_importance, use_container_width=True)

                # LLM Analysis for Feature Importance
                if llm:
                    st.markdown("**Why These Features Matter?**")
                    with st.spinner("Analyzing..."):
                        try:
                            analysis = llm.analyze_feature_importance(importance_df)
                            st.info(analysis)
                        except Exception as e:
                            st.caption(f"Analysis unavailable: {e}")
            else:
                st.info("Feature importance not available. Run training first.")

            # LLM Action Items
            if llm:
                st.markdown("---")
                st.markdown("**Action Items**")
                with st.spinner("Generating actions..."):
                    try:
                        rec_metrics = {
                            "win_rate": filtered_df["target"].mean(),
                            "high_risk_count": (filtered_df["risk_category"] == "High Risk").sum(),
                            "high_risk_value": filtered_df[filtered_df["risk_category"] == "High Risk"]["deal_amount"].sum(),
                            "avg_cycle": filtered_df["sales_cycle_days"].mean()
                        }
                        actions = llm.generate_action_items(rec_metrics)
                        st.success(actions)
                    except Exception as e:
                        st.caption(f"Could not generate actions: {e}")
        else:
            st.warning("Model metrics not found. Please run `python train.py` first.")

    # Tab 6: Alerts
    with tab6:
        st.subheader("Alerts - Immediate Attention Required")

        # High Risk Deals Alert
        st.markdown("**High Risk Deals (Risk > 70%)**")
        alerts_df = filtered_df.copy()
        high_risk_deals = alerts_df[alerts_df["risk_score"] > 0.7].sort_values("risk_score", ascending=False)

        if len(high_risk_deals) > 0:
            st.error(f"{len(high_risk_deals)} deals need immediate attention - Total value: ${high_risk_deals['deal_amount'].sum():,.0f}")

            for idx, deal in high_risk_deals.head(10).iterrows():
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
                    with col1:
                        st.write(f"**{deal['deal_id']}** - {deal['industry']}")
                    with col2:
                        st.write(f"${deal['deal_amount']:,.0f}")
                    with col3:
                        st.write(f"Risk: {deal['risk_score']:.0%}")
                    with col4:
                        st.write(f"Rep: {deal['sales_rep_id']}")

                    # LLM explanation for each high-risk deal
                    if llm:
                        with st.expander("Why at risk?"):
                            try:
                                explanation = llm.explain_deal_risk(deal)
                                st.warning(explanation)
                            except:
                                st.caption("Analysis unavailable")
                    st.markdown("---")
        else:
            st.success("No high-risk deals requiring immediate attention")

        # Stale Deals Alert
        st.markdown("**Stale Deals (Cycle > 90 days)**")
        stale_deals = alerts_df[alerts_df["sales_cycle_days"] > 90].sort_values("sales_cycle_days", ascending=False)

        if len(stale_deals) > 0:
            st.warning(f"{len(stale_deals)} deals stuck in pipeline > 90 days")

            stale_display = stale_deals[["deal_id", "deal_amount", "sales_cycle_days", "deal_stage", "sales_rep_id"]].head(10)
            stale_display.columns = ["Deal ID", "Amount", "Days in Pipeline", "Stage", "Rep"]
            st.dataframe(
                stale_display.style.format({"Amount": "${:,.0f}"}),
                use_container_width=True
            )
        else:
            st.success("No stale deals in pipeline")

        # Underperforming Reps Alert
        st.markdown("**Underperforming Reps (Win Rate < 35%)**")
        rep_performance = alerts_df.groupby("sales_rep_id").agg({
            "target": "mean",
            "deal_id": "count",
            "deal_amount": "sum"
        }).round(3)
        rep_performance.columns = ["Win Rate", "Deals", "Total Value"]
        underperforming = rep_performance[(rep_performance["Win Rate"] < 0.35) & (rep_performance["Deals"] >= 20)]

        if len(underperforming) > 0:
            st.warning(f"{len(underperforming)} reps need coaching")
            st.dataframe(
                underperforming.style.format({
                    "Win Rate": "{:.1%}",
                    "Total Value": "${:,.0f}"
                }),
                use_container_width=True
            )
        else:
            st.success("All reps performing above threshold")

        # Win Rate Drop Alert
        st.markdown("**Win Rate Trend Alert**")
        alerts_df["quarter"] = alerts_df["created_date"].dt.to_period("Q").astype(str)
        quarterly_wr = alerts_df.groupby("quarter")["target"].mean()

        if len(quarterly_wr) >= 2:
            recent_wr = quarterly_wr.iloc[-1]
            prev_wr = quarterly_wr.iloc[-2]
            change = recent_wr - prev_wr

            if change < -0.05:
                st.error(f"Win rate dropped {abs(change):.1%} from {prev_wr:.1%} to {recent_wr:.1%}")
                if llm:
                    with st.expander("Why did win rate drop?"):
                        try:
                            trend_df = pd.DataFrame({"Win Rate": quarterly_wr})
                            analysis = llm.analyze_pipeline_trend(trend_df)
                            st.info(analysis)
                        except:
                            st.caption("Analysis unavailable")
            elif change > 0.05:
                st.success(f"Win rate improved {change:.1%} from {prev_wr:.1%} to {recent_wr:.1%}")
            else:
                st.info(f"Win rate stable at {recent_wr:.1%}")

    # Tab 7: Forecast
    with tab7:
        st.subheader("Revenue Forecast")

        # Historical data for forecast
        forecast_df = filtered_df.copy()
        forecast_df["month"] = forecast_df["created_date"].dt.to_period("M")
        monthly_revenue = forecast_df.groupby("month").agg({
            "deal_amount": "sum",
            "target": "mean",
            "deal_id": "count"
        }).round(2)
        monthly_revenue.columns = ["Revenue", "Win Rate", "Deals"]
        monthly_revenue = monthly_revenue.reset_index()
        monthly_revenue["month"] = monthly_revenue["month"].astype(str)

        # Simple forecast based on trends
        st.markdown("**Historical Revenue Trend**")
        fig_revenue = px.line(
            monthly_revenue,
            x="month",
            y="Revenue",
            title="Monthly Revenue",
            markers=True
        )
        st.plotly_chart(fig_revenue, use_container_width=True)

        # Forecast calculation
        st.markdown("**Next Period Forecast**")

        # Calculate averages and trends
        avg_revenue = monthly_revenue["Revenue"].mean()
        avg_win_rate = monthly_revenue["Win Rate"].mean()
        avg_deals = monthly_revenue["Deals"].mean()

        # Recent trend (last 3 months)
        if len(monthly_revenue) >= 3:
            recent_revenue = monthly_revenue["Revenue"].tail(3).mean()
            recent_win_rate = monthly_revenue["Win Rate"].tail(3).mean()
            trend = (recent_revenue - avg_revenue) / avg_revenue if avg_revenue > 0 else 0
        else:
            recent_revenue = avg_revenue
            recent_win_rate = avg_win_rate
            trend = 0

        # Forecast scenarios
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Conservative**")
            conservative = recent_revenue * 0.9
            st.metric("Predicted Revenue", f"${conservative:,.0f}")
            st.caption("10% below recent average")

        with col2:
            st.markdown("**Expected**")
            expected = recent_revenue * (1 + trend * 0.5)
            st.metric("Predicted Revenue", f"${expected:,.0f}")
            st.caption("Based on current trend")

        with col3:
            st.markdown("**Optimistic**")
            optimistic = recent_revenue * 1.15
            st.metric("Predicted Revenue", f"${optimistic:,.0f}")
            st.caption("15% above recent average")

        # Pipeline-based forecast
        st.markdown("---")
        st.markdown("**Pipeline-Based Forecast**")

        # Current open deals (not closed)
        open_deals = forecast_df[forecast_df["deal_stage"] != "Closed"]
        total_pipeline_value = open_deals["deal_amount"].sum()
        weighted_pipeline = (open_deals["deal_amount"] * open_deals["win_probability"]).sum()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Pipeline Value", f"${total_pipeline_value:,.0f}")

        with col2:
            st.metric("Weighted Pipeline", f"${weighted_pipeline:,.0f}")
            st.caption("Adjusted by win probability")

        with col3:
            expected_closed = weighted_pipeline * avg_win_rate
            st.metric("Expected to Close", f"${expected_closed:,.0f}")

        # Forecast by segment
        st.markdown("---")
        st.markdown("**Forecast by Segment**")

        col1, col2 = st.columns(2)

        with col1:
            # By region
            region_forecast = forecast_df.groupby("region").agg({
                "deal_amount": "sum",
                "win_probability": "mean"
            }).round(2)
            region_forecast["Expected Revenue"] = region_forecast["deal_amount"] * region_forecast["win_probability"]
            region_forecast = region_forecast.sort_values("Expected Revenue", ascending=False)

            fig_region = px.bar(
                region_forecast.reset_index(),
                x="region",
                y="Expected Revenue",
                title="Expected Revenue by Region"
            )
            st.plotly_chart(fig_region, use_container_width=True)

        with col2:
            # By industry
            industry_forecast = forecast_df.groupby("industry").agg({
                "deal_amount": "sum",
                "win_probability": "mean"
            }).round(2)
            industry_forecast["Expected Revenue"] = industry_forecast["deal_amount"] * industry_forecast["win_probability"]
            industry_forecast = industry_forecast.sort_values("Expected Revenue", ascending=False)

            fig_industry = px.bar(
                industry_forecast.reset_index(),
                x="industry",
                y="Expected Revenue",
                title="Expected Revenue by Industry"
            )
            st.plotly_chart(fig_industry, use_container_width=True)

        # LLM Forecast Analysis
        if llm:
            st.markdown("---")
            st.markdown("**Forecast Analysis**")
            with st.spinner("Analyzing forecast..."):
                try:
                    # Calculate high risk deals for this tab
                    high_risk_in_forecast = forecast_df[forecast_df["risk_score"] > 0.7]
                    high_risk_count = len(high_risk_in_forecast)
                    high_risk_value = high_risk_in_forecast["deal_amount"].sum()

                    forecast_prompt = f"""
Current pipeline data:
- Total Pipeline: ${total_pipeline_value:,.0f}
- Weighted Pipeline: ${weighted_pipeline:,.0f}
- Average Win Rate: {avg_win_rate:.0%}
- Recent Revenue Trend: {trend:+.1%}
- High Risk Deals: {high_risk_count} worth ${high_risk_value:,.0f}

In 4 bullet points:
- What is the realistic revenue forecast?
- What risks could reduce this forecast?
- What actions could improve the forecast?
- Which segment has highest growth potential?
"""
                    analysis = llm._call_llm(forecast_prompt)
                    st.info(analysis)
                except Exception as e:
                    st.caption(f"Analysis unavailable: {e}")


if __name__ == "__main__":
    main()
