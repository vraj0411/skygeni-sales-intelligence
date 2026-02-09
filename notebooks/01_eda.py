#!/usr/bin/env python3
"""
SkyGeni Sales Intelligence - Exploratory Data Analysis

This script performs comprehensive EDA on the sales data.
Run with: python notebooks/01_eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Create output directory
output_dir = Path("reports/figures")
output_dir.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load the sales data."""
    df = pd.read_csv("data/raw/skygeni_sales_data.csv")
    df["created_date"] = pd.to_datetime(df["created_date"])
    df["closed_date"] = pd.to_datetime(df["closed_date"])
    df["target"] = (df["outcome"] == "Won").astype(int)
    return df


def basic_stats(df):
    """Print basic statistics."""
    print("=" * 60)
    print("BASIC STATISTICS")
    print("=" * 60)

    print(f"\nTotal Deals: {len(df):,}")
    print(f"Date Range: {df['created_date'].min().date()} to {df['created_date'].max().date()}")

    print(f"\nOutcome Distribution:")
    print(df["outcome"].value_counts())
    print(f"Win Rate: {df['target'].mean():.2%}")

    print(f"\nDeal Amount:")
    print(f"  - Mean: ${df['deal_amount'].mean():,.2f}")
    print(f"  - Median: ${df['deal_amount'].median():,.2f}")
    print(f"  - Min: ${df['deal_amount'].min():,.2f}")
    print(f"  - Max: ${df['deal_amount'].max():,.2f}")

    print(f"\nSales Cycle Days:")
    print(f"  - Mean: {df['sales_cycle_days'].mean():.1f}")
    print(f"  - Median: {df['sales_cycle_days'].median():.1f}")

    print(f"\nUnique Values:")
    print(f"  - Sales Reps: {df['sales_rep_id'].nunique()}")
    print(f"  - Industries: {df['industry'].nunique()}")
    print(f"  - Regions: {df['region'].nunique()}")
    print(f"  - Product Types: {df['product_type'].nunique()}")
    print(f"  - Lead Sources: {df['lead_source'].nunique()}")
    print(f"  - Deal Stages: {df['deal_stage'].nunique()}")


def insight_1_win_rate_by_segment(df):
    """
    INSIGHT 1: Win Rate Analysis by Segment

    Business Question: Which segments are performing best/worst?
    """
    print("\n" + "=" * 60)
    print("INSIGHT 1: Win Rate by Segment")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # By Region
    region_wr = df.groupby("region")["target"].agg(["mean", "count"]).round(3)
    region_wr.columns = ["Win Rate", "Count"]
    region_wr = region_wr.sort_values("Win Rate", ascending=True)

    ax = axes[0, 0]
    bars = ax.barh(region_wr.index, region_wr["Win Rate"])
    ax.set_xlabel("Win Rate")
    ax.set_title("Win Rate by Region")
    ax.axvline(df["target"].mean(), color="red", linestyle="--", label="Overall")
    for bar, count in zip(bars, region_wr["Count"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f"n={count}", va="center")

    print("\nBy Region:")
    print(region_wr)

    # By Industry
    industry_wr = df.groupby("industry")["target"].agg(["mean", "count"]).round(3)
    industry_wr.columns = ["Win Rate", "Count"]
    industry_wr = industry_wr.sort_values("Win Rate", ascending=True)

    ax = axes[0, 1]
    bars = ax.barh(industry_wr.index, industry_wr["Win Rate"])
    ax.set_xlabel("Win Rate")
    ax.set_title("Win Rate by Industry")
    ax.axvline(df["target"].mean(), color="red", linestyle="--")

    print("\nBy Industry:")
    print(industry_wr)

    # By Lead Source
    source_wr = df.groupby("lead_source")["target"].agg(["mean", "count"]).round(3)
    source_wr.columns = ["Win Rate", "Count"]
    source_wr = source_wr.sort_values("Win Rate", ascending=True)

    ax = axes[1, 0]
    bars = ax.barh(source_wr.index, source_wr["Win Rate"])
    ax.set_xlabel("Win Rate")
    ax.set_title("Win Rate by Lead Source")
    ax.axvline(df["target"].mean(), color="red", linestyle="--")

    print("\nBy Lead Source:")
    print(source_wr)

    # By Product Type
    product_wr = df.groupby("product_type")["target"].agg(["mean", "count"]).round(3)
    product_wr.columns = ["Win Rate", "Count"]
    product_wr = product_wr.sort_values("Win Rate", ascending=True)

    ax = axes[1, 1]
    bars = ax.barh(product_wr.index, product_wr["Win Rate"])
    ax.set_xlabel("Win Rate")
    ax.set_title("Win Rate by Product Type")
    ax.axvline(df["target"].mean(), color="red", linestyle="--")

    print("\nBy Product Type:")
    print(product_wr)

    plt.tight_layout()
    plt.savefig(output_dir / "insight_1_win_rate_by_segment.png", dpi=150)
    plt.close()

    # Business Insight
    print("\nBUSINESS INSIGHT:")
    best_region = region_wr["Win Rate"].idxmax()
    worst_region = region_wr["Win Rate"].idxmin()
    print(f"  - Best performing region: {best_region} ({region_wr.loc[best_region, 'Win Rate']:.1%})")
    print(f"  - Worst performing region: {worst_region} ({region_wr.loc[worst_region, 'Win Rate']:.1%})")
    print(f"  - Action: Investigate what {best_region} team does differently")


def insight_2_time_trends(df):
    """
    INSIGHT 2: Win Rate Trend Over Time

    Business Question: Is win rate dropping as the CRO complained?
    """
    print("\n" + "=" * 60)
    print("INSIGHT 2: Win Rate Trend Over Time")
    print("=" * 60)

    df["quarter"] = df["created_date"].dt.to_period("Q").astype(str)
    df["month"] = df["created_date"].dt.to_period("M").astype(str)

    quarterly = df.groupby("quarter").agg({
        "target": "mean",
        "deal_id": "count",
        "deal_amount": "sum"
    }).round(3)
    quarterly.columns = ["Win Rate", "Deal Count", "Total Value"]

    print("\nQuarterly Metrics:")
    print(quarterly)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Win rate trend
    ax = axes[0]
    ax.plot(quarterly.index, quarterly["Win Rate"], marker="o", linewidth=2, markersize=8)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Win Rate")
    ax.set_title("Win Rate Trend by Quarter")
    ax.tick_params(axis="x", rotation=45)

    # Add trend line
    x_numeric = np.arange(len(quarterly))
    z = np.polyfit(x_numeric, quarterly["Win Rate"], 1)
    p = np.poly1d(z)
    ax.plot(quarterly.index, p(x_numeric), "--", color="red", label=f"Trend (slope={z[0]:.4f})")
    ax.legend()

    # Deal volume
    ax = axes[1]
    ax.bar(quarterly.index, quarterly["Deal Count"], alpha=0.7)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Deal Count")
    ax.set_title("Deal Volume by Quarter")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "insight_2_time_trends.png", dpi=150)
    plt.close()

    # Calculate recent vs historical
    quarters = quarterly.index.tolist()
    if len(quarters) >= 4:
        recent_wr = quarterly.loc[quarters[-2:]]["Win Rate"].mean()
        historical_wr = quarterly.loc[quarters[:-2]]["Win Rate"].mean()
        change = recent_wr - historical_wr

        print(f"\nBUSINESS INSIGHT:")
        print(f"  - Recent 2Q win rate: {recent_wr:.1%}")
        print(f"  - Historical win rate: {historical_wr:.1%}")
        print(f"  - Change: {change:+.1%}")

        if change < -0.05:
            print(f"  - ALERT: Win rate has dropped significantly!")
            print(f"  - Action: Deep dive into recent quarter deal characteristics")


def insight_3_sales_cycle_impact(df):
    """
    INSIGHT 3: Sales Cycle Impact on Win Rate

    Business Question: Are longer deals less likely to close?
    """
    print("\n" + "=" * 60)
    print("INSIGHT 3: Sales Cycle Impact")
    print("=" * 60)

    # Bin sales cycle days
    df["cycle_bin"] = pd.cut(
        df["sales_cycle_days"],
        bins=[0, 15, 30, 60, 90, float("inf")],
        labels=["0-15", "16-30", "31-60", "61-90", "90+"]
    )

    cycle_analysis = df.groupby("cycle_bin").agg({
        "target": ["mean", "count"],
        "deal_amount": "mean"
    }).round(3)
    cycle_analysis.columns = ["Win Rate", "Count", "Avg Amount"]

    print("\nWin Rate by Sales Cycle Length:")
    print(cycle_analysis)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Win rate by cycle
    ax = axes[0]
    ax.bar(cycle_analysis.index, cycle_analysis["Win Rate"])
    ax.set_xlabel("Sales Cycle (Days)")
    ax.set_ylabel("Win Rate")
    ax.set_title("Win Rate by Sales Cycle Length")
    ax.axhline(df["target"].mean(), color="red", linestyle="--", label="Overall")
    ax.legend()

    # Box plot
    ax = axes[1]
    df.boxplot(column="sales_cycle_days", by="outcome", ax=ax)
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Sales Cycle (Days)")
    ax.set_title("Sales Cycle Distribution by Outcome")
    plt.suptitle("")

    plt.tight_layout()
    plt.savefig(output_dir / "insight_3_sales_cycle.png", dpi=150)
    plt.close()

    won_cycle = df[df["target"] == 1]["sales_cycle_days"].mean()
    lost_cycle = df[df["target"] == 0]["sales_cycle_days"].mean()

    print(f"\nBUSINESS INSIGHT:")
    print(f"  - Average cycle for Won deals: {won_cycle:.1f} days")
    print(f"  - Average cycle for Lost deals: {lost_cycle:.1f} days")
    print(f"  - Deals taking >60 days have {cycle_analysis.loc['61-90', 'Win Rate']:.1%} win rate")
    print(f"  - Action: Focus on accelerating deals stuck in pipeline >30 days")


def custom_metric_1_deal_health_score(df):
    """
    CUSTOM METRIC 1: Deal Health Score

    A composite score based on multiple factors that predicts deal success.
    """
    print("\n" + "=" * 60)
    print("CUSTOM METRIC 1: Deal Health Score")
    print("=" * 60)

    # Calculate rep win rates
    rep_wr = df.groupby("sales_rep_id")["target"].mean()
    df["rep_win_rate"] = df["sales_rep_id"].map(rep_wr)

    # Calculate industry averages
    industry_cycle = df.groupby("industry")["sales_cycle_days"].mean()
    df["industry_avg_cycle"] = df["industry"].map(industry_cycle)

    # Deal Health Score components:
    # 1. Rep performance (40%)
    # 2. Cycle efficiency (30%) - faster than industry avg is better
    # 3. Deal stage progression (30%)

    stage_scores = {
        "Qualified": 0.2, "Demo": 0.4, "Proposal": 0.6,
        "Negotiation": 0.8, "Closed": 1.0
    }
    df["stage_score"] = df["deal_stage"].map(stage_scores)

    cycle_efficiency = (df["industry_avg_cycle"] / df["sales_cycle_days"]).clip(0, 2) / 2

    df["deal_health_score"] = (
        df["rep_win_rate"] * 0.4 +
        cycle_efficiency * 0.3 +
        df["stage_score"] * 0.3
    ).round(3)

    # Analyze correlation with outcome
    correlation = df["deal_health_score"].corr(df["target"])

    print(f"\nDeal Health Score Statistics:")
    print(df["deal_health_score"].describe())
    print(f"\nCorrelation with Win/Loss: {correlation:.3f}")

    # Bin and analyze
    df["health_bin"] = pd.qcut(df["deal_health_score"], q=5, labels=["Very Low", "Low", "Medium", "High", "Very High"])
    health_analysis = df.groupby("health_bin")["target"].agg(["mean", "count"])
    health_analysis.columns = ["Win Rate", "Count"]

    print("\nWin Rate by Deal Health Score:")
    print(health_analysis)

    print(f"\nBUSINESS INSIGHT:")
    print(f"  - Health score has {correlation:.2f} correlation with outcomes")
    print(f"  - 'Very High' health deals win at {health_analysis.loc['Very High', 'Win Rate']:.1%}")
    print(f"  - 'Very Low' health deals win at {health_analysis.loc['Very Low', 'Win Rate']:.1%}")
    print(f"  - Action: Prioritize improving health score factors for at-risk deals")

    return df


def custom_metric_2_rep_consistency_score(df):
    """
    CUSTOM METRIC 2: Rep Consistency Score

    Measures how consistent a rep's performance is (lower variance = more reliable)
    """
    print("\n" + "=" * 60)
    print("CUSTOM METRIC 2: Rep Consistency Score")
    print("=" * 60)

    # Calculate rep statistics
    rep_stats = df.groupby("sales_rep_id").agg({
        "target": ["mean", "std", "count"],
        "deal_amount": ["mean", "sum"]
    }).round(3)
    rep_stats.columns = ["Win Rate", "Win Std", "Deal Count", "Avg Amount", "Total Revenue"]

    # Consistency = 1 - normalized std
    # Higher is better (more consistent)
    rep_stats["Consistency Score"] = (1 - rep_stats["Win Std"].fillna(0) / rep_stats["Win Rate"].clip(lower=0.01)).clip(0, 1)

    # Filter to reps with enough deals
    rep_stats_filtered = rep_stats[rep_stats["Deal Count"] >= 50]

    print("\nTop 10 Most Consistent Reps (min 50 deals):")
    print(rep_stats_filtered.sort_values("Consistency Score", ascending=False).head(10)[
        ["Win Rate", "Deal Count", "Consistency Score", "Total Revenue"]
    ])

    print("\nBottom 10 Least Consistent Reps (min 50 deals):")
    print(rep_stats_filtered.sort_values("Consistency Score", ascending=True).head(10)[
        ["Win Rate", "Deal Count", "Consistency Score", "Total Revenue"]
    ])

    print(f"\nBUSINESS INSIGHT:")
    print(f"  - Consistent reps are more predictable for forecasting")
    print(f"  - Pair inconsistent reps with consistent mentors")
    print(f"  - Action: Use consistency score in pipeline forecasting models")


def main():
    """Run full EDA."""
    print("SkyGeni Sales Intelligence - Exploratory Data Analysis")
    print("=" * 60)

    # Load data
    df = load_data()

    # Basic stats
    basic_stats(df)

    # Business Insights (required: at least 3)
    insight_1_win_rate_by_segment(df)
    insight_2_time_trends(df)
    insight_3_sales_cycle_impact(df)

    # Custom Metrics (required: at least 2)
    df = custom_metric_1_deal_health_score(df)
    custom_metric_2_rep_consistency_score(df)

    print("\n" + "=" * 60)
    print("EDA Complete! Figures saved to reports/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
