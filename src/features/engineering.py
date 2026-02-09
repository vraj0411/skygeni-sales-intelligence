"""Feature engineering for SkyGeni Sales Intelligence."""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create advanced features for deal analysis."""

    def __init__(self):
        self.rep_stats: Optional[pd.DataFrame] = None
        self.industry_stats: Optional[pd.DataFrame] = None
        self.lead_source_stats: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Calculate aggregate statistics from training data."""

        # Rep-level statistics
        self.rep_stats = df.groupby("sales_rep_id").agg({
            "target": ["mean", "count", "std"],
            "deal_amount": "mean",
            "sales_cycle_days": "mean"
        }).round(4)
        self.rep_stats.columns = [
            "rep_win_rate", "rep_deal_count", "rep_win_std",
            "rep_avg_deal_amount", "rep_avg_cycle"
        ]
        self.rep_stats["rep_win_std"] = self.rep_stats["rep_win_std"].fillna(0)

        # Industry-level statistics
        self.industry_stats = df.groupby("industry").agg({
            "target": "mean",
            "deal_amount": "mean",
            "sales_cycle_days": "mean"
        }).round(4)
        self.industry_stats.columns = [
            "industry_win_rate", "industry_avg_amount", "industry_avg_cycle"
        ]

        # Lead source statistics
        self.lead_source_stats = df.groupby("lead_source").agg({
            "target": "mean",
            "deal_amount": "mean"
        }).round(4)
        self.lead_source_stats.columns = ["lead_source_win_rate", "lead_source_avg_amount"]

        logger.info("Feature statistics computed")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features to dataframe."""
        df = df.copy()

        # Merge rep statistics
        df = df.merge(
            self.rep_stats, left_on="sales_rep_id", right_index=True, how="left"
        )

        # Merge industry statistics
        df = df.merge(
            self.industry_stats, left_on="industry", right_index=True, how="left"
        )

        # Merge lead source statistics
        df = df.merge(
            self.lead_source_stats, left_on="lead_source", right_index=True, how="left"
        )

        # Custom metrics from the challenge

        # 1. Deal Health Score
        df["deal_health_score"] = self._calculate_deal_health_score(df)

        # 2. Pipeline Momentum Index (at deal level - relative positioning)
        df["deal_size_vs_avg"] = df["deal_amount"] / df["industry_avg_amount"]

        # 3. Rep Consistency Score
        df["rep_consistency_score"] = 1 - (df["rep_win_std"] / df["rep_win_rate"].clip(lower=0.01))
        df["rep_consistency_score"] = df["rep_consistency_score"].clip(0, 1)

        # 4. Cycle Efficiency
        df["cycle_efficiency"] = df["industry_avg_cycle"] / df["sales_cycle_days"].clip(lower=1)
        df["cycle_efficiency"] = df["cycle_efficiency"].clip(0, 3)

        # 5. Deal Amount Tier
        df["deal_tier"] = pd.qcut(
            df["deal_amount"], q=4, labels=["Small", "Medium", "Large", "Enterprise"]
        )

        # 6. Is High Value Deal
        df["is_high_value"] = (df["deal_amount"] > df["deal_amount"].quantile(0.75)).astype(int)

        # 7. Rep Experience Score (based on deal count)
        df["rep_experience"] = pd.cut(
            df["rep_deal_count"],
            bins=[0, 50, 150, 300, float("inf")],
            labels=["Junior", "Mid", "Senior", "Expert"]
        )

        # Fill any NaN values (skip categorical columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        logger.info(f"Added {len(self.get_engineered_features())} engineered features")
        return df

    def _calculate_deal_health_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Deal Health Score = weighted combination of:
        - Rep win rate contribution (40%)
        - Industry win rate contribution (30%)
        - Cycle efficiency contribution (30%)
        """
        rep_component = df["rep_win_rate"] * 0.4
        industry_component = df["industry_win_rate"] * 0.3

        # Cycle efficiency: faster than average is better
        cycle_ratio = df["industry_avg_cycle"] / df["sales_cycle_days"].clip(lower=1)
        cycle_component = cycle_ratio.clip(0, 2) / 2 * 0.3

        health_score = (rep_component + industry_component + cycle_component).clip(0, 1)
        return health_score.round(4)

    def get_engineered_features(self) -> list:
        """Get list of engineered feature names."""
        return [
            "rep_win_rate", "rep_deal_count", "rep_win_std",
            "rep_avg_deal_amount", "rep_avg_cycle",
            "industry_win_rate", "industry_avg_amount", "industry_avg_cycle",
            "lead_source_win_rate", "lead_source_avg_amount",
            "deal_health_score", "deal_size_vs_avg",
            "rep_consistency_score", "cycle_efficiency", "is_high_value"
        ]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)


def calculate_pipeline_metrics(df: pd.DataFrame) -> Dict:
    """Calculate pipeline-level metrics for dashboard."""

    # Time-based cohort analysis
    df["cohort"] = df["created_date"].dt.to_period("Q")

    cohort_metrics = df.groupby("cohort").agg({
        "deal_id": "count",
        "target": "mean",
        "deal_amount": ["sum", "mean"],
        "sales_cycle_days": "mean"
    }).round(4)

    cohort_metrics.columns = [
        "deal_count", "win_rate", "total_revenue",
        "avg_deal_size", "avg_cycle_days"
    ]

    # Calculate Pipeline Momentum Index
    # PMI = (new_deals - lost_deals) / total_pipeline
    won_deals = df[df["target"] == 1].groupby("cohort").size()
    lost_deals = df[df["target"] == 0].groupby("cohort").size()
    total_deals = df.groupby("cohort").size()

    pmi = ((won_deals - lost_deals) / total_deals).round(4)
    cohort_metrics["pipeline_momentum_index"] = pmi

    # Sales Velocity = (Deals × Win Rate × ACV) / Sales Cycle
    cohort_metrics["sales_velocity"] = (
        cohort_metrics["deal_count"] *
        cohort_metrics["win_rate"] *
        cohort_metrics["avg_deal_size"] /
        cohort_metrics["avg_cycle_days"].clip(lower=1)
    ).round(2)

    return cohort_metrics.to_dict()
