"""Evaluation metrics for SkyGeni Sales Intelligence."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    brier_score_loss, log_loss
)
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for deal risk scoring."""

    def __init__(self, y_true: np.ndarray, y_prob: np.ndarray):
        self.y_true = y_true
        self.y_prob = y_prob

    def get_all_metrics(self) -> Dict:
        """Calculate all evaluation metrics."""
        return {
            "roc_metrics": self._roc_metrics(),
            "pr_metrics": self._precision_recall_metrics(),
            "calibration_metrics": self._calibration_metrics(),
            "threshold_analysis": self._threshold_analysis()
        }

    def _roc_metrics(self) -> Dict:
        """Calculate ROC curve metrics."""
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_prob)
        roc_auc = auc(fpr, tpr)

        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]

        return {
            "auc": roc_auc,
            "optimal_threshold": optimal_threshold,
            "fpr_at_optimal": fpr[optimal_idx],
            "tpr_at_optimal": tpr[optimal_idx]
        }

    def _precision_recall_metrics(self) -> Dict:
        """Calculate precision-recall metrics."""
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_prob)
        pr_auc = auc(recall, precision)

        # F1 scores at each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores[:-1])  # Last element is undefined

        return {
            "pr_auc": pr_auc,
            "optimal_threshold": thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.5,
            "precision_at_optimal": precision[optimal_idx],
            "recall_at_optimal": recall[optimal_idx],
            "f1_at_optimal": f1_scores[optimal_idx]
        }

    def _calibration_metrics(self) -> Dict:
        """Assess probability calibration."""
        brier = brier_score_loss(self.y_true, self.y_prob)
        logloss = log_loss(self.y_true, self.y_prob)

        # Binned calibration
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(self.y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, 9)

        calibration_data = []
        for i in range(10):
            mask = bin_indices == i
            if mask.sum() > 0:
                calibration_data.append({
                    "bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                    "predicted_prob": self.y_prob[mask].mean(),
                    "actual_rate": self.y_true[mask].mean(),
                    "count": mask.sum()
                })

        return {
            "brier_score": brier,
            "log_loss": logloss,
            "calibration_by_bin": calibration_data
        }

    def _threshold_analysis(self) -> List[Dict]:
        """Analyze performance at different thresholds."""
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        results = []

        for thresh in thresholds:
            y_pred = (self.y_prob >= thresh).astype(int)
            tp = ((y_pred == 1) & (self.y_true == 1)).sum()
            fp = ((y_pred == 1) & (self.y_true == 0)).sum()
            tn = ((y_pred == 0) & (self.y_true == 0)).sum()
            fn = ((y_pred == 0) & (self.y_true == 1)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results.append({
                "threshold": thresh,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn)
            })

        return results


def calculate_business_metrics(
    df: pd.DataFrame,
    risk_threshold: float = 0.6
) -> Dict:
    """
    Calculate business-relevant metrics from scored deals.

    These metrics matter to the CRO:
    - Deals at risk value
    - Potential revenue loss
    - Win rate by segment
    """
    high_risk = df[df["risk_score"] >= risk_threshold]
    low_risk = df[df["risk_score"] < risk_threshold]

    metrics = {
        "total_deals": len(df),
        "high_risk_count": len(high_risk),
        "high_risk_percentage": len(high_risk) / len(df) * 100,
        "high_risk_total_value": high_risk["deal_amount"].sum(),
        "potential_loss_if_all_high_risk_lost": high_risk["deal_amount"].sum(),
        "expected_loss": (high_risk["deal_amount"] * high_risk["risk_score"]).sum(),
        "win_rate_high_risk": high_risk["target"].mean() if len(high_risk) > 0 else 0,
        "win_rate_low_risk": low_risk["target"].mean() if len(low_risk) > 0 else 0,
        "avg_deal_amount_high_risk": high_risk["deal_amount"].mean() if len(high_risk) > 0 else 0,
        "avg_deal_amount_low_risk": low_risk["deal_amount"].mean() if len(low_risk) > 0 else 0
    }

    return metrics


def segment_analysis(df: pd.DataFrame) -> Dict:
    """Analyze risk by different segments."""
    segments = {}

    # By region
    segments["by_region"] = df.groupby("region").agg({
        "risk_score": "mean",
        "target": "mean",
        "deal_amount": ["sum", "count"]
    }).round(4).to_dict()

    # By industry
    segments["by_industry"] = df.groupby("industry").agg({
        "risk_score": "mean",
        "target": "mean",
        "deal_amount": ["sum", "count"]
    }).round(4).to_dict()

    # By lead source
    segments["by_lead_source"] = df.groupby("lead_source").agg({
        "risk_score": "mean",
        "target": "mean",
        "deal_amount": ["sum", "count"]
    }).round(4).to_dict()

    # By sales rep (top 10 by deal count)
    rep_analysis = df.groupby("sales_rep_id").agg({
        "risk_score": "mean",
        "target": "mean",
        "deal_amount": ["sum", "count"]
    }).round(4)
    rep_analysis.columns = ["avg_risk", "win_rate", "total_value", "deal_count"]
    rep_analysis = rep_analysis.sort_values("deal_count", ascending=False).head(10)
    segments["top_reps"] = rep_analysis.to_dict()

    return segments
