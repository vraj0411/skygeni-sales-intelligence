"""SHAP-based explainability for deal risk predictions."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Install with: pip install shap")


class SHAPExplainer:
    """
    SHAP-based model explainability.

    Provides:
    - Global feature importance
    - Local explanations for individual deals
    - Interaction effects
    """

    def __init__(self, model, feature_columns: List[str]):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")

        self.model = model
        self.feature_columns = feature_columns
        self.explainer = None
        self.shap_values = None
        self.expected_value = None

    def fit(self, X: pd.DataFrame, sample_size: int = 500) -> "SHAPExplainer":
        """
        Create SHAP explainer and calculate values.

        Args:
            X: Feature dataframe
            sample_size: Number of samples for background data
        """
        X_features = X[self.feature_columns]

        # Sample background data for efficiency
        if len(X_features) > sample_size:
            background = X_features.sample(sample_size, random_state=42)
        else:
            background = X_features

        # Create explainer based on model type
        model_type = type(self.model).__name__

        if "XGB" in model_type or "LGBM" in model_type:
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                background
            )

        # Calculate SHAP values
        self.shap_values = self.explainer.shap_values(X_features)
        self.expected_value = self.explainer.expected_value

        logger.info("SHAP explainer fitted")
        return self

    def get_global_importance(self) -> pd.DataFrame:
        """Get global feature importance from SHAP values."""
        if self.shap_values is None:
            raise ValueError("Call fit() first")

        # Handle binary classification (use positive class)
        values = self.shap_values
        if isinstance(values, list):
            values = values[1]  # Positive class

        importance = np.abs(values).mean(axis=0)

        return pd.DataFrame({
            "feature": self.feature_columns,
            "shap_importance": importance
        }).sort_values("shap_importance", ascending=False)

    def explain_deal(
        self,
        deal: pd.Series,
        X_background: pd.DataFrame,
        top_n: int = 5
    ) -> Dict:
        """
        Explain prediction for a single deal.

        Returns top factors contributing to risk score.
        """
        deal_features = deal[self.feature_columns].values.reshape(1, -1)

        # Get SHAP values for this deal
        shap_vals = self.explainer.shap_values(deal_features)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # Positive class

        shap_vals = shap_vals[0]

        # Create explanation
        explanations = []
        sorted_idx = np.argsort(np.abs(shap_vals))[::-1]

        for idx in sorted_idx[:top_n]:
            feature = self.feature_columns[idx]
            value = deal[feature]
            shap_contribution = shap_vals[idx]

            direction = "increases" if shap_contribution > 0 else "decreases"

            explanations.append({
                "feature": feature,
                "value": value,
                "shap_value": shap_contribution,
                "direction": direction,
                "impact": "win probability" if shap_contribution > 0 else "loss risk"
            })

        return {
            "deal_id": deal.get("deal_id", "unknown"),
            "top_factors": explanations,
            "base_probability": float(self.expected_value[1]) if isinstance(
                self.expected_value, list
            ) else float(self.expected_value)
        }

    def generate_natural_language_explanation(
        self,
        deal: pd.Series,
        X_background: pd.DataFrame
    ) -> str:
        """
        Generate human-readable explanation for sales team.

        Example output:
        "This deal has a 72% risk of loss primarily because:
        1. The sales rep has a below-average win rate (32% vs 48% company avg)
        2. The deal is taking longer than typical for this industry (45 days vs 30 avg)
        3. The lead source (Outbound) historically has lower conversion"
        """
        explanation = self.explain_deal(deal, X_background)

        risk_score = 1 - deal.get("win_probability", 0.5)
        risk_pct = int(risk_score * 100)

        lines = [f"This deal has a {risk_pct}% risk of loss primarily because:"]

        for i, factor in enumerate(explanation["top_factors"][:3], 1):
            feature = factor["feature"].replace("_", " ").title()
            value = factor["value"]
            direction = "contributing to risk" if factor["shap_value"] < 0 else "helping win rate"

            lines.append(f"{i}. {feature}: {value} ({direction})")

        return "\n".join(lines)

    def get_interaction_effects(self, top_n: int = 5) -> pd.DataFrame:
        """Get top feature interactions."""
        if self.shap_values is None:
            raise ValueError("Call fit() first")

        # This requires shap_interaction_values which is expensive
        # For now, return feature correlations with SHAP values
        values = self.shap_values
        if isinstance(values, list):
            values = values[1]

        # Simple correlation-based interaction proxy
        interactions = []
        n_features = len(self.feature_columns)

        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr = np.corrcoef(values[:, i], values[:, j])[0, 1]
                if not np.isnan(corr):
                    interactions.append({
                        "feature_1": self.feature_columns[i],
                        "feature_2": self.feature_columns[j],
                        "interaction_strength": abs(corr)
                    })

        interactions_df = pd.DataFrame(interactions)
        return interactions_df.sort_values(
            "interaction_strength", ascending=False
        ).head(top_n)
