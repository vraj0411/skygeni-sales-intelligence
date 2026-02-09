"""LLM-based insights generation using Groq Cloud."""

import os
from typing import Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq not installed. Install with: pip install groq")


class InsightsGenerator:
    """Generate natural language insights using Groq LLM."""

    def __init__(self, api_key: Optional[str] = None):
        if not GROQ_AVAILABLE:
            raise ImportError("Groq package not installed. Run: pip install groq")

        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key required")

        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"

    def _call_llm(self, prompt: str, max_tokens: int = 512) -> str:
        """Call Groq LLM API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a sales analytics expert. Give answers in bullet points only. Be specific with numbers. Focus on WHY things happen, not descriptions. No introductions or conclusions. Just the points."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Error: {str(e)}"

    def analyze_risk_distribution(self, risk_data: Dict) -> str:
        """Analyze why deals are at risk."""
        prompt = f"""
Data:
- High Risk: {risk_data.get('high_risk', 0)} deals (${risk_data.get('high_risk_value', 0):,.0f})
- Medium Risk: {risk_data.get('medium_risk', 0)} deals
- Low Risk: {risk_data.get('low_risk', 0)} deals
- Avg Risk Score: {risk_data.get('avg_risk', 0):.1%}

In 3-4 bullet points, explain:
- Why are deals falling into high risk category?
- What factors are causing this risk distribution?
"""
        return self._call_llm(prompt)

    def analyze_region_performance(self, region_data: pd.DataFrame) -> str:
        """Explain why regions perform differently."""
        data_str = region_data.to_string()
        prompt = f"""
Region Performance Data:
{data_str}

In 3-4 bullet points explain:
- Which region is underperforming and WHY?
- What specific factors cause the performance gap?
- What action fixes this?
"""
        return self._call_llm(prompt)

    def analyze_industry_performance(self, industry_data: pd.DataFrame) -> str:
        """Explain industry performance differences."""
        data_str = industry_data.to_string()
        prompt = f"""
Industry Performance Data:
{data_str}

In 3-4 bullet points explain:
- Which industry has lowest win rate and WHY?
- What causes performance differences between industries?
"""
        return self._call_llm(prompt)

    def analyze_pipeline_trend(self, trend_data: pd.DataFrame) -> str:
        """Explain pipeline trends over time."""
        data_str = trend_data.to_string()
        prompt = f"""
Quarterly Pipeline Data:
{data_str}

In 3-4 bullet points explain:
- Is win rate improving or declining? By how much?
- WHY is this trend happening?
- Which quarter had the biggest change and why?
"""
        return self._call_llm(prompt)

    def analyze_lead_source(self, source_data: pd.DataFrame) -> str:
        """Explain lead source performance."""
        data_str = source_data.to_string()
        prompt = f"""
Lead Source Performance:
{data_str}

In 3-4 bullet points explain:
- Which lead source converts best/worst and WHY?
- What makes certain sources more effective?
"""
        return self._call_llm(prompt)

    def analyze_rep_performance(self, rep_data: pd.DataFrame) -> str:
        """Explain sales rep performance patterns."""
        data_str = rep_data.head(10).to_string()
        prompt = f"""
Top Sales Rep Data:
{data_str}

In 3-4 bullet points explain:
- What separates top performers from others?
- WHY do some reps have higher risk scores?
- What patterns indicate rep needs coaching?
"""
        return self._call_llm(prompt)

    def explain_deal_risk(self, deal: pd.Series) -> str:
        """Explain why a specific deal is at risk."""
        prompt = f"""
Deal Details:
- Amount: ${deal.get('deal_amount', 0):,.0f}
- Industry: {deal.get('industry', 'Unknown')}
- Region: {deal.get('region', 'Unknown')}
- Rep: {deal.get('sales_rep_id', 'Unknown')}
- Risk Score: {deal.get('risk_score', 0):.0%}
- Sales Cycle: {deal.get('sales_cycle_days', 0)} days
- Stage: {deal.get('deal_stage', 'Unknown')}
- Lead Source: {deal.get('lead_source', 'Unknown')}
- Rep Win Rate: {deal.get('rep_win_rate', 0):.0%}

In 3 bullet points:
- WHY is this deal at {deal.get('risk_score', 0):.0%} risk?
- What specific factors are causing this?
- One action to save this deal
"""
        return self._call_llm(prompt, max_tokens=256)

    def analyze_feature_importance(self, features: pd.DataFrame) -> str:
        """Explain what features drive predictions."""
        data_str = features.head(10).to_string()
        prompt = f"""
Top Predictive Features:
{data_str}

In 3-4 bullet points explain:
- WHY are these features most important for predicting deal outcomes?
- What does each top feature tell us about deal success/failure?
"""
        return self._call_llm(prompt)

    def generate_action_items(self, metrics: Dict) -> str:
        """Generate specific action items."""
        prompt = f"""
Current State:
- Win Rate: {metrics.get('win_rate', 0):.0%}
- High Risk Deals: {metrics.get('high_risk_count', 0)}
- At-Risk Value: ${metrics.get('high_risk_value', 0):,.0f}
- Avg Cycle: {metrics.get('avg_cycle', 0):.0f} days

Give exactly 4 specific actions as bullet points:
- What to do THIS WEEK to improve win rate?
- Which deals need immediate attention?
- What process change helps most?
- What metric to track daily?
"""
        return self._call_llm(prompt, max_tokens=300)

    def analyze_forecast(self, forecast_data: Dict) -> str:
        """Analyze revenue forecast."""
        prompt = f"""
Forecast Data:
- Total Pipeline: ${forecast_data.get('total_pipeline', 0):,.0f}
- Weighted Pipeline: ${forecast_data.get('weighted_pipeline', 0):,.0f}
- Average Win Rate: {forecast_data.get('avg_win_rate', 0):.0%}
- Revenue Trend: {forecast_data.get('trend', 0):+.1%}
- High Risk Value: ${forecast_data.get('high_risk_value', 0):,.0f}

In 4 bullet points:
- What is realistic revenue expectation?
- What risks could reduce forecast?
- What actions improve forecast?
- Which segment has highest potential?
"""
        return self._call_llm(prompt, max_tokens=350)

    def explain_alert(self, alert_type: str, data: Dict) -> str:
        """Explain why an alert was triggered."""
        if alert_type == "high_risk_deal":
            prompt = f"""
Alert: High Risk Deal
- Deal Amount: ${data.get('amount', 0):,.0f}
- Risk Score: {data.get('risk_score', 0):.0%}
- Days in Pipeline: {data.get('cycle_days', 0)}
- Rep Win Rate: {data.get('rep_win_rate', 0):.0%}

In 2 bullet points:
- Why is this deal flagged?
- What action to take now?
"""
        elif alert_type == "win_rate_drop":
            prompt = f"""
Alert: Win Rate Dropped
- Previous: {data.get('prev_rate', 0):.0%}
- Current: {data.get('current_rate', 0):.0%}
- Change: {data.get('change', 0):+.1%}

In 2 bullet points:
- Why did win rate drop?
- What to investigate first?
"""
        else:
            prompt = f"Explain this alert: {data}"

        return self._call_llm(prompt, max_tokens=200)


def get_insights_generator() -> Optional[InsightsGenerator]:
    """Factory function to create InsightsGenerator with API key."""
    api_key = "GROQ_API_KEY"
    try:
        return InsightsGenerator(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize InsightsGenerator: {e}")
        return None
