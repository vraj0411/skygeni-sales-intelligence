# SkyGeni Sales Intelligence System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This project addresses the SkyGeni Data Science Challenge: **helping a CRO understand why win rates are dropping and what actions to take.**

### The Problem
> "Our win rate has dropped over the last two quarters, but pipeline volume looks healthy. I don't know what exactly is going wrong or what my team should focus on."

### The Solution
A comprehensive **Sales Intelligence System** that:
1. **Diagnoses** win rate issues through data analysis
2. **Predicts** deal risk using ML models
3. **Recommends** actions through explainable insights
4. **Monitors** pipeline health via real-time dashboard

---

## Project Structure

```
skygeni/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Scored deals, metrics
├── notebooks/
│   └── 01_eda.py              # Exploratory Data Analysis
├── src/
│   ├── data/
│   │   ├── loader.py          # Data loading utilities
│   │   └── preprocessor.py    # Data preprocessing
│   ├── features/
│   │   └── engineering.py     # Feature engineering
│   ├── models/
│   │   └── risk_scorer.py     # Deal Risk Scoring model
│   ├── evaluation/
│   │   ├── metrics.py         # Model evaluation
│   │   └── explainability.py  # SHAP explanations
│   └── api/
│       └── main.py            # FastAPI endpoints
├── models/                     # Saved model artifacts
├── config/
│   └── config.yaml            # Configuration
├── tests/                      # Unit tests
├── app.py                      # Streamlit dashboard
├── train.py                    # Training pipeline
├── requirements.txt
└── README.md
```

---

## Quick Start

### Installation

```bash
# Clone or navigate to project
cd skygeni

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# 1. Run EDA (generates insights and figures)
python notebooks/01_eda.py

# 2. Train the model
python train.py

# 3. Launch dashboard
streamlit run app.py
```

---

## Part 1: Problem Framing

### Real Business Problem
The CRO's concern about dropping win rates likely stems from multiple root causes:

**Root Cause Hypothesis Framework:**
```
Win Rate Drop Investigation
├── Volume Problem? → Pipeline volume is healthy (ruled out by CRO)
├── Velocity Problem? → Are deals taking longer to close?
├── Conversion Problem? → Which stage has the biggest drop-off?
├── Quality Problem? → Are we pursuing wrong-fit deals?
└── Execution Problem? → Rep performance variance?
```

### Key Questions for the AI System
1. **Diagnostic:** Which segments (region/industry/rep) are underperforming?
2. **Predictive:** Which open deals are at highest risk of loss?
3. **Prescriptive:** What actions can improve outcomes?

### Metrics That Matter
- **Win Rate by Cohort** - Time-based trend analysis
- **Stage Conversion Rates** - Funnel drop-off points
- **Sales Velocity** = (Deals × Win Rate × ACV) / Cycle Days
- **Deal Health Score** - Custom composite metric
- **Rep Consistency Score** - Performance reliability

---

## Part 2: Data Exploration & Insights

### Key Insights Discovered

**Insight 1: Regional Performance Gap**
- Significant variance in win rates across regions
- Action: Investigate top-performing region's practices

**Insight 2: Win Rate Trend Confirmation**
- Data confirms CRO's concern about recent quarters
- Recent 2Q vs Historical shows measurable decline

**Insight 3: Sales Cycle Impact**
- Deals >60 days have significantly lower win rates
- Action: Implement deal velocity alerts

### Custom Metrics

**1. Deal Health Score**
```
Health Score = (Rep Win Rate × 0.4) +
               (Cycle Efficiency × 0.3) +
               (Stage Progress × 0.3)
```
- Correlation with outcomes: ~0.45
- Predictive of deal success

**2. Rep Consistency Score**
```
Consistency = 1 - (Win Rate Std Dev / Mean Win Rate)
```
- Identifies reliable vs volatile performers
- Useful for forecasting accuracy

---

## Part 3: Decision Engine - Deal Risk Scoring

### Approach: XGBoost with SHAP Explainability

**Why XGBoost?**
- Handles mixed feature types well
- Robust to outliers
- Feature importance built-in
- Fast inference for production

**Model Architecture:**
```
Input Features (20+)
├── Encoded categoricals (industry, region, rep, etc.)
├── Numerical (deal_amount, sales_cycle_days)
├── Engineered (deal_health_score, rep_win_rate, etc.)
└── Time features (quarter, month, days_since_start)
        ↓
    XGBoost Classifier
        ↓
    Probability Calibration (Isotonic)
        ↓
    Risk Score (0-1) + Category (High/Medium/Low)
```

**Performance:**
- AUC: ~0.75-0.80
- Cross-validated for robustness
- Calibrated probabilities for business decisions

### How Sales Leaders Use This

1. **Daily:** Review high-risk deal alerts
2. **Weekly:** Pipeline risk distribution report
3. **Monthly:** Segment performance analysis

**Example Alert:**
> "Deal #D00123 ($50K, FinTech, Rep_12) has 78% risk of loss.
> Top factors: Long sales cycle (85 days), below-average rep performance."

---

## Part 4: System Design

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 SALES INSIGHT SYSTEM                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [CRM Data]──→[Ingestion]──→[Feature Store]                 │
│       │            │              │                          │
│       └────────────┴──────────────┘                          │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           ML Pipeline (Scheduled)                     │   │
│  │  [EDA] → [Features] → [Train] → [Score] → [Store]    │   │
│  └──────────────────────────────────────────────────────┘   │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 Alert Engine                          │   │
│  │  • High risk deals → Slack/Email to rep              │   │
│  │  • Win rate drop → Alert to CRO                      │   │
│  │  • Anomalies → Dashboard flag                        │   │
│  └──────────────────────────────────────────────────────┘   │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Dashboard (Streamlit)                    │   │
│  │  • Risk overview    • Pipeline analytics             │   │
│  │  • Rep performance  • Deal explorer                  │   │
│  │  • Model insights   • Export reports                 │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Data Ingestion | Pandas/Polars | Load and validate CRM exports |
| Feature Store | Local CSV (prod: Redis/Feast) | Serve features for scoring |
| ML Pipeline | XGBoost + SHAP | Train and score deals |
| Alert Engine | Python + Webhooks | Send notifications |
| Dashboard | Streamlit | Interactive visualization |
| API | FastAPI | Programmatic access |

### Schedule
- **Every 6 hours:** Re-score all open deals
- **Daily:** Retrain model with new closed deals
- **Weekly:** Full EDA report generation

### Failure Cases & Limitations
- **Cold start:** New reps/industries have limited history
- **Data quality:** Missing or incorrect CRM data
- **Model drift:** Market conditions change over time
- **Latency:** Batch scoring, not real-time

---

## Part 5: Reflection

### Weakest Assumptions
1. **Historical patterns persist** - Market conditions may shift
2. **Data is representative** - Sample may not reflect all scenarios
3. **Features capture reality** - Important signals may be missing (e.g., email sentiment)

### Production Challenges
1. **CRM Integration** - Real-time data sync is complex
2. **User Adoption** - Sales teams may distrust/ignore model
3. **Feedback Loop** - Need mechanism to learn from rep actions
4. **Explainability** - SHAP helps but isn't always intuitive

### Given 1 Month, I Would Build:
1. **LLM-powered insights** - Natural language explanations
2. **Survival analysis** - Time-to-close predictions
3. **Causal inference** - True driver identification (not just correlation)
4. **A/B testing framework** - Measure recommendation impact
5. **Slack bot** - Proactive deal coaching

### Least Confident About:
- Calibration of probabilities in tail scenarios
- Generalization to very different deal profiles
- Long-term model stability without retraining

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python 3.11+ |
| Data | Pandas, Polars, NumPy |
| ML | XGBoost, LightGBM, Scikit-learn |
| Explainability | SHAP |
| Visualization | Plotly, Seaborn, Matplotlib |
| Dashboard | Streamlit |
| API | FastAPI |
| MLOps | MLflow (experiment tracking) |
| Testing | Pytest |

---

## Results Summary

| Metric | Value |
|--------|-------|
| Model AUC | ~0.75-0.80 |
| Custom Metrics | 2 (Deal Health, Rep Consistency) |
| Business Insights | 3+ (Segment, Time, Cycle analysis) |
| Dashboard Views | 5 tabs |
| Code Coverage | Basic tests included |

---

## Demo

To see the system in action:

```bash
# Terminal 1: Run training
python train.py

# Terminal 2: Launch dashboard
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## License

MIT License - feel free to use and modify.

---

## Author

SkyGeni Data Science Challenge
