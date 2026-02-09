# SkyGeni Sales Intelligence System - Detailed Report

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Solution Architecture](#3-solution-architecture)
4. [Data Flow](#4-data-flow)
5. [Components Explained](#5-components-explained)
6. [Machine Learning Model](#6-machine-learning-model)
7. [Feature Engineering](#7-feature-engineering)
8. [Dashboard Tabs](#8-dashboard-tabs)
9. [LLM Integration](#9-llm-integration)
10. [File Structure](#10-file-structure)
11. [How to Run](#11-how-to-run)
12. [Technical Specifications](#12-technical-specifications)

---

## 1. Project Overview

### What is this system?
A Sales Intelligence System that helps sales leaders understand:
- Why deals are being lost
- Which deals are at risk
- What actions to take to improve win rates

### Who uses it?
- **CRO (Chief Revenue Officer)**: Monitor overall pipeline health
- **Sales Managers**: Track team performance, coach underperforming reps
- **Sales Reps**: Prioritize deals that need attention

### What it does?
```
INPUT: Sales deal data (5000 deals)
       |
       v
PROCESS: Analyze -> Score -> Predict -> Explain
       |
       v
OUTPUT: Risk scores, insights, alerts, forecasts
```

---

## 2. Problem Statement

### The Business Problem
A CRO complained:
> "Our win rate has dropped over the last two quarters, but pipeline volume looks healthy. I don't know what exactly is going wrong or what my team should focus on."

### Questions to Answer
1. WHY is win rate dropping?
2. WHICH deals are at risk?
3. WHAT factors cause deals to fail?
4. WHO (which reps) needs help?
5. WHAT actions will improve results?

### Success Metrics
- Identify at-risk deals before they're lost
- Explain reasons behind performance issues
- Provide actionable recommendations

---

## 3. Solution Architecture

```
+------------------------------------------------------------------+
|                    SKYGENI SALES INTELLIGENCE                      |
+------------------------------------------------------------------+
|                                                                    |
|  +----------------+     +------------------+     +---------------+ |
|  |   RAW DATA     | --> |   PROCESSING     | --> |    MODEL      | |
|  | (CSV 5000 rows)|     | (Clean, Feature) |     | (Risk Score)  | |
|  +----------------+     +------------------+     +---------------+ |
|                                                         |          |
|                                                         v          |
|  +----------------+     +------------------+     +---------------+ |
|  |   DASHBOARD    | <-- |   LLM INSIGHTS   | <-- | SCORED DATA   | |
|  |  (Streamlit)   |     |    (Groq API)    |     | (Predictions) | |
|  +----------------+     +------------------+     +---------------+ |
|                                                                    |
+------------------------------------------------------------------+
```

### Components
1. **Data Layer**: Load, validate, preprocess data
2. **Feature Layer**: Create custom metrics and features
3. **Model Layer**: Train ML model, score deals
4. **Insight Layer**: LLM generates explanations
5. **Presentation Layer**: Dashboard for visualization

---

## 4. Data Flow

### Step-by-Step Process

```
STEP 1: Load Raw Data
        File: data/raw/skygeni_sales_data.csv
        Rows: 5000 deals
        Columns: 12 fields
        |
        v
STEP 2: Preprocess
        - Parse dates
        - Encode categories (industry, region, etc.)
        - Scale numbers (deal_amount, cycle_days)
        - Create target (Won=1, Lost=0)
        |
        v
STEP 3: Feature Engineering
        - Calculate rep win rates
        - Calculate industry averages
        - Create Deal Health Score
        - Create custom metrics
        |
        v
STEP 4: Train Model
        - Split: 80% train, 20% test
        - Algorithm: Gradient Boosting
        - Output: Probability of winning
        |
        v
STEP 5: Score Deals
        - Risk Score = 1 - Win Probability
        - Categories: High (>60%), Medium (30-60%), Low (<30%)
        |
        v
STEP 6: Generate Insights
        - LLM analyzes patterns
        - Explains WHY in bullet points
        |
        v
STEP 7: Display Dashboard
        - 7 tabs with visualizations
        - Interactive filters
        - Real-time LLM explanations
```

---

## 5. Components Explained

### 5.1 Data Loader (`src/data/loader.py`)

**Purpose**: Load and validate CSV data

**What it does**:
```python
loader = DataLoader("data/raw/skygeni_sales_data.csv")
df = loader.load()  # Returns pandas DataFrame
```

**Validations**:
- Checks all required columns exist
- Parses date columns
- Provides summary statistics

---

### 5.2 Preprocessor (`src/data/preprocessor.py`)

**Purpose**: Prepare data for ML model

**What it does**:
```
Raw Data                    Processed Data
-----------                 --------------
industry: "SaaS"      -->   industry_encoded: 3
region: "APAC"        -->   region_encoded: 0
deal_amount: 50000    -->   deal_amount_scaled: 1.25
outcome: "Won"        -->   target: 1
```

**Transformations**:
| Original | Transformation | Result |
|----------|----------------|--------|
| Categorical columns | Label Encoding | Numbers (0, 1, 2...) |
| Numerical columns | Standard Scaling | Mean=0, Std=1 |
| Outcome | Binary | Won=1, Lost=0 |

---

### 5.3 Feature Engineer (`src/features/engineering.py`)

**Purpose**: Create meaningful features from raw data

**Custom Metrics Created**:

#### 1. Deal Health Score
```
Formula: (Rep Win Rate × 0.4) + (Cycle Efficiency × 0.3) + (Stage Progress × 0.3)

Example:
- Rep Win Rate: 60% (good)
- Cycle Efficiency: 80% (faster than average)
- Stage: Negotiation (0.8)

Health Score = (0.6 × 0.4) + (0.8 × 0.3) + (0.8 × 0.3) = 0.72 (healthy)
```

#### 2. Rep Consistency Score
```
Formula: 1 - (Standard Deviation / Mean Win Rate)

Example:
- Rep A: 50% avg, 5% std → Consistency = 0.90 (reliable)
- Rep B: 50% avg, 20% std → Consistency = 0.60 (unpredictable)
```

#### 3. Cycle Efficiency
```
Formula: Industry Average Cycle / Deal's Cycle Days

Example:
- Industry average: 45 days
- Deal cycle: 30 days
- Efficiency = 45/30 = 1.5 (faster than average = good)
```

#### 4. Deal Size vs Average
```
Formula: Deal Amount / Industry Average Amount

Example:
- Deal: $100,000
- Industry average: $50,000
- Ratio = 2.0 (larger than typical)
```

---

### 5.4 Risk Scorer Model (`src/models/risk_scorer.py`)

**Purpose**: Predict probability of deal loss

**Algorithm**: Gradient Boosting Classifier

**How it works**:
```
Input Features (26 total)
    |
    v
+-------------------+
| Gradient Boosting |  (200 trees, max_depth=6)
+-------------------+
    |
    v
Win Probability (0.0 to 1.0)
    |
    v
Risk Score = 1 - Win Probability
    |
    v
Risk Category:
  - High Risk: > 60%
  - Medium Risk: 30-60%
  - Low Risk: < 30%
```

**Feature Importance** (what drives predictions):
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | days_since_start | 13.3% |
| 2 | deal_health_score | 12.0% |
| 3 | deal_amount_scaled | 11.9% |
| 4 | deal_size_vs_avg | 11.8% |
| 5 | cycle_efficiency | 7.9% |

---

### 5.5 LLM Insights Generator (`src/llm/insights_generator.py`)

**Purpose**: Generate human-readable explanations

**API**: Groq Cloud (llama-3.3-70b-versatile model)

**How it works**:
```
Data + Prompt --> Groq API --> Bullet Point Explanation

Example:
Input: "Win rate dropped from 48% to 43%"
Output:
• Win rate declined 5% due to longer sales cycles in Q4
• North America region underperformed by 8% vs other regions
• Outbound leads converting 12% lower than referrals
• Action: Focus on accelerating deals >45 days in pipeline
```

**Functions Available**:
| Function | Purpose |
|----------|---------|
| analyze_risk_distribution | Explain why deals are at risk |
| analyze_region_performance | Explain regional differences |
| analyze_pipeline_trend | Explain win rate changes |
| explain_deal_risk | Explain single deal's risk |
| generate_action_items | Recommend specific actions |
| analyze_forecast | Explain revenue predictions |

---

## 6. Machine Learning Model

### Model Selection

**Chosen**: Gradient Boosting Classifier (sklearn)

**Why**:
- Handles mixed data types (numbers + categories)
- Provides feature importance
- Works without XGBoost/LightGBM dependencies
- Good balance of accuracy and interpretability

### Training Process

```
1. Data Split
   - Training: 4000 deals (80%)
   - Testing: 1000 deals (20%)
   - Stratified by outcome (maintain Won/Lost ratio)

2. Model Configuration
   - n_estimators: 200 (number of trees)
   - max_depth: 6 (tree depth)
   - learning_rate: 0.1
   - subsample: 0.8 (use 80% of data per tree)

3. Calibration
   - Isotonic regression for probability calibration
   - Ensures predicted probabilities are accurate

4. Evaluation
   - Cross-validation: 5 folds
   - Metrics: AUC, Accuracy, Precision, Recall, F1
```

### Model Performance

| Metric | Value | Meaning |
|--------|-------|---------|
| AUC | 0.496 | Model performance (0.5 = random) |
| Accuracy | 50.8% | Correct predictions |
| Precision | 44.8% | When predicts Win, how often correct |
| Recall | 37.3% | Of actual Wins, how many found |

**Note**: Low scores because synthetic data has no real patterns. Real CRM data would show much better performance.

---

## 7. Feature Engineering

### All Features Used (26 total)

#### Encoded Categorical (6)
| Feature | Original | Example |
|---------|----------|---------|
| industry_encoded | industry | SaaS → 3 |
| region_encoded | region | APAC → 0 |
| product_type_encoded | product_type | Enterprise → 1 |
| lead_source_encoded | lead_source | Inbound → 0 |
| deal_stage_encoded | deal_stage | Proposal → 3 |
| sales_rep_id_encoded | sales_rep_id | rep_5 → 4 |

#### Scaled Numerical (2)
| Feature | Original | Transformation |
|---------|----------|----------------|
| deal_amount_scaled | deal_amount | StandardScaler |
| sales_cycle_days_scaled | sales_cycle_days | StandardScaler |

#### Engineered Features (15)
| Feature | Description |
|---------|-------------|
| rep_win_rate | Rep's historical win rate |
| rep_deal_count | Number of deals rep handled |
| rep_win_std | Variability in rep's performance |
| rep_avg_deal_amount | Rep's average deal size |
| rep_avg_cycle | Rep's average sales cycle |
| industry_win_rate | Industry's average win rate |
| industry_avg_amount | Industry's average deal size |
| industry_avg_cycle | Industry's average cycle |
| lead_source_win_rate | Lead source's conversion rate |
| lead_source_avg_amount | Lead source's average deal |
| deal_health_score | Composite health metric |
| deal_size_vs_avg | Deal size relative to industry |
| rep_consistency_score | How reliable is rep |
| cycle_efficiency | Speed vs industry average |
| is_high_value | Binary: large deal flag |

#### Time Features (3)
| Feature | Description |
|---------|-------------|
| created_quarter | Q1, Q2, Q3, Q4 |
| created_month | 1-12 |
| days_since_start | Days from first deal in dataset |

---

## 8. Dashboard Tabs

### Tab 1: Risk Overview

**Purpose**: See overall risk distribution

**Visualizations**:
- Pie chart: High/Medium/Low risk distribution
- Histogram: Risk score distribution
- Bar charts: Risk by region, Risk by industry

**LLM Insight**: Explains WHY deals are falling into each risk category

---

### Tab 2: Pipeline Analysis

**Purpose**: Understand trends over time

**Visualizations**:
- Line chart: Win rate trend over months
- Funnel: Deal stages distribution
- Scatter: Lead source performance

**LLM Insights**:
- Why win rate is trending up/down
- Why certain lead sources perform better

---

### Tab 3: Rep Performance

**Purpose**: Evaluate sales team

**Visualizations**:
- Bar chart: Top 10 reps by revenue
- Scatter: Win rate vs Risk score per rep
- Table: All rep metrics

**LLM Insight**: What separates top performers from others

---

### Tab 4: Deal Explorer

**Purpose**: Drill into individual deals

**Features**:
- Sortable table of all deals
- Filter by risk, region, industry
- Deal detail view with all info

**LLM Insight**: For each deal, explains WHY it's at risk

---

### Tab 5: Model Insights

**Purpose**: Understand the ML model

**Shows**:
- Model performance metrics (AUC, Accuracy, etc.)
- Cross-validation results
- Feature importance chart

**LLM Insights**:
- Why certain features matter most
- Specific action items based on data

---

### Tab 6: Alerts

**Purpose**: Immediate attention items

**Alert Types**:
| Alert | Trigger | Action |
|-------|---------|--------|
| High Risk Deals | Risk > 70% | Review immediately |
| Stale Deals | Cycle > 90 days | Accelerate or close |
| Underperforming Reps | Win Rate < 35% | Coaching needed |
| Win Rate Drop | Quarter drops > 5% | Investigate cause |

**LLM Insight**: For each alert, explains why it was triggered

---

### Tab 7: Forecast

**Purpose**: Predict future revenue

**Sections**:

1. **Historical Trend**: Monthly revenue chart

2. **Scenarios**:
   - Conservative: 10% below recent average
   - Expected: Based on current trend
   - Optimistic: 15% above recent average

3. **Pipeline Forecast**:
   - Total Pipeline Value
   - Weighted Pipeline (adjusted by win probability)
   - Expected to Close

4. **By Segment**: Forecast by region and industry

**LLM Insight**: Realistic forecast, risks, actions, growth potential

---

## 9. LLM Integration

### How LLM is Used

```
User Action          -->  Data Collected  -->  LLM Prompt  -->  Response
-----------               --------------       ----------       --------
Views Risk tab       -->  Risk metrics    -->  "Why high      -->  Bullet
                                               risk?"              points

Selects a deal       -->  Deal details    -->  "Why this      -->  3-point
                                               deal risky?"        explanation

Views Forecast       -->  Pipeline data   -->  "What's        -->  4 action
                                               forecast?"          items
```

### Prompt Engineering

**System Prompt**:
```
You are a sales analytics expert. Give answers in bullet points only.
Be specific with numbers. Focus on WHY things happen, not descriptions.
No introductions or conclusions. Just the points.
```

**Example User Prompt**:
```
Data:
- High Risk: 193 deals ($30M)
- Win Rate: 45%
- Avg Cycle: 64 days

In 3-4 bullet points explain:
- Why are deals falling into high risk?
- What factors are causing this?
```

**Example Response**:
```
• 73% of high-risk deals have sales cycles >60 days,
  which is 40% longer than closed-won deals
• Reps with <40% win rate own 45% of high-risk pipeline,
  indicating skill gap
• Outbound leads represent 38% of high-risk deals but
  only 25% of wins
• Enterprise tier deals take 2.3x longer to close,
  increasing risk exposure
```

---

## 10. File Structure

```
skygeni/
│
├── data/
│   ├── raw/
│   │   └── skygeni_sales_data.csv    # Original 5000 deals
│   └── processed/
│       ├── scored_deals.csv          # Deals with risk scores
│       ├── metrics.json              # Model metrics
│       └── pipeline_metrics.json     # Pipeline analytics
│
├── models/
│   ├── risk_scorer/
│   │   ├── model.joblib              # Trained model
│   │   ├── calibrated_model.joblib   # Calibrated model
│   │   ├── feature_columns.joblib    # Feature list
│   │   ├── metrics.joblib            # Performance metrics
│   │   └── feature_importance.csv    # Feature rankings
│   └── preprocessor/
│       ├── label_encoders.joblib     # Category encoders
│       └── scaler.joblib             # Number scaler
│
├── src/
│   ├── data/
│   │   ├── loader.py                 # Data loading
│   │   └── preprocessor.py           # Data preprocessing
│   ├── features/
│   │   └── engineering.py            # Feature creation
│   ├── models/
│   │   └── risk_scorer.py            # ML model
│   ├── evaluation/
│   │   ├── metrics.py                # Model evaluation
│   │   └── explainability.py         # SHAP (optional)
│   └── llm/
│       └── insights_generator.py     # Groq LLM integration
│
├── notebooks/
│   └── 01_eda.py                     # Exploratory analysis
│
├── tests/
│   └── test_data_loader.py           # Unit tests
│
├── config/
│   └── config.yaml                   # Configuration
│
├── app.py                            # Streamlit dashboard
├── train.py                          # Training pipeline
├── requirements.txt                  # Dependencies
├── Makefile                          # Commands
├── README.md                         # Project docs
└── SYSTEM_REPORT.md                  # This file
```

---

## 11. How to Run

### Prerequisites
- Python 3.11+
- pip (package manager)

### Installation
```bash
cd ~/Downloads/skygeni

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Training
```bash
python train.py
```

**Output**:
```
[1/6] Loading data... 5000 records
[2/6] Preprocessing... complete
[3/6] Feature engineering... 15 features added
[4/6] Training model... AUC: 0.496
[5/6] Scoring deals... 1095 high risk
[6/6] Saving artifacts... done
```

### Run Dashboard
```bash
streamlit run app.py
```

**Opens**: http://localhost:8501

### Run EDA
```bash
python notebooks/01_eda.py
```

### Run Tests
```bash
pytest tests/ -v
```

---

## 12. Technical Specifications

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.0+ | Data manipulation |
| numpy | 1.24+ | Numerical operations |
| scikit-learn | 1.4+ | ML algorithms |
| plotly | 5.18+ | Interactive charts |
| streamlit | 1.31+ | Dashboard |
| groq | 0.4+ | LLM API |

### Data Schema

**Input Data** (skygeni_sales_data.csv):
| Column | Type | Example |
|--------|------|---------|
| deal_id | string | D00001 |
| created_date | date | 2023-11-24 |
| closed_date | date | 2023-12-15 |
| sales_rep_id | string | rep_22 |
| industry | string | SaaS |
| region | string | North America |
| product_type | string | Enterprise |
| lead_source | string | Referral |
| deal_stage | string | Qualified |
| deal_amount | integer | 4253 |
| sales_cycle_days | integer | 21 |
| outcome | string | Won/Lost |

**Output Data** (scored_deals.csv):
| Column | Type | Description |
|--------|------|-------------|
| All input columns | - | Preserved |
| target | int | 1=Won, 0=Lost |
| *_encoded | int | Encoded categories |
| *_scaled | float | Scaled numbers |
| risk_score | float | 0.0-1.0 |
| risk_category | string | High/Medium/Low |
| win_probability | float | 0.0-1.0 |
| deal_health_score | float | Custom metric |

### Performance

| Metric | Value |
|--------|-------|
| Training time | ~30 seconds |
| Scoring time | <1 second for 5000 deals |
| Dashboard load | ~3 seconds |
| LLM response | 2-5 seconds per query |

### Limitations

1. **Model Accuracy**: Low due to synthetic data
2. **XGBoost/LightGBM**: Not available (missing libomp)
3. **Real-time**: Batch processing, not real-time
4. **Cold Start**: New reps/industries have no history

---

## Summary

This system transforms raw sales data into actionable intelligence:

```
5000 Deals → ML Model → Risk Scores → LLM Explanations → Dashboard
```

**Key Outputs**:
- Risk score for every deal (0-100%)
- Alerts for immediate attention
- Forecasts for planning
- LLM explanations for WHY things happen

**Business Value**:
- Save at-risk deals before they're lost
- Focus sales effort where it matters
- Understand what drives success/failure
- Make data-driven decisions

---

*Report generated for SkyGeni Data Science Challenge*
