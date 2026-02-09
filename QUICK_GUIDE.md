# SkyGeni - Quick Guide

## What is this?
A system that tells sales teams:
- Which deals will likely FAIL
- WHY they will fail
- WHAT to do about it

---

## How it Works (Simple)

```
Sales Data (5000 deals)
        |
        v
   ML Model scores each deal (0-100% risk)
        |
        v
   LLM explains WHY in plain English
        |
        v
   Dashboard shows everything visually
```

---

## Key Concepts

### Risk Score
- **0-30%**: Low Risk (likely to win)
- **30-60%**: Medium Risk (needs attention)
- **60-100%**: High Risk (likely to lose)

### Deal Health Score
Combines 3 factors:
1. How good is the sales rep? (40%)
2. How fast is the deal moving? (30%)
3. How far in the sales process? (30%)

Higher = healthier deal

---

## Dashboard Tabs

| Tab | Shows | Use When |
|-----|-------|----------|
| Risk Overview | Risk distribution | Daily check |
| Pipeline Analysis | Trends over time | Weekly review |
| Rep Performance | Team stats | Manager meetings |
| Deal Explorer | Individual deals | Deal review |
| Model Insights | What drives risk | Understanding system |
| Alerts | Urgent items | Daily priority |
| Forecast | Future revenue | Planning |

---

## LLM Explanations

The system uses AI (Groq) to explain:
- Why is this deal at risk?
- Why did win rate drop?
- What actions to take?

Example output:
```
• Deal has 78% risk due to 85-day sales cycle (avg is 45)
• Rep_12 has 32% win rate, below team average of 45%
• Action: Escalate to manager or offer discount to close
```

---

## Files You Need to Know

| File | What it does |
|------|--------------|
| `train.py` | Trains the ML model |
| `app.py` | Runs the dashboard |
| `data/raw/skygeni_sales_data.csv` | Input data |
| `data/processed/scored_deals.csv` | Output with scores |

---

## Commands

```bash
# Train model (run first)
python train.py

# Start dashboard
streamlit run app.py

# Run analysis
python notebooks/01_eda.py
```

---

## Flow Diagram

```
+-------------+     +-------------+     +-------------+
|   LOAD      | --> |   PROCESS   | --> |   TRAIN     |
| 5000 deals  |     | Clean data  |     | ML model    |
+-------------+     +-------------+     +-------------+
                                              |
                                              v
+-------------+     +-------------+     +-------------+
|  DASHBOARD  | <-- |   EXPLAIN   | <-- |   SCORE     |
|  7 tabs     |     | LLM reasons |     | Risk 0-100% |
+-------------+     +-------------+     +-------------+
```

---

## Example Use Case

**Scenario**: CRO asks "Why is win rate dropping?"

**System Response**:
1. Shows win rate trend chart (Pipeline Analysis tab)
2. LLM explains:
   - Win rate dropped 5% from Q3 to Q4
   - North America region underperforming
   - Outbound leads converting 12% worse
3. Alerts tab shows:
   - 193 high-risk deals worth $30M
   - 15 reps need coaching
4. Forecast tab shows:
   - Expected revenue: $11M
   - At-risk value: $10M

**Action**: Focus on North America, improve outbound process

---

## Tech Stack (Simple)

| What | Tool |
|------|------|
| Language | Python |
| ML | scikit-learn |
| Charts | Plotly |
| Dashboard | Streamlit |
| AI Explanations | Groq (Llama 3.3) |
| Data | Pandas |

---

## Key Metrics Explained

| Metric | What it means | Good value |
|--------|---------------|------------|
| Win Rate | % of deals won | >50% |
| Risk Score | Chance of losing | <30% |
| AUC | Model accuracy | >0.7 |
| Cycle Days | Time to close | <45 days |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No data found | Run `python train.py` first |
| LLM not working | Check internet connection |
| Charts not loading | Refresh browser |
| Model metrics low | Expected with synthetic data |

---

*For detailed documentation, see SYSTEM_REPORT.md*
