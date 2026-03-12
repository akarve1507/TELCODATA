# IBM Telco Churn — Streamlit App + Power BI Exports

This repo contains a Streamlit decision app that reads churn scoring outputs from `exports/`.

## 1) Required files
From your Colab notebook, export the following into `exports/`:
- `mart_churn_scores.csv`
- `dq_report.csv`
- `monitoring_metrics.csv`
- `threshold_policy.json`
- *(optional)* `deployment_model.joblib`

## 2) Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 3) Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. In Streamlit Community Cloud: **New app**
3. Select your repo and set **Main file path** to: `app.py`

## 4) Power BI
Use `exports/mart_churn_scores.csv` as your main table.
Suggested pages:
- Executive: risk distribution + expected profit curve screenshot
- Campaign list: high-risk customers + drivers + action
- Data health: `dq_report.csv` + `monitoring_metrics.csv`

## Notes
- This app uses **expected value** profit (no ground-truth labels in production scoring).
- Threshold policies from the notebook (profit-opt and F1-opt) are loaded from `threshold_policy.json`.
