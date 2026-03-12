import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About this project")

st.markdown(
"""
### IBM Telco Churn — Decision Intelligence

This Streamlit app is the **delivery layer** for a churn analytics pipeline.

It is designed to demonstrate **FAANG-style signals**:
- **Business optimization**: cost-aware targeting with expected value logic
- **Explainability**: customer-level driver summaries and segment filters
- **Data quality & monitoring**: lightweight checks + drift metrics snapshot

### What you need in `exports/`
Place these files in the repository folder `exports/`:
- `mart_churn_scores.csv`
- `dq_report.csv`
- `monitoring_metrics.csv`
- `threshold_policy.json`

These are produced by your Colab notebook export step.

### Deployment
- Deploy on **Streamlit Community Cloud** with main file: `app.py`.
"""
)
