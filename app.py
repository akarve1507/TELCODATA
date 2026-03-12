import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="IBM Telco Churn Decision App", page_icon="📉", layout="wide")

REPO_DIR = Path(__file__).resolve().parent
EXPORTS_DIR = REPO_DIR / "exports"
IMAGES_DIR = REPO_DIR / "images"
def require_columns(df, cols, df_name="dataframe"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"{df_name} is missing required columns: {missing}")
        st.write("Available columns:", list(df.columns))
        st.stop()
@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def file_status_block():
    st.subheader("Project Files")
    expected = {
        "mart_churn_scores.csv": EXPORTS_DIR / "mart_churn_scores.csv",
        "dq_report.csv": EXPORTS_DIR / "dq_report.csv",
        "monitoring_metrics.csv": EXPORTS_DIR / "monitoring_metrics.csv",
        "threshold_policy.json": EXPORTS_DIR / "threshold_policy.json",
    }
    rows = []
    for name, p in expected.items():
        rows.append({"file": name, "present": p.exists(), "path": str(p)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    missing = [r["file"] for r in rows if not r["present"]]
    if missing:
        st.warning(
            "Missing required export files: " + ", ".join(missing) +
            ".\n\nPlace your export CSV/JSON files into the repo folder `exports/` and redeploy."
        )
        st.stop()


def risk_band(p: float) -> str:
    if p >= 0.70:
        return "High"
    if p >= 0.40:
        return "Medium"
    return "Low"


def compute_ev_threshold(cost: float, clv: float, success: float) -> float:
    denom = max(success * clv, 1e-9)
    return float(cost / denom)


def profit_from_targeting(y_true: np.ndarray, probs: np.ndarray, t: float, clv: float, cost: float, success: float) -> dict:
    # Expected profit approximation: target if prob >= t
    target = probs >= t
    # Expected saved value: sum(p_i * success * clv) over targeted
    saved = float(np.sum(probs[target] * success * clv))
    spend = float(np.sum(target) * cost)
    return {
        "targeted": int(np.sum(target)),
        "expected_saved_value": saved,
        "offer_spend": spend,
        "expected_profit": saved - spend,
        "target_rate": float(np.mean(target))
    }


st.title("📉 IBM Telco Churn — Decision Optimization App")
st.caption("A deployable decision tool built from the churn analytics pipeline: scoring, risk bands, cost-aware targeting, and monitoring.")

with st.expander("✅ Check required files", expanded=False):
    file_status_block()

# Load exports
mart = load_csv(EXPORTS_DIR / "mart_churn_scores.csv")
dq = load_csv(EXPORTS_DIR / "dq_report.csv")
mon = load_csv(EXPORTS_DIR / "monitoring_metrics.csv")
policy = load_json(EXPORTS_DIR / "threshold_policy.json")

# Sidebar controls
st.sidebar.header("Decision Controls")
retention_cost = st.sidebar.number_input("Retention offer cost ($)", min_value=0.0, value=20.0, step=1.0)
clv = st.sidebar.number_input("Customer lifetime value saved (CLV, $)", min_value=0.0, value=300.0, step=10.0)
success = st.sidebar.slider("Retention success probability", min_value=0.0, max_value=1.0, value=0.30, step=0.05)

st.sidebar.header("Policy")
profit_t = float(policy.get("threshold_profit_opt", 0.05))
f1_t = float(policy.get("threshold_f1_opt", 0.34))
ev_t = compute_ev_threshold(retention_cost, clv, success)

policy_choice = st.sidebar.radio(
    "Choose threshold policy",
    options=["Profit-opt (from notebook)", "F1-opt (from notebook)", "Expected Value (closed-form)", "Manual"],
    index=0
)

if policy_choice == "Profit-opt (from notebook)":
    chosen_t = profit_t
elif policy_choice == "F1-opt (from notebook)":
    chosen_t = f1_t
elif policy_choice == "Expected Value (closed-form)":
    chosen_t = ev_t
else:
    chosen_t = st.sidebar.slider("Manual threshold", 0.0, 1.0, float(profit_t), 0.01)

# Main KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers", f"{len(mart):,}")
col2.metric("Mean churn probability", f"{mart['churn_prob'].mean():.3f}")
col3.metric("High risk %", f"{(mart['risk_band']=='High').mean()*100:.1f}%")
col4.metric("Chosen threshold", f"{chosen_t:.3f}")

st.divider()

# Decision Optimization Panel
st.subheader("1) Business Optimization")
left, right = st.columns([1.2, 1])

with left:
    st.markdown("### Targeting Summary")
    probs = mart["churn_prob"].to_numpy()
    # We do not have true labels in deployment scoring table; use expected value only
    summary = profit_from_targeting(None, probs, chosen_t, clv, retention_cost, success)

    st.write(
        f"**Targeted customers:** {summary['targeted']:,}  \\n"        f"**Target rate:** {summary['target_rate']*100:.1f}%  \\n"        f"**Expected saved value:** ${summary['expected_saved_value']:,.0f}  \\n"        f"**Offer spend:** ${summary['offer_spend']:,.0f}  \\n"        f"**Expected profit:** ${summary['expected_profit']:,.0f}"
    )

    st.markdown("### Profit curve (expected value)")
    grid = np.linspace(0.01, 0.99, 99)
    profits = []
    for t in grid:
        s = profit_from_targeting(None, probs, t, clv, retention_cost, success)
        profits.append(s["expected_profit"])

    chart_df = pd.DataFrame({"threshold": grid, "expected_profit": profits})
    st.line_chart(chart_df, x="threshold", y="expected_profit")

with right:
    st.markdown("### Notes")
    st.write(
        "- **Profit-opt** and **F1-opt** thresholds come from your notebook evaluation.\n"
        "- **Expected Value** threshold uses the closed-form decision rule: target when `p >= cost / (success * CLV)`.\n"
        "- This app uses *expected* profit (no ground-truth labels in production scoring)."
    )
    st.code(
        "EV_i = p_i * success * CLV - cost\n"
        "Target if EV_i > 0  =>  p_i >= cost/(success*CLV)",
        language="text",
    )

st.divider()

# Customer Explorer
st.subheader("2) Customer Explorer")
filters = st.columns(4)

band = filters[0].multiselect("Risk band", options=sorted(mart["risk_band"].unique()), default=sorted(mart["risk_band"].unique()))
contract = filters[1].multiselect("Contract", options=sorted(mart["Contract"].unique()), default=sorted(mart["Contract"].unique()))
internet = filters[2].multiselect("InternetService", options=sorted(mart["InternetService"].unique()), default=sorted(mart["InternetService"].unique()))
pay = filters[3].multiselect("PaymentMethod", options=sorted(mart["PaymentMethod"].unique()), default=sorted(mart["PaymentMethod"].unique()))

view = mart[
    mart["risk_band"].isin(band)
    & mart["Contract"].isin(contract)
    & mart["InternetService"].isin(internet)
    & mart["PaymentMethod"].isin(pay)
].copy()

view = view.sort_values("churn_prob", ascending=False)

st.write(f"Showing **{len(view):,}** customers")

show_cols = [
    "churn_prob", "risk_band", "recommended_action", "top_drivers",
    "tenure", "MonthlyCharges", "TotalCharges", "EngagementScore",
    "Contract", "InternetService", "PaymentMethod",
]
show_cols = [c for c in show_cols if c in view.columns]

st.dataframe(view[show_cols].head(200), use_container_width=True)

st.divider()

# Data Health
st.subheader("3) Data Quality + Monitoring")
colA, colB = st.columns(2)

with colA:
    st.markdown("### Data Quality")
    st.dataframe(dq, use_container_width=True)

with colB:
    st.markdown("### Monitoring Snapshot")
    st.dataframe(mon, use_container_width=True)

st.caption("Tip: append new rows to monitoring_metrics.csv for each refresh to visualize drift over time.")
