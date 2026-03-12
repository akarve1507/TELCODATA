import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Page config + styling
# =========================
st.set_page_config(
    page_title="RetentionIQ | Churn Decision Intelligence",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Reduce top padding */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
/* Clean card look */
.kpi-card {
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 16px;
    background: rgba(255,255,255,0.03);
}
.small-muted { color: rgba(255,255,255,0.65); font-size: 0.9rem; }
.section-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 0.25rem; }
hr { border: none; border-top: 1px solid rgba(255,255,255,0.12); margin: 1rem 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================
# Paths
# =========================
REPO_DIR = Path(__file__).resolve().parent
EXPORTS_DIR = REPO_DIR / "exports"

EXPECTED_FILES = {
    "mart_churn_scores.csv": EXPORTS_DIR / "mart_churn_scores.csv",
    "dq_report.csv": EXPORTS_DIR / "dq_report.csv",
    "monitoring_metrics.csv": EXPORTS_DIR / "monitoring_metrics.csv",
    "threshold_policy.json": EXPORTS_DIR / "threshold_policy.json",
}


# =========================
# Helpers
# =========================
def stop_with_file_instructions(missing: list[str]) -> None:
    st.error("Missing required export files in the GitHub repo.")
    st.write("Your Streamlit app can only read files that are committed to GitHub.")
    st.write("Place these inside the repo folder `exports/`:")
    st.code("\n".join(missing))
    st.stop()


def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"`{name}` is missing required columns: {missing}")
        st.write("Available columns:", list(df.columns))
        st.stop()


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def risk_band_from_prob(p: float) -> str:
    if p >= 0.70:
        return "High"
    if p >= 0.40:
        return "Medium"
    return "Low"


def compute_ev_threshold(cost: float, clv: float, success: float) -> float:
    denom = max(success * clv, 1e-9)
    return float(cost / denom)


def expected_profit_summary(probs: np.ndarray, t: float, clv: float, cost: float, success: float) -> dict:
    target = probs >= t
    saved = float(np.sum(probs[target] * success * clv))
    spend = float(np.sum(target) * cost)
    return {
        "targeted": int(np.sum(target)),
        "target_rate": float(np.mean(target)),
        "expected_saved_value": saved,
        "offer_spend": spend,
        "expected_profit": saved - spend,
    }


def safe_sorted_unique(series: pd.Series) -> list:
    # Avoid NaN issues, convert to python scalars, sort
    vals = [v for v in series.dropna().unique().tolist()]
    return sorted(vals)


def format_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


def kpi_card(title: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="section-title">{title}</div>
            <div style="font-size:1.6rem; font-weight:800; margin-top:0.2rem;">{value}</div>
            <div class="small-muted">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Header
# =========================
st.title("📉 RetentionIQ — Churn Decision Intelligence (IBM Telco Demo)")
st.caption(
    "A decision-focused churn analytics product: risk scoring, cost-aware targeting, customer-level explanations, and monitoring."
)

# =========================
# File check
# =========================
missing_files = [name for name, p in EXPECTED_FILES.items() if not p.exists()]
with st.expander("✅ Deployment check: required files", expanded=False):
    rows = [{"file": k, "present": EXPECTED_FILES[k].exists(), "path": str(EXPECTED_FILES[k])} for k in EXPECTED_FILES]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    if missing_files:
        stop_with_file_instructions(missing_files)

# =========================
# Load exports
# =========================
mart = load_csv(EXPORTS_DIR / "mart_churn_scores.csv")
dq = load_csv(EXPORTS_DIR / "dq_report.csv")
mon = load_csv(EXPORTS_DIR / "monitoring_metrics.csv")
policy = load_json(EXPORTS_DIR / "threshold_policy.json")

# Required columns for the app (non-negotiable for product quality)
require_columns(mart, ["churn_prob"], "mart_churn_scores.csv")

# Backward compatibility: if some columns are missing, we create them gracefully
if "risk_band" not in mart.columns:
    mart["risk_band"] = mart["churn_prob"].apply(risk_band_from_prob)

if "recommended_action" not in mart.columns:
    def _default_action(band: str) -> str:
        if band == "High":
            return "Priority retention offer + contract/autopay push"
        if band == "Medium":
            return "Low-cost nudge + service engagement bundle"
        return "No offer; monitor"
    mart["recommended_action"] = mart["risk_band"].apply(_default_action)

if "top_drivers" not in mart.columns:
    mart["top_drivers"] = "Not available (export missing `top_drivers`)"

# Make sure numeric
mart["churn_prob"] = pd.to_numeric(mart["churn_prob"], errors="coerce").fillna(0.0)

# =========================
# Sidebar: Decision controls
# =========================
st.sidebar.header("Decision Controls")
retention_cost = st.sidebar.number_input("Retention offer cost ($)", min_value=0.0, value=20.0, step=1.0)
clv = st.sidebar.number_input("Customer lifetime value saved (CLV, $)", min_value=0.0, value=300.0, step=10.0)
success = st.sidebar.slider("Retention success probability", 0.0, 1.0, 0.30, 0.05)

st.sidebar.header("Threshold Policy")

profit_t = float(policy.get("threshold_profit_opt", 0.05))
f1_t = float(policy.get("threshold_f1_opt", 0.34))
ev_t = compute_ev_threshold(retention_cost, clv, success)

policy_choice = st.sidebar.radio(
    "Policy",
    options=["Profit-opt (Notebook)", "F1-opt (Notebook)", "Expected Value (Closed-form)", "Manual"],
    index=0
)

if policy_choice == "Profit-opt (Notebook)":
    chosen_t = profit_t
elif policy_choice == "F1-opt (Notebook)":
    chosen_t = f1_t
elif policy_choice == "Expected Value (Closed-form)":
    chosen_t = ev_t
else:
    chosen_t = st.sidebar.slider("Manual threshold", 0.0, 1.0, float(profit_t), 0.01)

st.sidebar.markdown("---")
st.sidebar.caption(
    "This app uses **expected profit** (no labels in production scoring). "
    "Notebook evaluation remains your ground-truth benchmark."
)

# =========================
# Tabs
# =========================
tab_exec, tab_customers, tab_insights, tab_health, tab_about = st.tabs(
    ["📌 Executive Summary", "🔎 Customer Explorer", "📊 Segment Insights", "🧪 Data Health", "ℹ️ About"]
)

# =========================
# EXEC SUMMARY
# =========================
with tab_exec:
    probs = mart["churn_prob"].to_numpy()
    summary = expected_profit_summary(probs, chosen_t, clv, retention_cost, success)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Customers", f"{len(mart):,}", "Scored customers in mart_churn_scores.csv")
    with k2:
        kpi_card("Mean churn probability", f"{mart['churn_prob'].mean():.3f}", "Average risk in current batch")
    with k3:
        high_pct = float((mart["risk_band"] == "High").mean() * 100)
        kpi_card("High-risk share", f"{high_pct:.1f}%", "Risk band: churn_prob ≥ 0.70")
    with k4:
        kpi_card("Chosen threshold", f"{chosen_t:.3f}", f"Policy: {policy_choice}")

    st.markdown("<hr>", unsafe_allow_html=True)

    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Business Optimization (Expected Profit)")
        st.write(
            f"- **Targeted customers:** {summary['targeted']:,}\n"
            f"- **Target rate:** {summary['target_rate']*100:.1f}%\n"
            f"- **Expected saved value:** {format_money(summary['expected_saved_value'])}\n"
            f"- **Offer spend:** {format_money(summary['offer_spend'])}\n"
            f"- **Expected profit:** **{format_money(summary['expected_profit'])}**"
        )

        st.markdown("#### Profit curve vs threshold")
        grid = np.linspace(0.01, 0.99, 99)
        profits = [expected_profit_summary(probs, t, clv, retention_cost, success)["expected_profit"] for t in grid]
        chart_df = pd.DataFrame({"threshold": grid, "expected_profit": profits})
        st.line_chart(chart_df, x="threshold", y="expected_profit")

    with right:
        st.subheader("Decision Rule")
        st.code(
            "EV_i = p_i * success * CLV - cost\n"
            "Target if EV_i > 0  ⇒  p_i >= cost / (success * CLV)\n",
            language="text",
        )

        st.markdown("#### Current inputs")
        st.write(
            f"- Offer cost: **{format_money(retention_cost)}**\n"
            f"- CLV saved: **{format_money(clv)}**\n"
            f"- Success rate: **{success:.2f}**\n"
            f"- EV threshold: **{ev_t:.3f}**"
        )

        st.markdown("#### Risk distribution")
        band_counts = mart["risk_band"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).astype(int)
        st.bar_chart(band_counts)

# =========================
# CUSTOMER EXPLORER
# =========================
with tab_customers:
    st.subheader("Customer Explorer (Who to target + why)")
    st.caption("Filter customers, inspect risk, drivers, and recommended actions. Export top candidates for campaigns.")

    # Robust filter widgets
    filters = st.columns(4)

    risk_options = safe_sorted_unique(mart["risk_band"])
    band = filters[0].multiselect("Risk band", options=risk_options, default=risk_options)

    # Only show filters if columns exist
    def multiselect_if_exists(col, label, container, default_all=True):
        if col not in mart.columns:
            container.info(f"{label} unavailable (missing column: {col})")
            return None
        opts = safe_sorted_unique(mart[col])
        default = opts if default_all else []
        return container.multiselect(label, options=opts, default=default)

    contract = multiselect_if_exists("Contract", "Contract", filters[1])
    internet = multiselect_if_exists("InternetService", "InternetService", filters[2])
    pay = multiselect_if_exists("PaymentMethod", "PaymentMethod", filters[3])

    view = mart[mart["risk_band"].isin(band)].copy()
    if contract is not None:
        view = view[view["Contract"].isin(contract)]
    if internet is not None:
        view = view[view["InternetService"].isin(internet)]
    if pay is not None:
        view = view[view["PaymentMethod"].isin(pay)]

    view = view.sort_values("churn_prob", ascending=False)

    c1, c2, c3 = st.columns([1, 1, 1.2])
    top_n = c1.slider("Preview top N", 50, 500, 200, 50)
    min_prob = c2.slider("Minimum churn probability", 0.0, 1.0, float(chosen_t), 0.01)
    view = view[view["churn_prob"] >= min_prob]

    c3.write(f"**Filtered customers:** {len(view):,}")

    show_cols = [
        "churn_prob", "risk_band", "recommended_action", "top_drivers",
        "tenure", "MonthlyCharges", "TotalCharges", "EngagementScore",
        "Contract", "InternetService", "PaymentMethod",
    ]
    show_cols = [c for c in show_cols if c in view.columns]

    st.dataframe(view[show_cols].head(top_n), use_container_width=True)

    # Download for campaigns
    st.download_button(
        label="⬇️ Download filtered customers (CSV)",
        data=view[show_cols].to_csv(index=False).encode("utf-8"),
        file_name="filtered_customers.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Customer Drill-down (Explainability)")

    # Choose a row to inspect
    if len(view) == 0:
        st.info("No customers match the current filters.")
    else:
        view_reset = view.reset_index(drop=True)
        idx = st.selectbox("Select a customer row (from filtered table)", options=list(range(min(len(view_reset), 500))))
        row = view_reset.iloc[int(idx)].to_dict()

        a, b = st.columns([1, 1])
        with a:
            st.markdown("##### Risk & Recommendation")
            st.write(f"**Churn probability:** {row.get('churn_prob', 0.0):.3f}")
            st.write(f"**Risk band:** {row.get('risk_band', '—')}")
            st.write(f"**Recommended action:** {row.get('recommended_action', '—')}")
            st.write(f"**Top drivers:** {row.get('top_drivers', '—')}")
        with b:
            st.markdown("##### Key attributes")
            keys = ["tenure", "MonthlyCharges", "TotalCharges", "EngagementScore", "Contract", "InternetService", "PaymentMethod"]
            attrs = {k: row.get(k, "—") for k in keys if k in row}
            st.json(attrs)

# =========================
# SEGMENT INSIGHTS
# =========================
with tab_insights:
    st.subheader("Segment Insights (What drives risk at scale)")
    st.caption("This section helps stakeholders understand dominant churn patterns by segment.")

    # Top decile analysis
    dec = st.slider("High-risk slice (top % by churn probability)", 5, 30, 10, 5)
    cut = mart["churn_prob"].quantile(1 - dec / 100)
    high = mart[mart["churn_prob"] >= cut].copy()
    rest = mart[mart["churn_prob"] < cut].copy()

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        kpi_card("High-risk customers", f"{len(high):,}", f"Top {dec}% by churn probability")
    with s2:
        kpi_card("Avg prob (high-risk)", f"{high['churn_prob'].mean():.3f}", "Mean predicted risk in slice")
    with s3:
        kpi_card("Avg prob (rest)", f"{rest['churn_prob'].mean():.3f}", "Mean predicted risk outside slice")
    with s4:
        lift = (high["churn_prob"].mean() / max(rest["churn_prob"].mean(), 1e-9))
        kpi_card("Risk lift (high/rest)", f"{lift:.2f}×", "Relative risk concentration")

    st.markdown("<hr>", unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    def show_segment_share(col: str):
        if col not in mart.columns:
            st.info(f"Segment view unavailable: missing column `{col}`")
            return
        seg = pd.crosstab(mart[col], mart["churn_prob"] >= cut, normalize="columns")
        seg.columns = ["Rest", f"Top{dec}%"]
        seg = seg.sort_values(f"Top{dec}%", ascending=False)
        st.dataframe(seg, use_container_width=True)

    with left:
        st.markdown("#### Segment composition (High-risk vs Rest)")
        show_segment_share("Contract")
        show_segment_share("InternetService")
        show_segment_share("PaymentMethod")

    with right:
        st.markdown("#### Driver text frequency (top drivers)")
        # Simple: count driver phrases
        driver_series = high["top_drivers"].fillna("").astype(str)
        tokens = []
        for s in driver_series.tolist():
            parts = [p.strip() for p in s.split(";") if p.strip()]
            tokens.extend(parts)
        if tokens:
            freq = pd.Series(tokens).value_counts().head(12)
            st.bar_chart(freq)
        else:
            st.info("No driver text available (top_drivers column empty).")

# =========================
# DATA HEALTH
# =========================
with tab_health:
    st.subheader("Data Quality + Monitoring")
    st.caption("Production maturity: freshness, schema checks, drift indicators, and monitoring snapshots.")

    a, b = st.columns(2)

    with a:
        st.markdown("### Data Quality Report")
        st.dataframe(dq, use_container_width=True)

        if "generated_at" in dq.columns:
            st.write("**Last generated:**", str(dq.loc[0, "generated_at"]))
        st.download_button(
            "⬇️ Download dq_report.csv",
            data=dq.to_csv(index=False).encode("utf-8"),
            file_name="dq_report.csv",
            mime="text/csv",
            use_container_width=True
        )

    with b:
        st.markdown("### Monitoring Snapshot")
        st.dataframe(mon, use_container_width=True)

        # Quick visual: PSI bars if present
        psi_cols = [c for c in ["psi_prob", "psi_tenure", "psi_monthlycharges"] if c in mon.columns]
        if psi_cols:
            psi_vals = mon[psi_cols].iloc[0].astype(float)
            st.markdown("#### Drift indicators (PSI)")
            st.bar_chart(psi_vals)

        st.download_button(
            "⬇️ Download monitoring_metrics.csv",
            data=mon.to_csv(index=False).encode("utf-8"),
            file_name="monitoring_metrics.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Monitoring policy (recommended)")
    st.write(
        "- Alert if **PSI > 0.2** for key features (distribution shift)\n"
        "- Alert if mean/p90 score shifts materially vs baseline\n"
        "- Retrain if performance drops on newly labeled batches (PR-AUC drop > 0.05)\n"
        "- Re-check calibration if Brier increases materially"
    )

# =========================
# ABOUT
# =========================
with tab_about:
    st.subheader("About this project")
    st.write(
        "This is a portfolio-grade decision intelligence system built on the IBM Telco churn dataset.\n\n"
        "**What makes it senior-level:**\n"
        "- Business optimization (cost-aware targeting)\n"
        "- Decision thresholds as policies\n"
        "- Customer + segment explainability\n"
        "- Monitoring and drift awareness\n\n"
        "This app is designed to demonstrate end-to-end ownership: modeling → decisions → delivery."
    )

    st.markdown("### Data / Repo notes")
    st.write("- Export files must be committed to GitHub under `exports/` for Streamlit Cloud.")
    st.write(f"- App loaded at: `{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC`")

    st.markdown("### Files")
    st.code("\n".join([f"- {k}" for k in EXPECTED_FILES.keys()]))
