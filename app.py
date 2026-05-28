import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import io
import os
from datetime import datetime
from collections import defaultdict
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="SmartStock 📦", page_icon="📦", layout="wide")

# ── Rate limiting ────────────────────────────────────────────
if "request_counts" not in st.session_state:
    st.session_state.request_counts = defaultdict(list)

FREE_TIER_LIMIT = 3  # free analyses per day

def get_ip():
    try:
        return st.context.headers.get("X-Forwarded-For", "unknown").split(",")[0].strip()
    except:
        return "unknown"

def is_rate_limited():
    ip = get_ip()
    now = time.time()
    window = 24 * 60 * 60
    st.session_state.request_counts[ip] = [
        t for t in st.session_state.request_counts[ip] if now - t < window
    ]
    if len(st.session_state.request_counts[ip]) >= FREE_TIER_LIMIT:
        return True
    st.session_state.request_counts[ip].append(now)
    return False

# ── Data cleaning ────────────────────────────────────────────
def parse_date(val):
    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"]:
        try:
            return pd.to_datetime(val, format=fmt)
        except:
            continue
    return pd.NaT

def clean_data(df_raw):
    df = df_raw.copy()
    df.columns = df.columns.str.lower().str.strip()
    required = {"date", "product", "quantity_sold"}
    missing = required - set(df.columns)
    if missing:
        return None, f"Missing columns: {missing}"
    df = df.dropna(subset=["quantity_sold"])
    df["quantity_sold"] = pd.to_numeric(df["quantity_sold"], errors="coerce").fillna(0).astype(int)
    df = df[df["quantity_sold"] >= 0]
    df["date"] = df["date"].apply(parse_date)
    df = df.dropna(subset=["date"])
    df["product"] = df["product"].str.strip()
    df = df.sort_values("date").reset_index(drop=True)
    return df, None

def extract_stats(df):
    max_date = df["date"].max()
    last_30 = df[df["date"] >= max_date - pd.Timedelta(days=30)]
    prev_30 = df[(df["date"] >= max_date - pd.Timedelta(days=60)) &
                 (df["date"] < max_date - pd.Timedelta(days=30))]
    last_30_avg = last_30.groupby("product")["quantity_sold"].mean().round(1)
    prev_30_avg = prev_30.groupby("product")["quantity_sold"].mean().round(1)
    trend = ((last_30_avg - prev_30_avg) / prev_30_avg.replace(0, 1) * 100).round(1)

    # Fix — always add day_of_week column cleanly
    df2 = df.copy()
    df2["day_of_week"] = df2["date"].dt.day_name()
    best_day = df2.groupby(["product", "day_of_week"])["quantity_sold"].mean()
    best_day = best_day.reset_index().sort_values("quantity_sold", ascending=False).groupby("product").first()["day_of_week"]

    stats = {}
    for product in df["product"].unique():
        stats[product] = {
            "avg_daily_last_30_days": float(last_30_avg.get(product, 0)),
            "avg_daily_prev_30_days": float(prev_30_avg.get(product, 0)),
            "trend_percent": float(trend.get(product, 0)),
            "best_sales_day": str(best_day.get(product, "Unknown"))
        }
    return stats

def generate_alerts(stats, llm):
    prompt = PromptTemplate.from_template("""
Based on this sales data, generate a JSON array of alerts for each product.
Return ONLY valid JSON, no explanation, no markdown backticks.

For each product:
- product: product name
- status: one of critical, warning, or healthy
- message: one sentence plain English alert
- action: specific recommended action
- reorder_qty: recommended reorder quantity (integer)
- reorder_by_days: days until they should reorder (integer)

Sales data:
{stats}

Return only a JSON array.
""")
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"stats": json.dumps(stats, indent=2)})
    clean = raw.strip().replace("```json","").replace("```","").strip()
    return json.loads(clean)

def generate_forecast(stats, df, llm):
    prompt = PromptTemplate.from_template("""
You are an expert retail inventory analyst helping a small business owner.
Based on 6 months of sales data, provide practical inventory forecasts.

Be specific, use plain language, use actual product names and numbers.
Avoid jargon. Format with clear sections.

Sales Data:
{stats}

Date range: {date_range}
Products tracked: {num_products}

Provide: 30-day forecast, stockout risks, overstock warnings, reorder recommendations.
""")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "stats": json.dumps(stats, indent=2),
        "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
        "num_products": len(stats)
    })

def plot_trends(df):
    products = df["product"].unique()
    n = len(products)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4))
    if n == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, "__iter__") else [row])]
    for i, product in enumerate(products):
        pdf = df[df["product"] == product].copy()
        pdf = pdf.set_index("date").resample("D")["quantity_sold"].sum().reset_index()
        pdf["rolling"] = pdf["quantity_sold"].rolling(7).mean()
        axes_flat[i].bar(pdf["date"], pdf["quantity_sold"], alpha=0.3, color="steelblue")
        axes_flat[i].plot(pdf["date"], pdf["rolling"], color="red", linewidth=2, label="7-day avg")
        axes_flat[i].set_title(product, fontsize=10)
        axes_flat[i].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        axes_flat[i].legend(fontsize=8)
        axes_flat[i].grid(True, alpha=0.3)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    return fig

# ── UI ───────────────────────────────────────────────────────
st.title("📦 SmartStock")
st.subheader("AI-powered inventory forecasting for small businesses")

with st.expander("📋 How to use SmartStock"):
    st.markdown("""
    1. Export your sales data as a CSV from your POS or spreadsheet
    2. Make sure it has these columns: **date**, **product**, **quantity_sold**
    3. Upload it below and click Analyze
    4. Get AI-powered forecasts and reorder recommendations

    **Free tier:** 3 analyses per day | **Pro:** unlimited + email reports
    """)

st.markdown("---")

# Free tier info
st.info("🆓 Free tier: 3 analyses per day. Upgrade to Pro for unlimited access.")

uploaded_file = st.file_uploader("📄 Upload your sales CSV", type="csv")

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    df, error = clean_data(df_raw)

    if error:
        st.error(f"❌ {error}")
        st.stop()

    st.success(f"✅ Loaded {len(df)} rows across {df['product'].nunique()} products")

    # Data preview
    with st.expander("👀 Preview your data"):
        st.dataframe(df.head(20))

    # Summary table
    st.subheader("📊 Sales Summary")
    summary = df.groupby("product")["quantity_sold"].agg(
        Total_Sold="sum",
        Avg_Daily="mean",
        Days_Tracked="count"
    ).round(1).reset_index().sort_values("Total_Sold", ascending=False)
    st.dataframe(summary, use_container_width=True)

    # Charts
    st.subheader("📈 Sales Trends")
    fig = plot_trends(df)
    st.pyplot(fig)

    st.markdown("---")

    if st.button("🤖 Generate AI Forecast", use_container_width=True):
        if is_rate_limited():
            st.error("⚠️ You've used your 3 free analyses today. Come back tomorrow or upgrade to Pro!")
            st.stop()

        with st.spinner("Analyzing your inventory data..."):
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=st.secrets["GROQ_API_KEY"]
            )
            stats = extract_stats(df)

        # Alerts dashboard
        st.subheader("🚨 Inventory Alerts")
        with st.spinner("Generating alerts..."):
            try:
                alerts = generate_alerts(stats, llm)
                critical = [a for a in alerts if a["status"] == "critical"]
                warning = [a for a in alerts if a["status"] == "warning"]
                healthy = [a for a in alerts if a["status"] == "healthy"]

                if critical:
                    st.error(f"🔴 {len(critical)} Critical")
                    for a in critical:
                        st.error(f"**{a['product']}** — {a['message']}\n\n💡 {a['action']} (Reorder {a['reorder_qty']} units within {a['reorder_by_days']} days)")

                if warning:
                    st.warning(f"🟡 {len(warning)} Warnings")
                    for a in warning:
                        st.warning(f"**{a['product']}** — {a['message']}\n\n💡 {a['action']}")

                if healthy:
                    st.success(f"🟢 {len(healthy)} Healthy")
                    for a in healthy:
                        st.success(f"**{a['product']}** — {a['message']}")

            except Exception as e:
                st.warning(f"Could not parse alerts: {e}")

        # Full forecast
        st.subheader("📋 Full AI Forecast")
        with st.spinner("Writing detailed forecast..."):
            forecast = generate_forecast(stats, df, llm)
            st.write(forecast)

        # Download report
        st.markdown("---")
        report = f"SmartStock Inventory Report\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        report += "=" * 50 + "\nALERTS\n" + "=" * 50 + "\n"
        if alerts:
            for a in alerts:
                emoji = "🔴" if a["status"] == "critical" else "🟡" if a["status"] == "warning" else "🟢"
                report += f"{emoji} {a['product']}: {a['message']}\nAction: {a['action']}\n\n"
        report += "=" * 50 + "\nFULL FORECAST\n" + "=" * 50 + "\n" + forecast

        st.download_button(
            "⬇️ Download Full Report",
            data=report,
            file_name=f"smartstock_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
print("✅ app.py written!")
