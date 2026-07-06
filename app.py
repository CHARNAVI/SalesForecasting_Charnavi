import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 End-to-End Sales Forecasting & Demand Intelligence System")

# -----------------------------
# Load Dataset
# -----------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")

    df.columns = df.columns.str.strip()

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        errors="coerce",
        dayfirst=True
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        errors="coerce",
        dayfirst=True
    )

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter

    return df


df = load_data()

# -----------------------------
# Sidebar
# -----------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments"
    ]
)

# =====================================================
# PAGE 1
# =====================================================

if page == "Sales Overview":

    st.header("📊 Sales Overview Dashboard")

    col1, col2 = st.columns(2)

    with col1:

        yearly_sales = (
            df.groupby("Year")["Sales"]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            yearly_sales,
            x="Year",
            y="Sales",
            title="Total Sales by Year"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        monthly = (
            df.set_index("Order Date")
            .resample("ME")["Sales"]
            .sum()
            .reset_index()
        )

        fig = px.line(
            monthly,
            x="Order Date",
            y="Sales",
            title="Monthly Sales Trend"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Region & Category Analysis")

    region = st.selectbox(
        "Select Region",
        ["All"] + sorted(df["Region"].unique().tolist())
    )

    category = st.selectbox(
        "Select Category",
        ["All"] + sorted(df["Category"].unique().tolist())
    )

    filtered = df.copy()

    if region != "All":
        filtered = filtered[filtered["Region"] == region]

    if category != "All":
        filtered = filtered[filtered["Category"] == category]

    chart = (
        filtered.groupby(["Region", "Category"])["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        chart,
        x="Region",
        y="Sales",
        color="Category",
        barmode="group"
    )

    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# PAGE 2
# =====================================================

elif page == "Forecast Explorer":
    
    st.header("🔮 Forecast Explorer")

    from pathlib import Path
    import pandas as pd
    import plotly.express as px

    BASE_DIR = Path(__file__).parent
    forecast_file = BASE_DIR / "forecast_results.csv"

    st.write("Current Working Directory:", BASE_DIR)

    if not forecast_file.exists():
        st.error(f"forecast_results.csv not found.\n\nExpected Location:\n{forecast_file}")
        st.stop()

    forecast = pd.read_csv(forecast_file)

    st.success("Forecast file loaded successfully!")

    st.write("Columns in Forecast File:")
    st.write(list(forecast.columns))

    st.dataframe(forecast)

    # ----------------------------
    # Plot Forecast
    # ----------------------------

    if "Month" in forecast.columns and "Forecast" in forecast.columns:

        fig = px.line(
            forecast,
            x="Month",
            y="Forecast",
            markers=True,
            title="Monthly Sales Forecast"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.metric(
            "Average Forecast",
            f"{forecast['Forecast'].mean():,.2f}"
        )

        st.metric(
            "Maximum Forecast",
            f"{forecast['Forecast'].max():,.2f}"
        )

    else:
        st.error(
            "forecast_results.csv must contain 'Month' and 'Forecast' columns."
        )
# =====================================================
# PAGE 3
# =====================================================

elif page == "Anomaly Report":

    st.header("🚨 Sales Anomaly Report")

    weekly = (
        df.set_index("Order Date")
        .resample("W")["Sales"]
        .sum()
        .reset_index()
    )

    iso = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    weekly["Anomaly"] = iso.fit_predict(
        weekly[["Sales"]]
    )

    fig = px.line(
        weekly,
        x="Order Date",
        y="Sales",
        title="Weekly Sales"
    )

    anomaly = weekly[
        weekly["Anomaly"] == -1
    ]

    fig.add_scatter(
        x=anomaly["Order Date"],
        y=anomaly["Sales"],
        mode="markers",
        marker=dict(size=10,color="red"),
        name="Anomaly"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected Anomalies")

    st.dataframe(anomaly)

# =====================================================
# PAGE 4
# =====================================================

elif page == "Product Demand Segments":
    
    st.header("📦 Product Demand Segments")

    cluster = (
        df.groupby("Sub-Category")
          .agg({
              "Sales": "sum"
          })
          .reset_index()
    )

    scaler = StandardScaler()

    X = scaler.fit_transform(cluster[["Sales"]])

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    cluster["Cluster"] = kmeans.fit_predict(X)

    # Since only one feature exists, create two plotting columns
    cluster["PC1"] = X[:, 0]
    cluster["PC2"] = 0

    fig = px.scatter(
        cluster,
        x="PC1",
        y="PC2",
        color=cluster["Cluster"].astype(str),
        text="Sub-Category",
        title="Product Demand Segmentation"
    )

    fig.update_traces(
        textposition="top center",
        marker=dict(size=14)
    )

    fig.update_layout(
        xaxis_title="Scaled Sales",
        yaxis_title="Dummy Axis",
        height=650
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Details")

    st.dataframe(cluster, use_container_width=True)