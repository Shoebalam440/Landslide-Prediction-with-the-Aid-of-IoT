from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

DATA_PATH = "wsn_landslide_data.csv"
LABEL_COLUMN = "Label"
PROJECT_TITLE = "Landslide Prediction with the Aid of IoT"
SCENARIO_STATE_KEY = "scenario_history"
INPUT_STATE_KEY = "input_defaults"

PRESET_SCENARIOS: Dict[str, Dict[str, float]] = {
    "Mountain Village": {
        "Rainfall_mm": 260,
        "Slope_Angle": 65,
        "Soil_Saturation": 0.8,
        "Vegetation_Cover": 0.2,
    },
    "Coastal Highway": {
        "Rainfall_mm": 120,
        "Slope_Angle": 35,
        "Soil_Saturation": 0.55,
        "Proximity_to_Water": 0.9,
        "Distance_to_Road_m": 50,
    },
    "Urban Construction": {
        "Rainfall_mm": 80,
        "Slope_Angle": 25,
        "Soil_Saturation": 0.45,
        "Land_Use_Urban": 1,
        "Land_Use_Forest": 0,
    },
}

GLOSSARY = {
    "Rainfall_mm": "24h rainfall observed at the slope.",
    "Rainfall_3Day": "Cumulative rainfall over the last 3 days.",
    "Rainfall_7Day": "Cumulative rainfall over the last 7 days.",
    "Slope_Angle": "Average incline of the slope (degrees).",
    "Soil_Saturation": "Fraction of pore space filled with water.",
    "Vegetation_Cover": "Surface vegetation density (0-1).",
    "Soil_Erosion_Rate": "Estimated annual erosion rate.",
    "Microseismic_Activity": "Sensor counts of micro tremors.",
    "Acoustic_Emission_dB": "Acoustic sensor energy released by soil stress.",
    "Pore_Water_Pressure_kPa": "Water pressure within soil pores.",
    "Soil_Moisture_Content": "Volumetric water content fraction.",
    "NDVI_Index": "Normalized vegetation index from satellite imagery.",
    "Distance_to_Road_m": "Distance from site to nearest road segment.",
    "Earthquake_Activity": "Regional earthquake activity index.",
}

CORE_FEATURES = [
    "Rainfall_mm",
    "Rainfall_3Day",
    "Slope_Angle",
    "Soil_Saturation",
    "Soil_Moisture_Content",
    "Pore_Water_Pressure_kPa",
    "Vegetation_Cover",
    "NDVI_Index",
    "Microseismic_Activity",
    "Acoustic_Emission_dB",
    "Earthquake_Activity",
]

st.set_page_config(
    page_title=PROJECT_TITLE,
    page_icon="🌏",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    feature_cols = [col for col in df.columns if col != LABEL_COLUMN]
    X = df[feature_cols]
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, y_prob),
    }

    importance_series = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "confusion": confusion_matrix(y_test, y_pred, labels=[0, 1]),
        "importances": importance_series,
        "roc": (fpr, tpr),
        "y_test": y_test,
        "y_prob": y_prob,
    }


@st.cache_data(show_spinner=False)
def build_feature_metadata(df: pd.DataFrame):
    metadata = {}
    for col in df.columns:
        if col == LABEL_COLUMN:
            continue
        series = df[col].dropna()
        unique = series.unique()
        is_binary = len(unique) <= 2 and set(unique).issubset({0, 1})
        metadata[col] = {
            "type": "binary" if is_binary else "numeric",
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
        }
    return metadata


def hero_section(df: pd.DataFrame):
    total_records = len(df)
    positive_rate = df[LABEL_COLUMN].mean()
    monitored_sites = df["Elevation_m"].nunique() if "Elevation_m" in df else total_records

    hero = st.container()
    col_left, col_right = hero.columns([3, 2])
    with col_left:
        st.markdown("### Why this dashboard matters")
        st.markdown(
            """
            - **Detect** hazardous slopes early using IoT-fed soil & rainfall signals.
            - **Explain** every alert with transparent feature importance.
            - **Act** faster by simulating multiple “what-if” mitigation plans live.
            """
        )
        st.info(
            "Tip: Use the presets in the sidebar to instantly load common site conditions, "
            "then refine the numbers to match your scenario."
        )
    with col_right:
        st.markdown("### Live snapshot")
        col_a, col_b = st.columns(2)
        col_a.metric("Readings Logged", f"{total_records:,}")
        col_b.metric("Unique Sites", f"{monitored_sites:,}")
        st.progress(float(positive_rate))
        st.caption(f"Landslide frequency in dataset: {positive_rate:.1%}")


def initialize_state(feature_cols: List[str], metadata: Dict[str, Dict]):
    if SCENARIO_STATE_KEY not in st.session_state:
        st.session_state[SCENARIO_STATE_KEY] = []
    if INPUT_STATE_KEY not in st.session_state:
        st.session_state[INPUT_STATE_KEY] = {
            feature: metadata[feature]["median"] for feature in feature_cols
        }


def apply_preset(preset: Dict[str, float], metadata, feature_cols):
    new_defaults = st.session_state[INPUT_STATE_KEY].copy()
    for feature, value in preset.items():
        if feature in feature_cols:
            meta = metadata[feature]
            if meta["type"] == "binary":
                clamped = int(round(value))
            else:
                clamped = float(np.clip(value, meta["min"], meta["max"]))
            new_defaults[feature] = clamped
            widget_key = f"input_{feature}"
            st.session_state[widget_key] = clamped
    st.session_state[INPUT_STATE_KEY] = new_defaults
    st.toast("Preset applied. Scroll to the prediction tab to review inputs.")


def probability_gauge(probability: float):
    gauge_df = pd.DataFrame({"value": [probability]})
    chart = (
        alt.Chart(gauge_df)
        .mark_bar(cornerRadius=6)
        .encode(
            x=alt.X(
                "value:Q",
                scale=alt.Scale(domain=[0, 1]),
                title=None,
            ),
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(
                    domain=[0, 0.5, 1],
                    range=["#2ecc71", "#f1c40f", "#e74c3c"],
                ),
                legend=None,
            ),
        )
        .properties(height=40)
    )
    return chart


def roc_chart(fpr, tpr):
    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
    chart = (
        alt.Chart(roc_df)
        .mark_line()
        .encode(
            x=alt.X("FPR", title="False Positive Rate"),
            y=alt.Y("TPR", title="True Positive Rate"),
        )
        .properties(height=300)
    )
    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(strokeDash=[4, 4], color="#999999")
        .encode(x="x", y="y")
    )
    return chart + diagonal


def confusion_heatmap(conf_mat):
    df = pd.DataFrame(
        conf_mat,
        index=pd.Index(["Actual 0", "Actual 1"], name="Actual"),
        columns=pd.Index(["Pred 0", "Pred 1"], name="Predicted"),
    )
    df = df.reset_index().melt(id_vars="Actual", value_name="Count")
    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x="Predicted:N",
            y="Actual:N",
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="oranges")),
            text=alt.Text("Count:Q", format=",.0f"),
        )
        .mark_rect()
        .properties(height=200)
    )
    text = (
        alt.Chart(df)
        .mark_text(color="black")
        .encode(x="Predicted:N", y="Actual:N", text=alt.Text("Count:Q"))
    )
    return chart + text


def feature_importance_chart(importances: pd.Series, top_n: int = 15):
    top_features = (
        importances.rename("Importance")
        .reset_index()
        .rename(columns={"index": "Feature"})
        .head(top_n)
    )
    return (
        alt.Chart(top_features)
        .mark_bar()
        .encode(
            x=alt.X("Importance:Q"),
            y=alt.Y("Feature:N", sort="-x"),
            tooltip=["Feature", alt.Tooltip("Importance:Q", format=".4f")],
        )
        .properties(height=400)
    )


def compute_local_contributions(importances, metadata, inputs, top_k=5):
    contributions = []
    for feature, importance in importances.items():
        spread = metadata[feature]["max"] - metadata[feature]["min"] or 1
        normalized = (inputs[feature] - metadata[feature]["median"]) / spread
        score = importance * normalized
        contributions.append({"Feature": feature, "Contribution": score})
    contrib_df = (
        pd.DataFrame(contributions)
        .assign(abs_val=lambda d: d["Contribution"].abs())
        .sort_values("abs_val", ascending=False)
        .head(top_k)
    )
    contrib_df["Direction"] = contrib_df["Contribution"].apply(
        lambda x: "Risk ↑" if x >= 0 else "Risk ↓"
    )
    return contrib_df


def contribution_chart(contrib_df: pd.DataFrame):
    if contrib_df.empty:
        return None
    return (
        alt.Chart(contrib_df)
        .mark_bar()
        .encode(
            x=alt.X("Contribution:Q"),
            y=alt.Y("Feature:N", sort="-x"),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(domain=["Risk ↑", "Risk ↓"], range=["#e74c3c", "#2ecc71"]),
            ),
            tooltip=["Feature", alt.Tooltip("Contribution:Q", format=".4f")],
        )
        .properties(height=250)
    )


def scenario_history_table(history: List[Dict]):
    if not history:
        st.info("No scenarios saved yet. Submit a prediction to start comparing.")
        return
    df = pd.DataFrame(history)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download scenario log (.csv)",
        data=csv,
        file_name="scenario_history.csv",
        mime="text/csv",
        help="Share these simulations with your field team.",
    )

    if {"Slope_Angle", "Rainfall_mm"}.issubset(df.columns):
        scatter = (
            alt.Chart(df)
            .mark_circle(size=120)
            .encode(
                x=alt.X("Slope_Angle:Q"),
                y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Label:N", title="Risk Level"),
                tooltip=list(df.columns),
            )
            .properties(height=300)
        )
        st.altair_chart(scatter, use_container_width=True)


def what_if_analysis(model, feature_cols, base_inputs, feature_name, metadata):
    if metadata[feature_name]["type"] == "binary":
        values = np.array([0, 1])
    else:
        values = np.linspace(
            metadata[feature_name]["min"], metadata[feature_name]["max"], 25
        )
    data = []
    for value in values:
        row = base_inputs.copy()
        row[feature_name] = float(value)
        df_row = pd.DataFrame([row], columns=feature_cols)
        prob = float(model.predict_proba(df_row)[0][1])
        data.append({"Value": value, "Probability": prob})
    chart = (
        alt.Chart(pd.DataFrame(data))
        .mark_line(point=True)
        .encode(
            x=alt.X("Value:Q", title=feature_name),
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1])),
            tooltip=["Value", alt.Tooltip("Probability", format=".2%")],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def overview_tab(df: pd.DataFrame):
    st.subheader("Overview")
    total_records = len(df)
    positive_cases = int(df[LABEL_COLUMN].sum())
    positive_rate = positive_cases / total_records
    last_updated = datetime.fromtimestamp(Path(DATA_PATH).stat().st_mtime)

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Records", f"{total_records:,}")
    kpi_cols[1].metric("Positive Cases", f"{positive_cases:,}")
    kpi_cols[2].metric("Positive Rate", f"{positive_rate:.1%}")
    kpi_cols[3].metric("Dataset Updated", last_updated.strftime("%d %b %Y"))

    with st.expander("Peek at the latest data"):
        st.dataframe(df.head(100))

    st.markdown("#### Data Quality Snapshot")
    missing = (df.isna().sum() / len(df)).sort_values(ascending=False).head(10)
    miss_df = missing.reset_index().rename(
        columns={"index": "Feature", 0: "Missing_Ratio"}
    )
    miss_chart = (
        alt.Chart(miss_df)
        .mark_bar()
        .encode(
            x=alt.X("Missing_Ratio:Q", axis=alt.Axis(format="%")),
            y=alt.Y("Feature:N", sort="-x"),
        )
    )
    st.altair_chart(miss_chart, use_container_width=True)

    st.markdown("#### Key Feature Distributions")
    feature_samples = [
        col
        for col in [
            "Rainfall_mm",
            "Slope_Angle",
            "Soil_Saturation",
            "Pore_Water_Pressure_kPa",
        ]
        if col in df.columns
    ]
    chart_cols = st.columns(len(feature_samples))
    for chart_col, feature in zip(chart_cols, feature_samples):
        chart = (
            alt.Chart(df)
            .transform_density(
                feature,
                as_=[feature, "density"],
            )
            .mark_area()
            .encode(
                x=alt.X(f"{feature}:Q", title=feature),
                y="density:Q",
            )
        )
        chart_col.altair_chart(chart, use_container_width=True)


def insights_tab(model_bundle):
    st.subheader("Model Insights")
    metric_cols = st.columns(len(model_bundle["metrics"]))
    for col, (label, value) in zip(metric_cols, model_bundle["metrics"].items()):
        col.metric(label, f"{value:.3f}")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("##### ROC Curve")
        st.altair_chart(
            roc_chart(*model_bundle["roc"]),
            use_container_width=True,
        )
    with col_right:
        st.markdown("##### Confusion Matrix")
        st.altair_chart(
            confusion_heatmap(model_bundle["confusion"]),
            use_container_width=True,
        )

    st.markdown("##### Top Feature Importances")
    st.altair_chart(
        feature_importance_chart(model_bundle["importances"]),
        use_container_width=True,
    )


def prediction_tab(model_bundle, metadata):
    st.subheader("Manual Risk Assessment Playground")
    feature_cols = model_bundle["feature_cols"]
    model = model_bundle["model"]

    helper_cols = st.columns(3)
    helper_cols[0].markdown("**1. Configure Inputs**  \nEnter onsite readings or pick a preset.")
    helper_cols[1].markdown("**2. Review Probability**  \nGauge, KPI, and top drivers update instantly.")
    helper_cols[2].markdown("**3. Save Scenario**  \nEach submission is logged for comparison/export.")

    with st.expander("Need guidance on data entry?", expanded=False):
        st.write(
            "Use measurement devices or IoT feeds for accuracy. If a value is unknown, "
            "start from the suggested median and adjust once better information arrives."
        )

    with st.container():
        st.caption("Fill in site conditions. Use presets or tweak values manually.")
        cols = st.columns(3)
        inputs = {}
        with st.form("prediction_form"):
            for idx, feature in enumerate(feature_cols):
                meta = metadata[feature]
                default_value = st.session_state[INPUT_STATE_KEY].get(
                    feature, meta["median"]
                )
                column = cols[idx % len(cols)]
                widget_key = f"input_{feature}"
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = default_value
                feature_help = GLOSSARY.get(feature)
                if meta["type"] == "binary":
                    options = [0, 1]
                    current_value = int(round(st.session_state[widget_key]))
                    current_index = options.index(current_value) if current_value in options else 0
                    inputs[feature] = column.selectbox(
                        feature,
                        options=options,
                        index=current_index,
                        help=(
                            f"{feature_help} | " if feature_help else ""
                        )
                        + "0 = No / False, 1 = Yes / True",
                        key=widget_key,
                    )
                else:
                    step = (meta["max"] - meta["min"]) / 500 or 0.001
                    inputs[feature] = column.number_input(
                        feature,
                        min_value=meta["min"],
                        max_value=meta["max"],
                        value=float(st.session_state[widget_key]),
                        step=step,
                        help=feature_help,
                        key=widget_key,
                    )

            submitted = st.form_submit_button("Predict Landslide Probability")

    if submitted:
        st.session_state[INPUT_STATE_KEY] = inputs.copy()
        input_df = pd.DataFrame([inputs], columns=feature_cols)
        probability = float(model.predict_proba(input_df)[0][1])
        prediction = "High Risk" if probability >= 0.5 else "Low Risk"

        st.markdown("#### Prediction Summary")
        summary_cols = st.columns([2, 3])
        summary_cols[0].altair_chart(
            probability_gauge(probability), use_container_width=True
        )
        summary_cols[1].metric(
            "Probability of Landslide",
            f"{probability:.2%}",
            help="Probability threshold 0.5",
        )

        if prediction == "High Risk":
            st.error("High risk detected. Initiate mitigation immediately.")
        else:
            st.success("Low risk detected. Continue monitoring.")

        contrib_df = compute_local_contributions(
            model_bundle["importances"], metadata, inputs
        )
        st.markdown("##### Leading Factors")
        chart = contribution_chart(contrib_df)
        if chart:
            st.altair_chart(chart, use_container_width=True)

        scenario_record = {
            "Timestamp": datetime.now().strftime("%H:%M:%S"),
            "Probability": round(probability, 3),
            "Label": prediction,
        }
        for key in ["Rainfall_mm", "Slope_Angle", "Soil_Saturation"]:
            if key in inputs:
                scenario_record[key] = inputs[key]
        st.session_state[SCENARIO_STATE_KEY].append(scenario_record)

        st.markdown("##### What-if Analysis")
        feature_choice = st.selectbox(
            "Select a feature to stress-test",
            options=feature_cols,
        )
        what_if_analysis(model, feature_cols, inputs, feature_choice, metadata)

    st.markdown("#### Scenario History")
    scenario_history_table(st.session_state[SCENARIO_STATE_KEY])


def sidebar_content(metadata):
    st.sidebar.header("Guided Workflow")
    st.sidebar.write(
        """
        1. Review data quality and KPIs on the **Overview** tab.
        2. Study model behaviour inside **Insights**.
        3. Simulate scenarios in **Prediction** and save them for comparison.
        """
    )

    st.sidebar.subheader("Quick Presets")
    for name, preset in PRESET_SCENARIOS.items():
        if st.sidebar.button(name):
            apply_preset(preset, metadata, list(metadata.keys()))

    st.sidebar.subheader("Critical Sensors At a Glance")
    for feature in CORE_FEATURES:
        if feature in GLOSSARY:
            st.sidebar.markdown(f"- **{feature}** – {GLOSSARY[feature]}")

    secondary_glossary = {
        feature: desc for feature, desc in GLOSSARY.items() if feature not in CORE_FEATURES
    }

    with st.sidebar.expander("Full Feature Glossary"):
        for feature, description in secondary_glossary.items():
            st.markdown(f"**{feature}** – {description}")


def main():
    df = load_dataset(DATA_PATH)
    model_bundle = train_model(df)
    metadata = build_feature_metadata(df)
    initialize_state(model_bundle["feature_cols"], metadata)

    sidebar_content(metadata)

    st.title(f"🌏 {PROJECT_TITLE}")
    st.caption(
        "Interactive IoT-powered command center for monitoring, understanding, and forecasting "
        "landslide risk using high-resolution soil and weather intelligence."
    )
    hero_section(df)

    tab_overview, tab_insights, tab_prediction = st.tabs(
        ["Overview", "Model Insights", "Prediction Studio"]
    )

    with tab_overview:
        overview_tab(df)
    with tab_insights:
        insights_tab(model_bundle)
    with tab_prediction:
        prediction_tab(model_bundle, metadata)


if __name__ == "__main__":
    main()

