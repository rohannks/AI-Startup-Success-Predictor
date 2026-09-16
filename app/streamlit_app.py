# Import required libraries

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# Configure Streamlit page

st.set_page_config(
    page_title="AI Startup Success Predictor",
    page_icon="🚀",
    layout="wide"
)


# Project paths

BASE_PATH = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_PATH
    / "models"
    / "startup_success_model.pkl"
)

DATA_PATH = (
    BASE_PATH
    / "data"
    / "processed"
    / "integrated_startup_data.csv"
)

EVALUATION_PATH = (
    BASE_PATH
    / "reports"
    / "model_evaluation.csv"
)

COMPARISON_PATH = (
    BASE_PATH
    / "reports"
    / "model_comparison.csv"
)


# Custom styling

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# Load trained model

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


# Load startup data

@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


# Load evaluation results

@st.cache_data
def load_evaluation():

    if EVALUATION_PATH.exists():
        return pd.read_csv(EVALUATION_PATH)

    return pd.DataFrame()


# Load model comparison

@st.cache_data
def load_comparison():

    if COMPARISON_PATH.exists():
        return pd.read_csv(COMPARISON_PATH)

    return pd.DataFrame()


# Load project resources

model = load_model()

df = load_data()

evaluation_df = load_evaluation()

comparison_df = load_comparison()


# Application header

st.markdown(
    '<div class="main-title">🚀 AI Startup Success Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning based startup analytics and unicorn prediction'
    '</div>',
    unsafe_allow_html=True
)


# Sidebar navigation

st.sidebar.title("🚀 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📊 Startup Analytics",
        "💰 Funding Analysis",
        "🦄 Unicorn Analysis",
        "🤖 Success Prediction",
        "📈 Model Performance",
        "📋 Dataset Explorer",
        "ℹ️ About Project"
    ]
)



# Home page

if page == "🏠 Home":

    st.header("🏠 Project Overview")

    st.write(
        "AI Startup Success Predictor analyzes startup data, "
        "funding, valuation, sectors and other business factors "
        "to estimate potential unicorn status."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    total_startups = len(df)

    unicorn_count = int(
        df["is_unicorn"]
        .eq(True)
        .sum()
    )

    country_count = df["country"].nunique()

    sector_count = df["sector"].nunique()

    col1.metric(
        "Total Startups",
        f"{total_startups:,}"
    )

    col2.metric(
        "Unicorn Records",
        f"{unicorn_count:,}"
    )

    col3.metric(
        "Countries",
        f"{country_count:,}"
    )

    col4.metric(
        "Sectors",
        f"{sector_count:,}"
    )

    st.divider()

    st.subheader("📊 Data Sources")

    source_counts = (
        df["data_source"]
        .value_counts()
    )

    st.bar_chart(source_counts)

    st.subheader("🔍 Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )


# Startup Analytics

elif page == "📊 Startup Analytics":

    st.header("📊 Startup Analytics")

    st.write(
        "Explore startup distribution across sectors, countries "
        "and startup stages."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Startups by Sector")

        sector_counts = (
            df["sector"]
            .value_counts()
            .head(15)
        )

        st.bar_chart(
            sector_counts
        )

    with col2:

        st.subheader("Startups by Country")

        country_counts = (
            df["country"]
            .value_counts()
            .head(15)
        )

        st.bar_chart(
            country_counts
        )

    st.subheader("Startups by Stage")

    stage_counts = (
        df["startup_stage"]
        .value_counts()
    )

    st.bar_chart(
        stage_counts
    )

    st.subheader("Startups by Data Source")

    st.bar_chart(
        df["data_source"].value_counts()
    )


# Funding Analysis

elif page == "💰 Funding Analysis":

    st.header("💰 Funding Analysis")

    funding_data = df[
        df["funding_usd_millions"].notna()
    ].copy()

    col1, col2, col3 = st.columns(3)

    average_funding = (
        funding_data["funding_usd_millions"]
        .mean()
    )

    maximum_funding = (
        funding_data["funding_usd_millions"]
        .max()
    )

    col1.metric(
        "Funding Records",
        len(funding_data)
    )

    col2.metric(
        "Average Funding",
        f"${average_funding:,.2f}M"
    )

    col3.metric(
        "Maximum Funding",
        f"${maximum_funding:,.2f}M"
    )

    st.divider()

    st.subheader("Funding Distribution")

    funding_chart = (
        funding_data[
            "funding_usd_millions"
        ]
        .dropna()
        .sort_values(ascending=False)
        .head(20)
    )

    st.bar_chart(
        funding_chart
    )

    st.subheader("🏆 Top Funded Startups")

    top_funded = (
        funding_data[
            [
                "company",
                "funding_usd_millions",
                "sector",
                "country"
            ]
        ]
        .sort_values(
            "funding_usd_millions",
            ascending=False
        )
        .head(15)
    )

    st.dataframe(
        top_funded,
        use_container_width=True,
        hide_index=True
    )


# Unicorn Analysis

elif page == "🦄 Unicorn Analysis":

    st.header("🦄 Unicorn Analysis")

    unicorn_data = df[
        df["is_unicorn"] == True
    ].copy()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Unicorn Records",
        len(unicorn_data)
    )

    col2.metric(
        "Unicorn Countries",
        unicorn_data["country"].nunique()
    )

    col3.metric(
        "Unicorn Sectors",
        unicorn_data["sector"].nunique()
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Unicorns by Country")

        unicorn_country = (
            unicorn_data["country"]
            .value_counts()
            .head(15)
        )

        st.bar_chart(
            unicorn_country
        )

    with col2:

        st.subheader("Unicorns by Sector")

        unicorn_sector = (
            unicorn_data["sector"]
            .value_counts()
            .head(15)
        )

        st.bar_chart(
            unicorn_sector
        )

    st.subheader("🦄 Unicorn Companies")

    unicorn_table = unicorn_data[
        [
            "company",
            "sector",
            "country",
            "city",
            "startup_stage",
            "valuation_usd_millions"
        ]
    ].head(50)

    st.dataframe(
        unicorn_table,
        use_container_width=True,
        hide_index=True
    )


# Success Prediction

elif page == "🤖 Success Prediction":

    st.header("🤖 Startup Success Prediction")

    st.write(
        "Enter startup information and estimate the probability "
        "of unicorn status."
    )

    st.warning(
        "Prototype model: trained on a small labeled dataset. "
        "The prediction is an estimate, not a guaranteed outcome."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        founded_year = st.number_input(
            "Founded Year",
            min_value=1950,
            max_value=2026,
            value=2020,
            step=1
        )

        sector = st.text_input(
            "Sector",
            value="HealthTech"
        )

        subsector = st.text_input(
            "Subsector",
            value="AI"
        )

        country = st.text_input(
            "Country",
            value="India"
        )

        city = st.text_input(
            "City",
            value="Bengaluru"
        )

        startup_stage = st.selectbox(
            "Startup Stage",
            [
                "Early Stage",
                "Growth Stage",
                "Late Stage",
                "Seed",
                "Series A",
                "Series B",
                "Series C",
                "Series D",
                "Series E",
                "Series F"
            ]
        )

    with col2:

        funding = st.number_input(
            "Total Funding (USD Millions)",
            min_value=0.0,
            value=10.0,
            step=1.0
        )

        valuation = st.number_input(
            "Valuation (USD Millions)",
            min_value=0.0,
            value=100.0,
            step=10.0
        )

        employees = st.number_input(
            "Employees",
            min_value=1.0,
            value=100.0,
            step=1.0
        )

        startup_age = max(
            2026 - founded_year,
            0
        )

        funding_log = np.log1p(
            funding
        )

        valuation_log = np.log1p(
            valuation
        )

        funding_per_employee = (
            funding / employees
            if employees > 0
            else 0
        )

        st.info(
            f"Startup Age: {startup_age} years"
        )

    st.divider()

    predict_button = st.button(
        "🔮 Predict Startup Success",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        input_data = pd.DataFrame({

            "founded_year": [
                founded_year
            ],

            "sector": [
                sector
            ],

            "subsector": [
                subsector
            ],

            "country": [
                country
            ],

            "city": [
                city
            ],

            "funding_usd_millions": [
                funding
            ],

            "valuation_usd_millions": [
                valuation
            ],

            "employees": [
                employees
            ],

            "startup_stage": [
                startup_stage
            ],

            "startup_age": [
                startup_age
            ],

            "funding_log": [
                funding_log
            ],

            "valuation_log": [
                valuation_log
            ],

            "funding_per_employee": [
                funding_per_employee
            ]
        })

        prediction = model.predict(
            input_data
        )[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

        st.divider()

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if prediction == 1:

                st.success(
                    "🦄 Predicted Status: Potential Unicorn"
                )

            else:

                st.info(
                    "📊 Predicted Status: Non-Unicorn"
                )

        with result_col2:

            st.metric(
                "Unicorn Probability",
                f"{probability * 100:.2f}%"
            )

        st.progress(
            float(probability)
        )

        if probability >= 0.75:

            st.success(
                "High predicted unicorn probability."
            )

        elif probability >= 0.50:

            st.warning(
                "Moderate predicted unicorn probability."
            )

        else:

            st.info(
                "Low predicted unicorn probability."
            )

        st.subheader("📋 Input Summary")

        input_summary = pd.DataFrame({
            "Feature": [
                "Founded Year",
                "Sector",
                "Subsector",
                "Country",
                "City",
                "Funding",
                "Valuation",
                "Employees",
                "Startup Stage"
            ],
            "Value": [
                founded_year,
                sector,
                subsector,
                country,
                city,
                f"${funding:.2f}M",
                f"${valuation:.2f}M",
                employees,
                startup_stage
            ]
        })

        st.dataframe(
            input_summary,
            use_container_width=True,
            hide_index=True
        )


# Model Performance

elif page == "📈 Model Performance":

    st.header("📈 Model Performance")

    st.write(
        "Performance of the trained startup success prediction model."
    )

    st.divider()

    if not evaluation_df.empty:

        evaluation_values = dict(
            zip(
                evaluation_df["Metric"],
                evaluation_df["Score"]
            )
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric(
            "Accuracy",
            f"{evaluation_values.get('Accuracy', 0):.3f}"
        )

        col2.metric(
            "Precision",
            f"{evaluation_values.get('Precision', 0):.3f}"
        )

        col3.metric(
            "Recall",
            f"{evaluation_values.get('Recall', 0):.3f}"
        )

        col4.metric(
            "F1 Score",
            f"{evaluation_values.get('F1 Score', 0):.3f}"
        )

        col5.metric(
            "ROC-AUC",
            f"{evaluation_values.get('ROC-AUC', 0):.3f}"
        )

        st.subheader("Evaluation Metrics")

        st.bar_chart(
            evaluation_df.set_index("Metric")
        )

        st.dataframe(
            evaluation_df,
            use_container_width=True,
            hide_index=True
        )

    if not comparison_df.empty:

        st.subheader("🤖 Model Comparison")

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

    st.info(
        "The model achieved approximately 0.95 ROC-AUC on the held-out "
        "test set. Because the labeled dataset is small, this should be "
        "treated as a prototype result."
    )


# Dataset Explorer

elif page == "📋 Dataset Explorer":

    st.header("📋 Dataset Explorer")

    st.write(
        "Explore the integrated startup dataset."
    )

    search = st.text_input(
        "🔍 Search company"
    )

    if search:

        filtered_df = df[
            df["company"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]

    else:

        filtered_df = df

    st.write(
        f"Showing {len(filtered_df):,} records"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


# About Project

elif page == "ℹ️ About Project":

    st.header("ℹ️ About the Project")

    st.write(
        """
        AI Startup Success Predictor is a Machine Learning project
        designed to analyze startup characteristics and estimate
        potential unicorn status.

        The project combines startup datasets with data cleaning,
        integration, exploratory analysis, feature engineering,
        machine learning and an interactive Streamlit dashboard.
        """
    )

    st.divider()

    st.subheader("🔄 Project Pipeline")

    st.code(
        """
Raw Datasets
     ↓
Data Cleaning
     ↓
Dataset Integration
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Streamlit Dashboard
        """
    )

    st.subheader("🧠 Machine Learning")

    st.write(
        """
        Models evaluated in the project include:

        • Logistic Regression
        • Random Forest
        • Gradient Boosting

        The best-performing model was saved and integrated
        into the Streamlit prediction application.
        """
    )

    st.subheader("⚠️ Important Limitation")

    st.warning(
        "The current labeled modeling dataset contains only 112 records "
        "with 11 positive unicorn examples. Therefore, the model is a "
        "research/academic prototype and should not be considered a "
        "production investment or business decision system."
    )

    st.subheader("📊 Current Model Results")

    st.write(
        "Accuracy: 91.3%"
    )

    st.write(
        "Precision: 50.0%"
    )

    st.write(
        "Recall: 100.0%"
    )

    st.write(
        "F1 Score: 66.7%"
    )

    st.write(
        "ROC-AUC: 95.2%"
    )

    st.success(
        "🚀 AI Startup Success Predictor — Project Dashboard"
    )