import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml_utils import load_model, preprocess_data, save_model, train_model
from utils import (
    create_bytes_duration_plots,
    create_correlation_heatmap,
    create_error_rate_distributions,
    create_feature_distributions,
    create_protocol_analysis,
    create_service_analysis,
    create_traffic_distribution,
    get_basic_stats,
    get_data_description,
    get_data_quality_report,
    load_data,
)

# Page configuration
st.set_page_config(
    page_title="Network Intrusion Detection", page_icon="🔒", layout="wide"
)

# Title and introduction
st.title("Network Intrusion Detection System")
st.markdown("""
This dashboard provides data analysis and predictive modeling capabilities
for the Network Intrusion Detection dataset, helping to understand patterns in network
traffic and predict potential security threats.
""")


# Load data
@st.cache_data
def load_cached_data(dataset="train"):
    return load_data(dataset)


df = load_cached_data("train")
df_test = load_cached_data("test")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a section:",
    [
        "Overview",
        "Data Quality",
        "Traffic Analysis",
        "Protocol & Services",
        "Feature Analysis",
        "Predictive Analysis",
    ],
)

# Overview page
if page == "Overview":
    st.header("Dataset Overview")

    st.markdown("""
    The Network Intrusion Detection dataset contains network traffic data simulated in a military network environment.
    It simulates a typical US Air Force LAN under attack. The goal is to classify the traffic as either normal or anomalous,
    which can be useful for identifying potential security threats. The dataset is based on various features such as protocol,
    service, flag, and more. The dataset is useful for training machine learning models to detect potential security
    threats in network traffic.

    Characteristics of the dataset:

    * Sequence of TCP packets within a time duration.
    * Data flows between a source IP and a target IP under a defined protocol.
    * Each connection is labeled as normal or an attack.
    """)

    # Display basic stats
    stats = get_basic_stats(df)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", stats["Total Records"])
        st.metric("Protocol Types", stats["Protocol Types"])

    with col2:
        st.metric("Normal Traffic", stats["Normal Traffic"])
        st.metric("Services", stats["Services"])

    with col3:
        st.metric("Anomaly Traffic", stats["Anomaly Traffic"])
        st.metric("Features", stats["Number of Features"])

    # Sample data
    st.subheader("Sample Data")
    st.dataframe(df.head())

    # Data description
    st.subheader("Feature Description")
    st.markdown(get_data_description())

# Data Quality page
elif page == "Data Quality":
    st.header("Data Quality Report")

    quality_report = get_data_quality_report(df)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Missing Values", quality_report["Missing Values"])
        st.metric("Numeric Features", quality_report["Numeric Features"])

    with col2:
        st.metric("Duplicate Rows", quality_report["Duplicate Rows"])
        st.metric("Categorical Features", quality_report["Categorical Features"])

    # Data types information
    st.subheader("Data Types Information")
    st.dataframe(df.dtypes.astype(str))

    # Data statistics
    st.subheader("Data Statistics")
    st.dataframe(df.describe())

# Traffic Analysis page
elif page == "Traffic Analysis":
    st.header("Traffic Analysis")

    # Traffic distribution
    st.plotly_chart(create_traffic_distribution(df), use_container_width=True)

    # Traffic patterns over features
    st.subheader("Traffic Patterns")
    feature_dists = create_feature_distributions(df)
    for fig in feature_dists:
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
        - All traffic features exhibit highly skewed distributions, with most values concentrated near the lower ranges and occasional large outliers.
        - There are significant outliers in both src_bytes and dst_bytes.
        - The distributions for anomaly and normal classes differ, suggesting potential discriminatory power for classification.
    """)

    # Bytes vs Duration plots
    st.subheader("Bytes vs Duration")
    st.plotly_chart(create_bytes_duration_plots(df), use_container_width=True)

    st.markdown("""
        - Anomalies often exhibit larger values for destination bytes
        - Anomalies tend to have longer durations and higher source byte values, forming separable patterns from normal traffic.
    """)

    # Error rate distributions
    st.subheader("Error Rate Distributions")
    error_dists = create_error_rate_distributions(df)
    for fig in error_dists:
        st.plotly_chart(fig, use_container_width=True)

# Protocol & Services page
elif page == "Protocol & Services":
    st.header("Protocol and Services Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(create_protocol_analysis(df), use_container_width=True)

    with col2:
        st.plotly_chart(create_service_analysis(df), use_container_width=True)

    st.markdown("""
        - The majority of records are associated with a few dominant protocol types, such as tcp, udp, and icmp.
        - The service column shows a wide variety of services, with a few services like http, private, and domain_u being much more common than others.
    """)

    # Additional protocol details
    st.subheader("Protocol Details")
    st.dataframe(df.groupby("protocol_type")["class"].value_counts().unstack())

# Feature Analysis page
elif page == "Feature Analysis":
    st.header("Feature Analysis")

    # Correlation heatmap
    st.subheader("Feature Correlations")
    st.plotly_chart(create_correlation_heatmap(df), use_container_width=True)

    st.markdown("""
        - Strong positive and negative correlations are observed between several features.
        - Features like dst_host_same_srv_rate, dst_host_srv_serror_rate, and dst_host_count exhibit patterns that may be useful for model-building.
    """)

    # Feature statistics
    st.subheader("Feature Statistics")
    st.dataframe(df.describe())

# Predictive Analysis page
else:
    st.header("Predictive Analysis")

    # Model training section
    st.subheader("Model Training")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Set Size", len(df))
    with col2:
        st.metric("Test Set Size", len(df_test))

    if st.button("Train New Model"):
        with st.spinner("Training model..."):
            # Preprocess training data
            X_scaled, y, label_encoders, scaler = preprocess_data(df)

            # Train and evaluate model
            model, report, conf_matrix = train_model(X_scaled, y)

            # Save model
            save_model(model, label_encoders, scaler)

            # Display metrics
            st.success("Model trained successfully!")

            # Show classification report
            st.subheader("Model Performance")

            # Convert classification report to dataframe for better display
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

            # Display confusion matrix
            st.subheader("Confusion Matrix")
            fig = go.Figure(
                data=go.Heatmap(
                    z=conf_matrix,
                    x=["Predicted Normal", "Predicted Anomaly"],
                    y=["Actual Normal", "Actual Anomaly"],
                    colorscale="RdBu",
                )
            )
            st.plotly_chart(fig)

    # Prediction section for test data
    st.subheader("Make Predictions on Test Data")

    try:
        model, label_encoders, scaler = load_model()

        if st.button("Run Predictions on Test Data"):
            with st.spinner("Making predictions on test data..."):
                # Preprocess test data using existing encoders and scaler
                X_test_scaled, _, _, _ = preprocess_data(
                    df_test,
                    is_training=False,
                    existing_encoders=label_encoders,
                    existing_scaler=scaler,
                )

                # Make predictions
                predictions = model.predict(X_test_scaled)
                probabilities = model.predict_proba(X_test_scaled)

                # Add predictions to df_test
                df_test["predicted_class"] = [
                    "Normal" if p == 0 else "Anomaly" for p in predictions
                ]
                df_test["prediction_confidence"] = [max(prob) for prob in probabilities]

                # Display results
                st.write("#### Test Data Predictions")

                # Show sample of predictions
                results_df = pd.DataFrame(
                    {
                        "Predicted Class": [
                            "Normal" if p == 0 else "Anomaly" for p in predictions
                        ],
                        "Confidence": [max(prob) for prob in probabilities],
                    }
                )

                st.dataframe(df_test.head(10))

                # Show distribution of predictions
                fig = px.pie(
                    values=[sum(predictions == 0), sum(predictions == 1)],
                    names=["Normal", "Anomaly"],
                    title="Distribution of Predictions in Test Data",
                )
                st.plotly_chart(fig)

    except FileNotFoundError:
        st.warning("Please train a model first using the button above.")
