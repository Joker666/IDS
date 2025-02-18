import numpy as np
import pandas as pd
import plotly.express as px


def load_data(dataset="train"):
    """Load and preprocess the dataset"""
    if dataset == "train":
        df = pd.read_csv("./Train_data.csv")
    else:
        df = pd.read_csv("./Test_data.csv")
    return df


def get_basic_stats(df):
    """Calculate basic dataset statistics"""
    stats = {
        "Total Records": len(df),
        "Normal Traffic": len(df[df["class"] == "normal"]),
        "Anomaly Traffic": len(df[df["class"] == "anomaly"]),
        "Number of Features": len(df.columns),
        "Protocol Types": df["protocol_type"].nunique(),
        "Services": df["service"].nunique(),
    }
    return stats


def create_traffic_distribution(df):
    """Create traffic distribution plot"""
    traffic_dist = df["class"].value_counts()
    fig = px.pie(
        values=traffic_dist.values,
        names=traffic_dist.index,
        title="Distribution of Normal vs Anomaly Traffic",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    return fig


def create_protocol_analysis(df):
    """Analyze protocol types"""
    protocol_class = pd.crosstab(df["protocol_type"], df["class"])
    fig = px.bar(
        protocol_class,
        title="Protocol Distribution by Traffic Class",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    return fig


def create_service_analysis(df):
    """Analyze top services by traffic class"""
    # Get top 10 services
    top_10_services = df["service"].value_counts().head(10).index

    # Filter df and keep only needed columns
    long_df = df[df["service"].isin(top_10_services)][["service", "class"]]

    fig = px.histogram(
        long_df,
        x="service",
        color="class",
        title="Top 10 Services Distribution by Traffic Class",
        labels={"service": "Service", "count": "Count"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    return fig


def create_correlation_heatmap(df):
    """Create correlation heatmap for numeric features"""
    df_dropped = df.drop(["num_outbound_cmds", "is_host_login"], axis=1)
    numeric_cols = df_dropped.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]
    corr = df_numeric.corr()

    fig = px.imshow(
        corr,
        title="Feature Correlation Heatmap",
        color_continuous_scale="RdBu",
        aspect="auto",
    )
    return fig


def create_feature_distributions(df):
    """Create distribution plots for key numeric features"""
    key_features = ["duration", "src_bytes", "dst_bytes", "count", "srv_count"]
    figs = []

    for feature in key_features:
        fig = px.histogram(
            df,
            x=feature,
            color="class",
            title=f"Distribution of {feature}",
            marginal="box",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        figs.append(fig)

    return figs


def create_bytes_duration_plots(df):
    """Create scatterplots comparing duration with src_bytes and dst_bytes"""
    # Melt the dataframe to create long format for src and dst bytes
    bytes_df = pd.melt(
        df,
        value_vars=["src_bytes", "dst_bytes"],
        id_vars=["duration", "class"],
        var_name="byte_type",
        value_name="bytes",
    )

    fig = px.scatter(
        bytes_df,
        x="duration",
        y="bytes",
        color="class",
        facet_col="byte_type",
        labels={"duration": "Duration (s)", "bytes": "Bytes", "class": "Traffic Class"},
        title="Network Traffic Duration vs Bytes",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    # Clean up facet labels
    fig.for_each_annotation(
        lambda a: a.update(text=a.text.split("=")[1].replace("_", " ").title())
    )

    return fig


def create_error_rate_distributions(df):
    """Create box plots for error rate"""
    key_features = ["serror_rate", "rerror_rate", "srv_serror_rate", "srv_rerror_rate"]
    figs = []

    for feature in key_features:
        fig = px.box(
            df,
            x="class",
            y=feature,
            title=f"Distribution of {feature}",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        figs.append(fig)

    return figs


def get_data_quality_report(df):
    """Generate data quality report"""
    quality_report = {
        "Missing Values": df.isnull().sum().sum(),
        "Duplicate Rows": df.duplicated().sum(),
        "Numeric Features": len(df.select_dtypes(include=[np.number]).columns),
        "Categorical Features": len(df.select_dtypes(include=["object"]).columns),
    }
    return quality_report


def get_data_description():
    """Load content from description.md file"""
    with open("description.md", "r") as f:
        description = f.read()
    return description
