"""
Utility module for visualizations in the CA Prediction System
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import streamlit as st
import io
import base64
import sys
import os

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VIZ_CONFIG

def plot_attendance_distribution(df, column="attendance_percentage", by=None):
    """
    Plot distribution of attendance percentages
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column to plot
        by (str, optional): Column to group by
        
    Returns:
        go.Figure: Plotly figure object
    """
    if by and by in df.columns:
        fig = px.histogram(
            df, 
            x=column, 
            color=by,
            marginal="box",
            opacity=0.7,
            barmode="overlay",
            color_discrete_sequence=[VIZ_CONFIG["color_palette"]["primary"], VIZ_CONFIG["color_palette"]["ca_color"]],
            title=f"Distribution of {column.replace('_', ' ').title()} by {by.replace('_', ' ').title()}"
        )
    else:
        fig = px.histogram(
            df, 
            x=column,
            marginal="box",
            opacity=0.7,
            color_discrete_sequence=[VIZ_CONFIG["color_palette"]["primary"]],
            title=f"Distribution of {column.replace('_', ' ').title()}"
        )
    
    fig.update_layout(
        xaxis_title=column.replace('_', ' ').title(),
        yaxis_title="Count",
        legend_title=by.replace('_', ' ').title() if by else None,
        template="plotly_white"
    )
    
    # Add a vertical line at CA threshold if plotting attendance percentage
    if column == "attendance_percentage":
        fig.add_vline(
            x=90, 
            line_dash="dash", 
            line_color="red",
            annotation_text="CA Threshold (90%)",
            annotation_position="top right"
        )
    
    return fig

def plot_confusion_matrix(confusion_matrix_values, labels=["No-CA", "CA"]):
    """
    Plot confusion matrix
    
    Args:
        confusion_matrix_values (array): Confusion matrix values
        labels (list): Class labels
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Create annotation text
    annotations = []
    for i, row in enumerate(confusion_matrix_values):
        for j, value in enumerate(row):
            annotations.append(
                {
                    "x": j,
                    "y": i,
                    "xref": "x",
                    "yref": "y",
                    "text": str(value),
                    "showarrow": False,
                    "font": {"color": "white" if value > np.max(confusion_matrix_values) / 2 else "black"}
                }
            )
    
    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=confusion_matrix_values,
        x=labels,
        y=labels,
        colorscale="Blues",
        showscale=False
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis=dict(title="Predicted Label"),
        yaxis=dict(title="True Label"),
        annotations=annotations,
        template="plotly_white"
    )
    
    return fig

def plot_feature_importance(feature_importance, top_n=10):
    """
    Plot feature importance
    
    Args:
        feature_importance (dict): Dictionary mapping feature names to importance scores
        top_n (int): Number of top features to show
        
    Returns:
        go.Figure: Plotly figure object
    """
    if not feature_importance:
        return None
    
    # Sort by importance and take top N
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    # Create dataframe for plotting
    feature_df = pd.DataFrame(top_features, columns=["Feature", "Importance"])
    
    # Clean feature names
    feature_df["Feature"] = feature_df["Feature"].str.replace("_", " ").str.title()
    
    # Create figure
    fig = px.bar(
        feature_df,
        y="Feature",
        x="Importance",
        orientation="h",
        color="Importance",
        color_continuous_scale="Viridis",
        title=f"Top {top_n} Feature Importance"
    )
    
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
        template="plotly_white"
    )
    
    return fig

def plot_roc_curve(fpr, tpr, roc_auc):
    """
    Plot ROC curve
    
    Args:
        fpr (array): False positive rates
        tpr (array): True positive rates
        roc_auc (float): Area under the ROC curve
        
    Returns:
        go.Figure: Plotly figure object
    """
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr, 
            y=tpr,
            mode="lines",
            name=f"ROC Curve (AUC = {roc_auc:.3f})",
            line=dict(color=VIZ_CONFIG["color_palette"]["primary"], width=2)
        )
    )
    
    # Add diagonal line (random classifier)
    fig.add_trace(
        go.Scatter(
            x=[0, 1], 
            y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(color="gray", width=2, dash="dash")
        )
    )
    
    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curve",
        xaxis=dict(title="False Positive Rate"),
        yaxis=dict(title="True Positive Rate"),
        template="plotly_white",
        legend=dict(x=0.01, y=0.99, bordercolor="gray", borderwidth=1)
    )
    
    return fig

def plot_metric_comparison(models_metrics, metric="f1"):
    """
    Plot comparison of a specific metric across different models
    
    Args:
        models_metrics (dict): Dictionary mapping model names to metrics
        metric (str): Metric to compare
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Extract model names and metric values
    model_names = list(models_metrics.keys())
    metric_values = [metrics.get(metric, 0) for metrics in models_metrics.values()]
    
    # Create a dataframe for plotting
    df = pd.DataFrame({
        "Model": model_names,
        metric.title(): metric_values
    })
    
    # Sort by metric value (descending)
    df = df.sort_values(by=metric.title(), ascending=False)
    
    # Create figure
    fig = px.bar(
        df,
        x="Model",
        y=metric.title(),
        color=metric.title(),
        color_continuous_scale="Viridis",
        title=f"Model Comparison by {metric.title()}"
    )
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title=metric.title(),
        template="plotly_white"
    )
    
    # Add a horizontal line at the maximum metric value
    max_value = df[metric.title()].max()
    fig.add_hline(
        y=max_value,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Best: {max_value:.3f}",
        annotation_position="top right"
    )
    
    return fig

def plot_bubble_chart(df, x="academic_performance", y="attendance_percentage", size="absent_days", color="ca_status"):
    """
    Create a bubble chart visualization
    
    Args:
        df (pd.DataFrame): Input dataframe
        x (str): Column for x-axis
        y (str): Column for y-axis
        size (str): Column for bubble size
        color (str): Column for bubble color
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Create a copy of the dataframe
    plot_df = df.copy()
    
    # Convert categorical values to binary (for color mapping)
    if color in plot_df.columns and plot_df[color].dtype == 'object':
        if set(plot_df[color].unique()) == {"CA", "No-CA"}:
            # Create a new column for label
            plot_df["color_value"] = (plot_df[color] == "CA").astype(int)
            color_discrete_map = {0: VIZ_CONFIG["color_palette"]["no_ca_color"], 1: VIZ_CONFIG["color_palette"]["ca_color"]}
        else:
            # Use the original column
            plot_df["color_value"] = plot_df[color]
            color_discrete_map = None
    else:
        # Use the original column
        plot_df["color_value"] = plot_df[color]
        color_discrete_map = None
    
    # Create hover text
    plot_df["hover_text"] = plot_df.apply(
        lambda row: "<br>".join([
            f"Student ID: {row.get('student_id', 'N/A')}",
            f"School: {row.get('school', 'N/A')}",
            f"Grade: {row.get('grade', 'N/A')}",
            f"Gender: {row.get('gender', 'N/A')}",
            f"Academic Performance: {row.get('academic_performance', 'N/A'):.1f}%",
            f"Attendance: {row.get('attendance_percentage', 'N/A'):.1f}%",
            f"Present Days: {row.get('present_days', 'N/A')}",
            f"Absent Days: {row.get('absent_days', 'N/A')}",
            f"CA Status: {row.get('ca_status', 'N/A')}"
        ]),
        axis=1
    )
    
    # Create figure
    fig = px.scatter(
        plot_df,
        x=x,
        y=y,
        size=size,
        color="color_value",
        color_discrete_map=color_discrete_map,
        hover_name=plot_df.get("student_id", None),
        hover_data={
            x: True,
            y: True,
            size: True,
            "color_value": False,
            "hover_text": True
        },
        title=f"Relationship between {x.replace('_', ' ').title()} and {y.replace('_', ' ').title()}",
        size_max=VIZ_CONFIG["bubble_plot"]["max_size"],
        opacity=VIZ_CONFIG["bubble_plot"]["alpha"]
    )
    
    # Clean up axis labels
    fig.update_layout(
        xaxis_title=x.replace('_', ' ').title(),
        yaxis_title=y.replace('_', ' ').title(),
        legend_title=color.replace('_', ' ').title(),
        template="plotly_white",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    # Add a horizontal line at CA threshold if plotting attendance percentage
    if y == "attendance_percentage":
        fig.add_hline(
            y=90,
            line_dash="dash",
            line_color="red",
            annotation_text="CA Threshold (90%)",
            annotation_position="top right"
        )
    
    return fig

def plot_heatmap(df, x="grade", y="school", values="attendance_percentage", aggfunc="mean"):
    """
    Create a heatmap visualization
    
    Args:
        df (pd.DataFrame): Input dataframe
        x (str): Column for x-axis
        y (str): Column for y-axis
        values (str): Column for cell values
        aggfunc (str): Aggregation function
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Create pivot table
    pivot_df = pd.pivot_table(
        df,
        values=values,
        index=y,
        columns=x,
        aggfunc=aggfunc,
        fill_value=0
    )
    
    # Create heatmap
    fig = px.imshow(
        pivot_df,
        color_continuous_scale="RdYlGn_r" if values == "absent_days" else "RdYlGn",
        title=f"{values.replace('_', ' ').title()} by {y.replace('_', ' ').title()} and {x.replace('_', ' ').title()}",
        labels=dict(color=values.replace('_', ' ').title())
    )
    
    fig.update_layout(
        xaxis_title=x.replace('_', ' ').title(),
        yaxis_title=y.replace('_', ' ').title(),
        template="plotly_white"
    )
    
    return fig

def plot_stacked_bar(df, x="grade", color="ca_status", aggfunc="count", normalize=True):
    """
    Create a stacked bar chart
    
    Args:
        df (pd.DataFrame): Input dataframe
        x (str): Column for x-axis
        color (str): Column for stacking
        aggfunc (str): Aggregation function
        normalize (bool): Whether to normalize the bars to percentages
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Group data
    if normalize:
        # Calculate proportions within each x category
        grouped = df.groupby([x, color]).size().reset_index(name="Count")
        grouped["Percentage"] = grouped.groupby(x)["Count"].transform(lambda x: x / x.sum() * 100)
        y_value = "Percentage"
        y_title = "Percentage (%)"
    else:
        # Use raw counts
        grouped = df.groupby([x, color]).size().reset_index(name="Count")
        y_value = "Count"
        y_title = "Count"
    
    # Create stacked bar chart
    fig = px.bar(
        grouped,
        x=x,
        y=y_value,
        color=color,
        color_discrete_map={
            "CA": VIZ_CONFIG["color_palette"]["ca_color"],
            "No-CA": VIZ_CONFIG["color_palette"]["no_ca_color"]
        },
        title=f"Distribution of {color.replace('_', ' ').title()} by {x.replace('_', ' ').title()}"
    )
    
    fig.update_layout(
        xaxis_title=x.replace('_', ' ').title(),
        yaxis_title=y_title,
        legend_title=color.replace('_', ' ').title(),
        template="plotly_white",
        barmode="stack"
    )
    
    return fig

def plot_temporal_trends(df, time_col="academic_year", value_col="attendance_percentage", group_by=None):
    """
    Create a line chart of temporal trends
    
    Args:
        df (pd.DataFrame): Input dataframe
        time_col (str): Column for time values
        value_col (str): Column for the metric to track
        group_by (str, optional): Column to group by
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Group data by time column
    if group_by and group_by in df.columns:
        # Group by time and category
        grouped = df.groupby([time_col, group_by])[value_col].mean().reset_index()
        
        # Create line chart with multiple lines
        fig = px.line(
            grouped,
            x=time_col,
            y=value_col,
            color=group_by,
            markers=True,
            title=f"{value_col.replace('_', ' ').title()} Over Time by {group_by.replace('_', ' ').title()}"
        )
    else:
        # Group by time only
        grouped = df.groupby(time_col)[value_col].mean().reset_index()
        
        # Create line chart with single line
        fig = px.line(
            grouped,
            x=time_col,
            y=value_col,
            markers=True,
            title=f"{value_col.replace('_', ' ').title()} Over Time"
        )
    
    fig.update_layout(
        xaxis_title=time_col.replace('_', ' ').title(),
        yaxis_title=value_col.replace('_', ' ').title(),
        template="plotly_white"
    )
    
    # Add a horizontal line at CA threshold if plotting attendance percentage
    if value_col == "attendance_percentage":
        fig.add_hline(
            y=90,
            line_dash="dash",
            line_color="red",
            annotation_text="CA Threshold (90%)",
            annotation_position="top right"
        )
    
    return fig

def plot_correlation_matrix(df, columns=None, method='pearson'):
    """
    Create a correlation matrix heatmap
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list, optional): List of columns to include
        method (str): Correlation method
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Filter columns if specified
    if columns:
        # Ensure all columns exist
        valid_columns = [col for col in columns if col in df.columns]
        plot_df = df[valid_columns]
    else:
        # Use only numeric columns
        plot_df = df.select_dtypes(include=['number'])
    
    # Calculate correlation matrix
    corr_matrix = plot_df.corr(method=method)
    
    # Create correlation heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix",
        labels=dict(color="Correlation")
    )
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        template="plotly_white"
    )
    
    return fig

def plot_grade_progression(df, student_ids=None, time_col="academic_year", grade_col="grade"):
    """
    Create a visualization of grade progression over time
    
    Args:
        df (pd.DataFrame): Input dataframe
        student_ids (list, optional): List of student IDs to include
        time_col (str): Column for time values
        grade_col (str): Column for grade values
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Filter by student IDs if specified
    if student_ids:
        if "student_id" in df.columns:
            plot_df = df[df["student_id"].isin(student_ids)]
        else:
            plot_df = df
    else:
        # Sample 10 students if there are more than 10
        if "student_id" in df.columns and df["student_id"].nunique() > 10:
            sampled_ids = df["student_id"].sample(10).tolist()
            plot_df = df[df["student_id"].isin(sampled_ids)]
        else:
            plot_df = df
    
    # Create a pivot table with student_id as index, time_col as columns, and grade_col as values
    if "student_id" in plot_df.columns:
        pivot_df = plot_df.pivot_table(
            index="student_id",
            columns=time_col,
            values=grade_col,
            aggfunc="mean"
        )
        
        # Create a heatmap
        fig = px.imshow(
            pivot_df,
            color_continuous_scale="Viridis",
            title="Grade Progression Over Time",
            labels=dict(color="Grade")
        )
        
        fig.update_layout(
            xaxis_title=time_col.replace('_', ' ').title(),
            yaxis_title="Student ID",
            template="plotly_white"
        )
    else:
        # Create a group by showing average grade by time
        grouped = plot_df.groupby(time_col)[grade_col].mean().reset_index()
        
        # Create a line chart
        fig = px.line(
            grouped,
            x=time_col,
            y=grade_col,
            markers=True,
            title="Average Grade Progression Over Time"
        )
        
        fig.update_layout(
            xaxis_title=time_col.replace('_', ' ').title(),
            yaxis_title=grade_col.replace('_', ' ').title(),
            template="plotly_white"
        )
    
    return fig

def get_color_scale_for_risk(risk_level):
    """
    Get a color based on risk level
    
    Args:
        risk_level (float): Risk level (0-1)
        
    Returns:
        str: Hex color code
    """
    if risk_level >= 0.75:
        return "#ff6b6b"  # High risk (red)
    elif risk_level >= 0.5:
        return "#ffd43b"  # Medium-high risk (yellow)
    elif risk_level >= 0.25:
        return "#74c0fc"  # Medium-low risk (blue)
    else:
        return "#51cf66"  # Low risk (green)
