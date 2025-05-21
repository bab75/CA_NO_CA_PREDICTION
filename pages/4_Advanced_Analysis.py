"""
Advanced Analysis Page for the CA Prediction System
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import configuration
from config import MODEL_CONFIG

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualizations import (
    plot_feature_importance,
    plot_metric_comparison,
    plot_bubble_chart,
    plot_heatmap,
    plot_stacked_bar,
    plot_temporal_trends,
    plot_correlation_matrix,
    plot_grade_progression
)
from config import VIZ_CONFIG

# Set page config
st.set_page_config(
    page_title="Advanced Analysis - CA Prediction System",
    page_icon="📊",
    layout="wide"
)

# Add CSS for styling
st.markdown("""
<style>
    .analysis-container {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .insight-box {
        margin-top: 10px;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

def display_feature_importance_analysis():
    """
    Display feature importance analysis
    """
    st.subheader("Feature Importance Analysis")
    
    # Check if feature importances are available
    if "feature_importances" not in st.session_state:
        st.info("No feature importance data available. Please train models first.")
        return
    
    # Get feature importances
    feature_importances = st.session_state["feature_importances"]
    
    if not feature_importances:
        st.info("Feature importance data not available for the trained models.")
        return
    
    # Model selection
    model_options = list(feature_importances.keys())
    selected_model = st.selectbox(
        "Select Model",
        options=model_options,
        index=0 if model_options else 0,
        key="feature_importance_model"
    )
    
    if not selected_model or selected_model not in feature_importances:
        st.info("No feature importance data available for the selected model.")
        return
    
    importance = feature_importances[selected_model]
    
    if not importance:
        st.info("Feature importance data not available for the selected model.")
        return
    
    # Number of features to display
    top_n = st.slider(
        "Number of Features to Display",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
        key="feature_importance_top_n"
    )
    
    # Plot feature importance
    importance_fig = plot_feature_importance(importance, top_n=top_n)
    
    if importance_fig:
        st.plotly_chart(importance_fig, use_container_width=True)
        
        # Add insights
        st.markdown("""
        <div class="insight-box">
            <h4>Feature Importance Insights</h4>
            <p>Feature importance indicates which factors have the strongest influence on chronic absenteeism predictions:</p>
            <ul>
                <li>Higher values indicate more influential features</li>
                <li>Focus intervention strategies on the top factors</li>
                <li>Consider if the model's priorities align with domain expertise</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Could not generate feature importance visualization.")

def display_model_performance_comparison():
    """
    Display model performance comparison
    """
    st.subheader("Model Performance Comparison")
    
    # Check if model metrics are available
    if "model_metrics" not in st.session_state:
        st.info("No model metrics available. Please train models first.")
        return
    
    # Get model metrics
    model_metrics = st.session_state["model_metrics"]
    
    if not model_metrics:
        st.info("No model metrics available. Please train models first.")
        return
    
    # Available metrics
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metrics_display = {
        "accuracy": "Accuracy", 
        "precision": "Precision", 
        "recall": "Recall", 
        "f1": "F1 Score", 
        "roc_auc": "ROC AUC"
    }
    
    # Select metrics to compare
    selected_metrics = st.multiselect(
        "Select Metrics to Compare",
        options=metrics,
        default=["accuracy", "f1"],
        format_func=lambda x: metrics_display[x],
        key="model_comparison_metrics"
    )
    
    if not selected_metrics:
        st.info("Please select at least one metric to compare.")
        return
    
    # Create a subplot with multiple metrics
    rows = len(selected_metrics)
    fig = make_subplots(rows=rows, cols=1, subplot_titles=[metrics_display[m] for m in selected_metrics])
    
    for i, metric in enumerate(selected_metrics):
        comparison_fig = plot_metric_comparison(model_metrics, metric)
        
        if comparison_fig:
            # Extract the traces from the comparison figure
            for trace in comparison_fig.data:
                fig.add_trace(trace, row=i+1, col=1)
                
            # Update layout for this subplot
            fig.update_yaxes(title_text=metrics_display[metric], row=i+1, col=1)
    
    # Update the overall layout
    fig.update_layout(
        height=300 * rows,
        title="Model Performance Comparison",
        showlegend=False,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insights
    st.markdown("""
    <div class="insight-box">
        <h4>Model Comparison Insights</h4>
        <p>Compare different models across various performance metrics:</p>
        <ul>
            <li><strong>Accuracy:</strong> Overall correctness of predictions</li>
            <li><strong>Precision:</strong> How many of the positive predictions were correct</li>
            <li><strong>Recall:</strong> How many actual positive cases were caught</li>
            <li><strong>F1 Score:</strong> Balance between precision and recall</li>
            <li><strong>ROC AUC:</strong> Model's ability to distinguish between classes</li>
        </ul>
        <p>For chronic absenteeism prediction, high recall is often prioritized to ensure at-risk students are identified.</p>
    </div>
    """, unsafe_allow_html=True)

def display_attendance_academic_correlation():
    """
    Display correlation between attendance and academic performance
    """
    st.subheader("Attendance vs. Academic Performance")
    
    # Check if data is available
    data_sources = []
    
    if "historical_data" in st.session_state:
        data_sources.append("Historical Data")
    
    if "prediction_results" in st.session_state:
        data_sources.append("Prediction Results")
    
    if not data_sources:
        st.info("No data available for analysis. Please generate or upload data first.")
        return
    
    # Select data source
    data_source = st.radio(
        "Select Data Source",
        options=data_sources,
        key="attendance_academic_data_source"
    )
    
    # Get the selected data
    if data_source == "Historical Data":
        data = st.session_state["historical_data"]
        title_prefix = "Historical"
    else:  # Prediction Results
        data = st.session_state["prediction_results"]
        title_prefix = "Predicted"
    
    # Check if required columns exist
    if "academic_performance" not in data.columns or "attendance_percentage" not in data.columns:
        st.warning("Academic performance and/or attendance percentage columns not found in the data.")
        return
    
    # Create scatter plot
    st.subheader(f"{title_prefix} Attendance vs. Academic Performance")
    
    # Create bubble chart
    bubble_fig = plot_bubble_chart(
        data,
        x="academic_performance",
        y="attendance_percentage",
        size="absent_days" if "absent_days" in data.columns else None,
        color="ca_status" if "ca_status" in data.columns else "predicted_ca_status"
    )
    
    st.plotly_chart(bubble_fig, use_container_width=True)
    
    # Calculate correlation
    correlation = data["attendance_percentage"].corr(data["academic_performance"])
    
    # Display correlation
    st.metric("Correlation Coefficient", f"{correlation:.4f}")
    
    # Add insights
    st.markdown(f"""
    <div class="insight-box">
        <h4>Correlation Insights</h4>
        <p>The correlation coefficient between attendance and academic performance is <strong>{correlation:.4f}</strong>.</p>
        <ul>
            <li>A positive correlation (close to 1) indicates higher attendance is associated with better academic performance</li>
            <li>A negative correlation (close to -1) would indicate higher attendance is associated with worse performance</li>
            <li>A correlation near 0 indicates little to no relationship</li>
        </ul>
        <p>This visualization helps identify whether students with chronic absenteeism also struggle academically.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Additional group analysis
    st.subheader("Group Analysis")
    
    # Get columns for group analysis
    grouping_columns = [col for col in data.columns if col in [
        "grade", "gender", "school", "meal_code", "special_need", 
        "bus_long_trip", "shelter", "suspended", "dropout_status"
    ]]
    
    if not grouping_columns:
        st.info("No suitable columns found for group analysis.")
        return
    
    # Select column for group analysis
    group_column = st.selectbox(
        "Group By",
        options=grouping_columns,
        index=0 if "grade" in grouping_columns else 0,
        key="attendance_academic_group_column"
    )
    
    # Group data
    grouped_data = data.groupby(group_column).agg({
        "attendance_percentage": "mean",
        "academic_performance": "mean",
        "ca_status" if "ca_status" in data.columns else "predicted_ca_status": 
            lambda x: sum(x == "CA") / len(x) * 100
    }).reset_index()
    
    grouped_data.columns = [group_column, "Average Attendance %", "Average Academic Performance", "CA %"]
    
    # Sort by group column
    grouped_data = grouped_data.sort_values(group_column)
    
    # Create the figure
    fig = go.Figure()
    
    # Add bar for CA percentage
    fig.add_trace(go.Bar(
        x=grouped_data[group_column],
        y=grouped_data["CA %"],
        name="CA %",
        marker_color=VIZ_CONFIG["color_palette"]["ca_color"]
    ))
    
    # Add lines for attendance and academic performance
    fig.add_trace(go.Scatter(
        x=grouped_data[group_column],
        y=grouped_data["Average Attendance %"],
        name="Avg. Attendance %",
        mode="lines+markers",
        marker=dict(color=VIZ_CONFIG["color_palette"]["primary"])
    ))
    
    fig.add_trace(go.Scatter(
        x=grouped_data[group_column],
        y=grouped_data["Average Academic Performance"],
        name="Avg. Academic Performance",
        mode="lines+markers",
        marker=dict(color=VIZ_CONFIG["color_palette"]["secondary"])
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Attendance, Academic Performance, and CA % by {group_column.replace('_', ' ').title()}",
        xaxis_title=group_column.replace('_', ' ').title(),
        yaxis_title="Percentage",
        legend=dict(x=0.01, y=0.99, bordercolor="gray", borderwidth=1),
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_risk_distribution():
    """
    Display risk distribution across different dimensions
    """
    st.subheader("Risk Distribution Analysis")
    
    # Check if prediction results are available
    if "prediction_results" not in st.session_state:
        st.info("No prediction results available. Please run predictions first.")
        return
    
    # Get prediction results
    results = st.session_state["prediction_results"]
    
    # Check if risk score is available
    if "ca_risk_score" not in results.columns:
        st.warning("Risk score column not found in the prediction results.")
        return
    
    # Get columns for dimension analysis
    dimension_columns = [col for col in results.columns if col in [
        "grade", "gender", "school", "meal_code", "special_need", 
        "bus_long_trip", "shelter", "suspended", "dropout_status"
    ]]
    
    if not dimension_columns:
        st.info("No suitable columns found for dimension analysis.")
        return
    
    # Select primary and secondary dimensions
    col1, col2 = st.columns(2)
    
    with col1:
        primary_dimension = st.selectbox(
            "Primary Dimension",
            options=dimension_columns,
            index=0 if "grade" in dimension_columns else 0,
            key="risk_primary_dimension"
        )
    
    with col2:
        secondary_dimensions = [dim for dim in dimension_columns if dim != primary_dimension]
        
        if secondary_dimensions:
            secondary_dimension = st.selectbox(
                "Secondary Dimension (for Heatmap)",
                options=secondary_dimensions,
                index=0,
                key="risk_secondary_dimension"
            )
        else:
            secondary_dimension = None
    
    # Create risk distribution across primary dimension
    st.subheader(f"Risk Distribution by {primary_dimension.replace('_', ' ').title()}")
    
    # Group by primary dimension
    primary_grouped = results.groupby(primary_dimension).agg({
        "ca_risk_score": ["mean", "median", "std", "count"]
    }).reset_index()
    
    primary_grouped.columns = [primary_dimension, "Mean Risk", "Median Risk", "Risk StdDev", "Count"]
    
    # Sort by primary dimension
    primary_grouped = primary_grouped.sort_values(primary_dimension)
    
    # Create the figure
    fig = px.bar(
        primary_grouped,
        x=primary_dimension,
        y="Mean Risk",
        error_y="Risk StdDev",
        color="Mean Risk",
        hover_data=["Median Risk", "Count"],
        color_continuous_scale="RdYlGn_r",
        title=f"Average Risk Score by {primary_dimension.replace('_', ' ').title()}"
    )
    
    # Add a horizontal line at high risk threshold (0.75)
    fig.add_hline(
        y=0.75,
        line_dash="dash",
        line_color="red",
        annotation_text="High Risk Threshold (0.75)",
        annotation_position="top right"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=primary_dimension.replace('_', ' ').title(),
        yaxis_title="Average Risk Score",
        yaxis=dict(range=[0, 1]),
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create heatmap if secondary dimension is available
    if secondary_dimension:
        st.subheader(f"Risk Heatmap: {primary_dimension.replace('_', ' ').title()} vs {secondary_dimension.replace('_', ' ').title()}")
        
        # Create heatmap
        heatmap_fig = plot_heatmap(
            results, 
            x=primary_dimension, 
            y=secondary_dimension, 
            values="ca_risk_score", 
            aggfunc="mean"
        )
        
        st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # Risk distribution histogram
    st.subheader("Risk Score Distribution")
    
    # Create histogram of risk scores
    hist_fig = px.histogram(
        results,
        x="ca_risk_score",
        color="predicted_ca_status",
        nbins=20,
        opacity=0.7,
        color_discrete_map={
            "CA": VIZ_CONFIG["color_palette"]["ca_color"],
            "No-CA": VIZ_CONFIG["color_palette"]["no_ca_color"]
        },
        title="Distribution of Risk Scores"
    )
    
    # Add vertical lines for risk thresholds
    hist_fig.add_vline(
        x=0.75,
        line_dash="dash",
        line_color="red",
        annotation_text="High Risk (0.75)",
        annotation_position="top right"
    )
    
    hist_fig.add_vline(
        x=0.5,
        line_dash="dash",
        line_color="orange",
        annotation_text="Medium Risk (0.5)",
        annotation_position="top right"
    )
    
    hist_fig.add_vline(
        x=0.25,
        line_dash="dash",
        line_color="green",
        annotation_text="Low Risk (0.25)",
        annotation_position="top right"
    )
    
    # Update layout
    hist_fig.update_layout(
        xaxis_title="Risk Score",
        yaxis_title="Count",
        barmode="overlay",
        legend_title="Predicted Status",
        template="plotly_white"
    )
    
    st.plotly_chart(hist_fig, use_container_width=True)
    
    # Add insights
    st.markdown("""
    <div class="insight-box">
        <h4>Risk Distribution Insights</h4>
        <p>Understanding how risk is distributed across different dimensions helps target interventions:</p>
        <ul>
            <li>Identify which groups have the highest average risk scores</li>
            <li>Look for intersections of factors that create high-risk situations</li>
            <li>Consider how resources should be allocated based on risk distribution</li>
            <li>Risk thresholds: <span style="color: red;">High (≥0.75)</span>, <span style="color: orange;">Medium (≥0.5)</span>, <span style="color: green;">Low (<0.5)</span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def display_temporal_trends():
    """
    Display temporal attendance trends
    """
    st.subheader("Temporal Trends Analysis")
    
    # Check if historical data is available
    if "historical_data" not in st.session_state:
        st.info("No historical data available. Please generate or upload historical data first.")
        return
    
    # Get historical data
    historical_data = st.session_state["historical_data"]
    
    # Check if academic_year column exists
    if "academic_year" not in historical_data.columns:
        st.warning("Academic year column not found in the historical data.")
        return
    
    # Get metrics for trend analysis
    metric_options = [col for col in historical_data.columns if col in [
        "attendance_percentage", "academic_performance", "absent_days", "present_days"
    ]]
    
    if not metric_options:
        st.info("No suitable metrics found for trend analysis.")
        return
    
    # Select metric for trend analysis
    metric = st.selectbox(
        "Select Metric",
        options=metric_options,
        index=0 if "attendance_percentage" in metric_options else 0,
        key="temporal_trend_metric"
    )
    
    # Get grouping columns
    grouping_columns = [col for col in historical_data.columns if col in [
        "grade", "gender", "school", "meal_code", "special_need", 
        "bus_long_trip", "shelter", "suspended", "dropout_status"
    ]]
    
    # Select grouping column
    group_by = st.selectbox(
        "Group By",
        options=["None"] + grouping_columns,
        index=0,
        key="temporal_trend_group_by"
    )
    
    # Create trend plot
    st.subheader(f"Temporal Trends: {metric.replace('_', ' ').title()}")
    
    if group_by == "None":
        trend_fig = plot_temporal_trends(
            historical_data,
            time_col="academic_year",
            value_col=metric
        )
    else:
        trend_fig = plot_temporal_trends(
            historical_data,
            time_col="academic_year",
            value_col=metric,
            group_by=group_by
        )
    
    st.plotly_chart(trend_fig, use_container_width=True)
    
    # Add CA percentage trend if available
    if "ca_status" in historical_data.columns:
        st.subheader("Chronic Absenteeism Rate Over Time")
        
        # Calculate CA percentage by academic year
        ca_by_year = historical_data.groupby("academic_year")["ca_status"].apply(
            lambda x: sum(x == "CA") / len(x) * 100
        ).reset_index()
        ca_by_year.columns = ["Academic Year", "CA Percentage"]
        
        # Create line chart
        ca_fig = px.line(
            ca_by_year,
            x="Academic Year",
            y="CA Percentage",
            markers=True,
            title="Chronic Absenteeism Rate Over Time"
        )
        
        ca_fig.update_layout(
            xaxis_title="Academic Year",
            yaxis_title="CA Percentage (%)",
            template="plotly_white"
        )
        
        st.plotly_chart(ca_fig, use_container_width=True)
    
    # Monthly or Seasonal Patterns (if applicable)
    if "month" in historical_data.columns:
        st.subheader("Monthly Attendance Patterns")
        
        # Group by month
        monthly_data = historical_data.groupby("month").agg({
            "attendance_percentage": "mean",
            "ca_status": lambda x: sum(x == "CA") / len(x) * 100 if "ca_status" in historical_data.columns else None
        }).reset_index()
        
        # Create bar chart
        month_fig = px.bar(
            monthly_data,
            x="month",
            y="attendance_percentage",
            color="attendance_percentage",
            color_continuous_scale="RdYlGn",
            title="Average Attendance by Month"
        )
        
        month_fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Average Attendance (%)",
            template="plotly_white"
        )
        
        st.plotly_chart(month_fig, use_container_width=True)
    
    # Add insights
    st.markdown("""
    <div class="insight-box">
        <h4>Temporal Trends Insights</h4>
        <p>Analyzing how attendance and other metrics change over time provides valuable context:</p>
        <ul>
            <li>Identify long-term trends (improving or worsening attendance)</li>
            <li>Spot cyclical patterns (e.g., lower attendance in winter months)</li>
            <li>Evaluate the impact of past interventions by looking for changes in trends</li>
            <li>Project future attendance patterns based on historical data</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def display_cohort_analysis():
    """
    Display cohort analysis for student progression
    """
    st.subheader("Student Cohort Analysis")
    
    # Check if historical data is available
    if "historical_data" not in st.session_state:
        st.info("No historical data available. Please generate or upload historical data first.")
        return
    
    # Get historical data
    historical_data = st.session_state["historical_data"]
    
    # Check if required columns exist
    required_columns = ["student_id", "academic_year", "grade"]
    
    missing_columns = [col for col in required_columns if col not in historical_data.columns]
    
    if missing_columns:
        st.warning(f"Required columns missing for cohort analysis: {', '.join(missing_columns)}")
        return
    
    # Get unique student IDs
    student_ids = historical_data["student_id"].unique()
    
    # Get grade progression for students over years
    st.subheader("Grade Progression Analysis")
    
    # Create grade progression visualization
    progression_fig = plot_grade_progression(
        historical_data,
        time_col="academic_year",
        grade_col="grade"
    )
    
    st.plotly_chart(progression_fig, use_container_width=True)
    
    # Cohort attendance trends
    st.subheader("Cohort Attendance Trends")
    
    # Check if attendance_percentage column exists
    if "attendance_percentage" not in historical_data.columns:
        st.warning("Attendance percentage column not found in the historical data.")
    else:
        # Create a new grade cohort column based on the first grade observed for each student
        students_first_grade = historical_data.sort_values("academic_year").groupby("student_id")["grade"].first()
        historical_data_with_cohort = historical_data.copy()
        historical_data_with_cohort["grade_cohort"] = historical_data_with_cohort["student_id"].map(students_first_grade)
        
        # Group by cohort and academic year
        cohort_data = historical_data_with_cohort.groupby(["grade_cohort", "academic_year"])["attendance_percentage"].mean().reset_index()
        
        # Create line chart
        cohort_fig = px.line(
            cohort_data,
            x="academic_year",
            y="attendance_percentage",
            color="grade_cohort",
            markers=True,
            title="Average Attendance by Grade Cohort Over Time"
        )
        
        cohort_fig.update_layout(
            xaxis_title="Academic Year",
            yaxis_title="Average Attendance (%)",
            legend_title="Starting Grade",
            template="plotly_white"
        )
        
        st.plotly_chart(cohort_fig, use_container_width=True)
    
    # Retention and progression analysis
    st.subheader("Retention and Progression Analysis")
    
    # Create a pivot table of student counts by grade and academic year
    pivot_data = historical_data.pivot_table(
        index="grade",
        columns="academic_year",
        values="student_id",
        aggfunc="count",
        fill_value=0
    )
    
    # Create heatmap
    heatmap_fig = px.imshow(
        pivot_data,
        color_continuous_scale="Viridis",
        title="Student Count by Grade and Academic Year"
    )
    
    heatmap_fig.update_layout(
        xaxis_title="Academic Year",
        yaxis_title="Grade",
        template="plotly_white"
    )
    
    st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # Add insights
    st.markdown("""
    <div class="insight-box">
        <h4>Cohort Analysis Insights</h4>
        <p>Tracking students over time provides insights into progression patterns:</p>
        <ul>
            <li>Identify how attendance changes as students progress through grades</li>
            <li>Detect grade levels where attendance typically drops</li>
            <li>Analyze retention patterns and grade repetition</li>
            <li>Compare different cohorts to understand generational differences</li>
        </ul>
        <p>This analysis is particularly valuable for long-term planning and targeted interventions at critical transition points.</p>
    </div>
    """, unsafe_allow_html=True)

def display_correlation_analysis():
    """
    Display correlation analysis between different factors
    """
    st.subheader("Multi-Factor Correlation Analysis")
    
    # Check if data is available
    data_sources = []
    
    if "historical_data" in st.session_state:
        data_sources.append("Historical Data")
    
    if "prediction_results" in st.session_state:
        data_sources.append("Prediction Results")
    
    if not data_sources:
        st.info("No data available for analysis. Please generate or upload data first.")
        return
    
    # Select data source
    data_source = st.radio(
        "Select Data Source",
        options=data_sources,
        key="correlation_data_source"
    )
    
    # Get the selected data
    if data_source == "Historical Data":
        data = st.session_state["historical_data"]
    else:  # Prediction Results
        data = st.session_state["prediction_results"]
    
    # Get numerical columns for correlation analysis
    numerical_columns = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    if not numerical_columns:
        st.info("No numerical columns found for correlation analysis.")
        return
    
    # Select columns for correlation analysis
    selected_columns = st.multiselect(
        "Select Columns for Correlation Analysis",
        options=numerical_columns,
        default=[col for col in numerical_columns if col in [
            "attendance_percentage", "academic_performance", "present_days", "absent_days",
            "ca_risk_score" if "ca_risk_score" in numerical_columns else None
        ] and col is not None],
        key="correlation_columns"
    )
    
    if not selected_columns:
        st.info("Please select at least two columns for correlation analysis.")
        return
    
    if len(selected_columns) < 2:
        st.info("Please select at least two columns for correlation analysis.")
        return
    
    # Create correlation matrix
    st.subheader("Correlation Matrix")
    
    # Plot correlation matrix
    corr_fig = plot_correlation_matrix(data, columns=selected_columns)
    st.plotly_chart(corr_fig, use_container_width=True)
    
    # Add insights
    st.markdown("""
    <div class="insight-box">
        <h4>Correlation Matrix Insights</h4>
        <p>The correlation matrix shows relationships between different numerical factors:</p>
        <ul>
            <li>Values close to 1 indicate strong positive correlation (as one increases, the other increases)</li>
            <li>Values close to -1 indicate strong negative correlation (as one increases, the other decreases)</li>
            <li>Values close to 0 indicate little to no correlation</li>
        </ul>
        <p>Focus on strong correlations (both positive and negative) to understand factor relationships.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Pair plot for selected factors
    if len(selected_columns) > 1 and len(selected_columns) <= 4:
        st.subheader("Pair Plot")
        
        # Create scatter plots for each pair of selected columns
        for i, col1 in enumerate(selected_columns):
            for j, col2 in enumerate(selected_columns):
                if i < j:  # Avoid redundant plots
                    scatter_fig = px.scatter(
                        data,
                        x=col1,
                        y=col2,
                        color="ca_status" if "ca_status" in data.columns else "predicted_ca_status" if "predicted_ca_status" in data.columns else None,
                        opacity=0.7,
                        color_discrete_map={
                            "CA": VIZ_CONFIG["color_palette"]["ca_color"],
                            "No-CA": VIZ_CONFIG["color_palette"]["no_ca_color"]
                        },
                        title=f"{col1.replace('_', ' ').title()} vs {col2.replace('_', ' ').title()}"
                    )
                    
                    scatter_fig.update_layout(
                        xaxis_title=col1.replace('_', ' ').title(),
                        yaxis_title=col2.replace('_', ' ').title(),
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(scatter_fig, use_container_width=True)

def display_system_summary():
    """
    Display system summary report
    """
    st.subheader("System Summary Report")
    
    # Check if data is available
    if "historical_data" not in st.session_state and "prediction_results" not in st.session_state:
        st.info("No data available for summary. Please generate or upload data first.")
        return
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    # Historical data stats
    if "historical_data" in st.session_state:
        historical_data = st.session_state["historical_data"]
        
        historical_count = len(historical_data)
        historical_ca_count = sum(historical_data["ca_status"] == "CA") if "ca_status" in historical_data.columns else 0
        historical_ca_percentage = (historical_ca_count / historical_count * 100) if historical_count > 0 else 0
        
        col1.metric("Historical Records", historical_count)
        col2.metric("Historical CA Count", historical_ca_count)
    else:
        col1.metric("Historical Records", "N/A")
        col2.metric("Historical CA Count", "N/A")
    
    # Prediction results stats
    if "prediction_results" in st.session_state:
        prediction_results = st.session_state["prediction_results"]
        
        prediction_count = len(prediction_results)
        prediction_ca_count = sum(prediction_results["predicted_ca_status"] == "CA") if "predicted_ca_status" in prediction_results.columns else 0
        prediction_ca_percentage = (prediction_ca_count / prediction_count * 100) if prediction_count > 0 else 0
        
        col3.metric("Prediction Records", prediction_count)
        col4.metric("Predicted CA Count", prediction_ca_count)
    else:
        col3.metric("Prediction Records", "N/A")
        col4.metric("Predicted CA Count", "N/A")
    
    # CA percentage comparison
    if "historical_data" in st.session_state and "prediction_results" in st.session_state:
        historical_data = st.session_state["historical_data"]
        prediction_results = st.session_state["prediction_results"]
        
        historical_count = len(historical_data)
        historical_ca_count = sum(historical_data["ca_status"] == "CA") if "ca_status" in historical_data.columns else 0
        historical_ca_percentage = (historical_ca_count / historical_count * 100) if historical_count > 0 else 0
        
        prediction_count = len(prediction_results)
        prediction_ca_count = sum(prediction_results["predicted_ca_status"] == "CA") if "predicted_ca_status" in prediction_results.columns else 0
        prediction_ca_percentage = (prediction_ca_count / prediction_count * 100) if prediction_count > 0 else 0
        
        # Calculate the delta
        ca_delta = prediction_ca_percentage - historical_ca_percentage
        
        # Display the comparison
        st.subheader("CA Percentage Comparison")
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Historical CA %", f"{historical_ca_percentage:.2f}%")
        col2.metric("Predicted CA %", f"{prediction_ca_percentage:.2f}%")
        col3.metric("Change", f"{ca_delta:.2f}%", delta=ca_delta)
        
        # Create a bar chart comparing CA percentages
        comparison_data = pd.DataFrame({
            "Category": ["Historical", "Predicted"],
            "CA Percentage": [historical_ca_percentage, prediction_ca_percentage]
        })
        
        comparison_fig = px.bar(
            comparison_data,
            x="Category",
            y="CA Percentage",
            color="Category",
            color_discrete_map={
                "Historical": VIZ_CONFIG["color_palette"]["primary"],
                "Predicted": VIZ_CONFIG["color_palette"]["highlight"]
            },
            title="CA Percentage Comparison"
        )
        
        comparison_fig.update_layout(
            xaxis_title="",
            yaxis_title="CA Percentage (%)",
            template="plotly_white"
        )
        
        st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Model summary
    if "trained_models" in st.session_state:
        st.subheader("Model Summary")
        
        # Get models and metrics
        trained_models = st.session_state["trained_models"]
        model_metrics = st.session_state["model_metrics"]
        
        # Create a summary table
        model_names = [MODEL_CONFIG["models"][model] for model in trained_models.keys()]
        model_counts = [len(model_metrics)]
        
        # Best model
        if "best_model" in st.session_state:
            best_model = st.session_state["best_model"]
            best_model_name = MODEL_CONFIG["models"][best_model]
            best_model_metrics = model_metrics[best_model]
            
            st.markdown(f"**Best Model:** {best_model_name}")
            
            # Display best model metrics
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Accuracy", f"{best_model_metrics.get('accuracy', 0)*100:.2f}%")
            col2.metric("Precision", f"{best_model_metrics.get('precision', 0)*100:.2f}%")
            col3.metric("Recall", f"{best_model_metrics.get('recall', 0)*100:.2f}%")
            col4.metric("F1 Score", f"{best_model_metrics.get('f1', 0)*100:.2f}%")
    
    # Add insights and recommendations
    st.subheader("System Insights & Recommendations")
    
    st.markdown("""
    <div class="insight-box">
        <h4>Key Insights:</h4>
        <ul>
            <li>Monitor the predicted CA percentage compared to historical data</li>
            <li>Focus interventions on high-risk students and schools</li>
            <li>Use correlation analysis to understand the underlying factors</li>
            <li>Track cohort progression to identify critical transition points</li>
        </ul>
        
        <h4>Recommended Next Steps:</h4>
        <ol>
            <li>Implement targeted interventions based on risk factors</li>
            <li>Continuously collect attendance data to improve model accuracy</li>
            <li>Compare model predictions with actual outcomes</li>
            <li>Refine models based on performance and feedback</li>
            <li>Share insights with relevant stakeholders for coordinated action</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Custom CSS for animations and styling
    st.markdown("""
    <style>
        @keyframes shimmer {
            0% {
                background-position: -1000px 0;
            }
            100% {
                background-position: 1000px 0;
            }
        }
        
        .analytics-header {
            background: linear-gradient(150deg, #0F766E, #0891B2, #0369A1);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            position: relative;
            overflow: hidden;
        }
        
        .analytics-header::after {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, 
                rgba(255,255,255,0) 0%, 
                rgba(255,255,255,0.2) 50%, 
                rgba(255,255,255,0) 100%);
            animation: shimmer 3s infinite;
            background-size: 1000px 100%;
        }
        
        .analytics-header h1 {
            font-size: 2.5rem;
            margin: 0;
            padding: 0;
            position: relative;
            z-index: 1;
        }
        
        .analytics-header p {
            font-size: 1.1rem;
            margin-top: 10px;
            opacity: 0.9;
            position: relative;
            z-index: 1;
        }
        
        .analytics-icon {
            float: left;
            margin-right: 20px;
            position: relative;
            z-index: 1;
        }
        
        /* Navigation selection styling */
        .stSelectbox label {
            font-weight: 600;
            color: #0F766E;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(90deg, #0F766E, #0891B2);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .stButton > button:hover {
            background: linear-gradient(90deg, #0891B2, #0369A1);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }
        
        /* Content section animations */
        .main-content {
            animation: fadeIn 0.5s ease-in-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Custom header with animation
    st.markdown("""
    <div class="analytics-header">
        <div class="analytics-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="white">
                <path d="M3 3v18h18V3H3zm17 17H4V4h16v16zM7 10h2v7H7v-7zm4-3h2v10h-2V7zm4 6h2v4h-2v-4z"/>
            </svg>
        </div>
        <h1>Advanced Analysis</h1>
        <p>Discover deep insights about attendance patterns and model performance with sophisticated data analysis</p>
        <div style="clear: both;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hide the original icon (we're using our custom header)
    st.image("assets/analytics_icon.svg", width=0)
    
    # Navigation sidebar
    analysis_option = st.sidebar.selectbox(
        "Select Analysis Type",
        options=[
            "Feature Importance Analysis",
            "Model Performance Comparison",
            "Attendance vs. Academic Performance",
            "Risk Distribution Analysis",
            "Temporal Trends Analysis",
            "Cohort Analysis",
            "Multi-Factor Correlation Analysis",
            "System Summary Report"
        ]
    )
    
    # Display selected analysis
    if analysis_option == "Feature Importance Analysis":
        display_feature_importance_analysis()
    
    elif analysis_option == "Model Performance Comparison":
        display_model_performance_comparison()
    
    elif analysis_option == "Attendance vs. Academic Performance":
        display_attendance_academic_correlation()
    
    elif analysis_option == "Risk Distribution Analysis":
        display_risk_distribution()
    
    elif analysis_option == "Temporal Trends Analysis":
        display_temporal_trends()
    
    elif analysis_option == "Cohort Analysis":
        display_cohort_analysis()
    
    elif analysis_option == "Multi-Factor Correlation Analysis":
        display_correlation_analysis()
    
    elif analysis_option == "System Summary Report":
        display_system_summary()
    
    # Documentation
    with st.expander("Documentation: Advanced Analysis"):
        st.markdown("""
        ## Advanced Analysis Module Documentation

        This module provides in-depth analysis tools to explore the factors influencing chronic absenteeism and evaluate prediction models.

        ### Available Analysis Types

        1. **Feature Importance Analysis**
            - Visualize which factors have the strongest influence on chronic absenteeism predictions
            - Compare different models to see if they prioritize similar features
            - Use insights to focus intervention strategies

        2. **Model Performance Comparison**
            - Compare multiple models across different performance metrics
            - Understand the trade-offs between metrics like precision and recall
            - Select the most appropriate model for your specific needs

        3. **Attendance vs. Academic Performance**
            - Explore the relationship between attendance and academic achievement
            - Identify whether poor attendance is affecting specific demographic groups
            - Analyze correlations across different student segments

        4. **Risk Distribution Analysis**
            - Visualize how chronic absenteeism risk is distributed across dimensions
            - Identify high-risk groups that need immediate intervention
            - Understand the patterns in risk distribution for strategic planning

        5. **Temporal Trends Analysis**
            - Track changes in attendance patterns over time
            - Identify seasonal variations and long-term trends
            - Evaluate the impact of past interventions on attendance

        6. **Cohort Analysis**
            - Follow student groups as they progress through grades
            - Identify critical transition points where attendance issues emerge
            - Compare different cohorts to understand generational changes

        7. **Multi-Factor Correlation Analysis**
            - Discover relationships between different student attributes and attendance
            - Identify which factors correlate with chronic absenteeism
            - Build a better understanding of the complex interactions between variables

        8. **System Summary Report**
            - Get a high-level overview of the entire system
            - Compare historical and predicted CA percentages
            - View key metrics and recommendations for next steps

        ### Using the Analysis Tools

        - Select the desired analysis type from the sidebar
        - Configure the specific parameters for each analysis
        - Explore the visualizations and insights provided
        - Use the findings to inform intervention strategies and policy decisions

        ### Additional Tips

        - Look for patterns across multiple analysis types to build a comprehensive understanding
        - Pay attention to both system-wide trends and individual student factors
        - Use the insights to develop targeted intervention strategies
        - Share relevant visualizations with stakeholders to build support for initiatives
        """)

if __name__ == "__main__":
    main()
