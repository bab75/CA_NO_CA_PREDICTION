# Applying the changes to add percentage formatting to the risk score progress bars and add a hover help text.
"""
Predictions Page for the CA Prediction System
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import pickle
import base64
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_generator import generate_student_data, generate_school_names
from utils.model_utils import make_prediction, load_model
from config import DATA_CONFIG, MODEL_CONFIG, DROPDOWN_OPTIONS, DEFAULT_VALUES
from utils.visualizations import (
    plot_bubble_chart,
    plot_heatmap,
    plot_stacked_bar,
    get_color_scale_for_risk
)

# Set page config
st.set_page_config(
    page_title="Predictions - CA Prediction System",
    page_icon="📊",
    layout="wide"
)

# Add CSS for styling
st.markdown("""
<style>
    .risk-high {
        color: #ff6b6b;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffd43b;
        font-weight: bold;
    }
    .risk-low {
        color: #51cf66;
        font-weight: bold;
    }
    .prediction-card {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .student-details {
        margin-top: 10px;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .intervention-suggestion {
        margin-top: 15px;
        padding: 10px;
        background-color: #e3f2fd;
        border-left: 4px solid #4e89ae;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)

def display_prepare_data_tab():
    """
    Display the prepare current data tab
    """
    st.subheader("Current Year Data")

    # Data source options
    data_source = st.radio(
        "Select Data Source",
        options=["Generated Current Year Data", "Generate New Data", "Upload Data"],
        index=0,
        key="prediction_data_source"
    )

    # Default prediction data
    prediction_data = None

    if data_source == "Generated Current Year Data":
        # Check if current year data exists in session state
        if "current_data" in st.session_state:
            prediction_data = st.session_state["current_data"]
            st.success(f"Using generated current year data with {len(prediction_data)} records")
        else:
            st.warning("No current year data found. Please generate data using the 'Generate New Data' option below.")
            return None
    elif data_source == "Generate New Data":
        # Generate new current year data
        st.subheader("Generate Current Year Data")

        with st.expander("Data Generation Options", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                num_students = st.number_input(
                    "Number of Students",
                    min_value=DATA_CONFIG["min_students"],
                    max_value=DATA_CONFIG["max_students"],
                    value=DEFAULT_VALUES["num_students"],
                    step=100,
                    key="current_gen_num_students"
                )

            with col2:
                # Use current year as default
                current_year = DATA_CONFIG["max_year"]
                st.number_input(
                    "Academic Year",
                    min_value=DATA_CONFIG["min_year"],
                    max_value=DATA_CONFIG["max_year"],
                    value=current_year,
                    step=1,
                    key="current_gen_year",
                    disabled=True,
                    help="Current year data is always for the current academic year"
                )

            with col3:
                num_schools = st.number_input(
                    "Number of Schools",
                    min_value=1,
                    max_value=50,
                    value=DEFAULT_VALUES["num_schools"],
                    step=1,
                    key="current_gen_num_schools"
                )

            # School name pattern
            school_base_name = st.text_input(
                "School Name Pattern",
                value="School-",
                help="The base name for schools (e.g., '10U', 'A-SCHOOL')",
                key="current_gen_school_base_name"
            )

            # School names
            school_names = generate_school_names(school_base_name, num_schools)
            st.write("Generated School Names:", ", ".join(school_names))

            # Generate button
            if st.button("Generate Current Year Data", key="generate_current_data_btn"):
                with st.spinner("Generating current year data..."):
                    try:
                        # Generate data with is_historical=False for current year
                        current_data = generate_student_data(
                            num_students=num_students,
                            academic_years=[current_year],
                            schools=school_names,
                            num_schools=num_schools,
                            school_base_name=school_base_name,
                            is_historical=False
                        )

                        # Store in session state
                        st.session_state["current_data"] = current_data

                        # Set as prediction data
                        prediction_data = current_data

                        # Show success message with animation
                        st.success(f"Successfully generated {len(current_data)} student records for the current academic year!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error generating data: {str(e)}")

    else:
        # File uploader for prediction data
        uploaded_file = st.file_uploader(
            "Upload current year data (CSV or Excel)",
            type=["csv", "xlsx"],
            key="prediction_data_upload"
        )

        if uploaded_file is not None:
            try:
                # Determine file type
                file_extension = uploaded_file.name.split(".")[-1].lower()

                # Read the file
                if file_extension == "csv":
                    prediction_data = pd.read_csv(uploaded_file)
                elif file_extension == "xlsx":
                    prediction_data = pd.read_excel(uploaded_file)

                st.success(f"Successfully loaded data with {len(prediction_data)} records")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return None
        else:
            st.info("Please upload a data file")
            return None

    # Ensure we have prediction data
    if prediction_data is None:
        return None

    # Display data
    st.subheader("Data Preview")
    st.dataframe(prediction_data.head(5), use_container_width=True)

    # Data validation
    st.subheader("Data Validation")

    # Check if there's a trained model
    if "best_model" not in st.session_state:
        st.warning("No trained model found. Please train a model first.")
        return None

    # Get the required features from the best model
    best_model = st.session_state["best_model"]

    # Initialize model feature selections if they don't exist
    if "model_selected_categorical" not in st.session_state:
        st.session_state["model_selected_categorical"] = []
    if "model_selected_numerical" not in st.session_state:
        st.session_state["model_selected_numerical"] = []

    model_categorical = st.session_state["model_selected_categorical"]
    model_numerical = st.session_state["model_selected_numerical"]

    # Check if all required features are present
    missing_features = []
    for feature in model_categorical + model_numerical:
        if feature not in prediction_data.columns:
            missing_features.append(feature)

    if missing_features:
        st.error(f"The following features are missing in the prediction data: {', '.join(missing_features)}")

        # Suggest matching the data structure
        st.info("Please make sure your prediction data has the same structure as the training data.")
        return None

    # Store prediction data in session state
    st.session_state["prediction_data"] = prediction_data

    # Data validation success
    st.success("Data validation successful! The prediction data is compatible with the trained model.")

    # Show data statistics
    st.subheader("Data Statistics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(prediction_data))
    col2.metric("Schools", prediction_data["school"].nunique() if "school" in prediction_data.columns else "N/A")
    col3.metric("Grades", prediction_data["grade"].nunique() if "grade" in prediction_data.columns else "N/A")

    return prediction_data

def display_patterns_dashboard():
    """
    Display the patterns and correlations dashboard
    """
    st.subheader("Patterns & Correlations Dashboard")

    # Check if there's historical data
    if "historical_data" not in st.session_state:
        st.info("No historical data found. Patterns analysis requires historical data.")
        return

    historical_data = st.session_state["historical_data"]

    # Display historical patterns
    st.markdown("### Historical Attendance Patterns")

    # Create tabs for different pattern analyses
    pattern_tabs = st.tabs([
        "Attendance by Grade",
        "Attendance by School",
        "Gender Distribution",
        "Special Needs Impact"
    ])

    # Attendance by Grade
    with pattern_tabs[0]:
        # Create a stacked bar chart of CA status by grade
        if "grade" in historical_data.columns and "ca_status" in historical_data.columns:
            grade_fig = plot_stacked_bar(historical_data, x="grade", color="ca_status")
            st.plotly_chart(grade_fig, use_container_width=True)

            st.markdown("""
            **Pattern Analysis:** This chart shows the distribution of Chronic Absenteeism (CA) across different grades.
            - Higher CA percentages in certain grades may indicate transition challenges
            - Grade levels with significantly higher CA rates may need targeted interventions
            """)

    # Attendance by School
    with pattern_tabs[1]:
        # Create a heatmap of attendance percentage by school and grade
        if "school" in historical_data.columns and "grade" in historical_data.columns and "attendance_percentage" in historical_data.columns:
            school_fig = plot_heatmap(historical_data, x="grade", y="school", values="attendance_percentage")
            st.plotly_chart(school_fig, use_container_width=True)

            st.markdown("""
            **Pattern Analysis:** This heatmap shows average attendance percentages across schools and grades.
            - Darker areas indicate lower attendance percentages
            - Schools with consistently low attendance may need system-wide interventions
            - Specific grade-school combinations with low attendance may need targeted support
            """)

    # Gender Distribution
    with pattern_tabs[2]:
        # Create a stacked bar chart of CA status by gender
        if "gender" in historical_data.columns and "ca_status" in historical_data.columns:
            gender_fig = plot_stacked_bar(historical_data, x="gender", color="ca_status")
            st.plotly_chart(gender_fig, use_container_width=True)

            st.markdown("""
            **Pattern Analysis:** This chart shows the distribution of Chronic Absenteeism (CA) across different genders.
            - Gender-specific attendance patterns may help tailor interventions
            - Significant differences may indicate social or cultural factors affecting attendance
            """)

    # Special Needs Impact
    with pattern_tabs[3]:
        # Create a stacked bar chart of CA status by special needs status
        if "special_need" in historical_data.columns and "ca_status" in historical_data.columns:
            special_needs_fig = plot_stacked_bar(historical_data, x="special_need", color="ca_status")
            st.plotly_chart(special_needs_fig, use_container_width=True)

            st.markdown("""
            **Pattern Analysis:** This chart shows the impact of special needs status on Chronic Absenteeism (CA).
            - Students with special needs may have different attendance patterns
            - This can help identify if additional support services are needed
            """)

    # Pattern identification and editing
    st.subheader("Pattern Configuration")

    # Initialize patterns if not already in session state
    if "identified_patterns" not in st.session_state:
        # Default patterns based on common findings
        st.session_state["identified_patterns"] = [
            {
                "name": "Grade Transition",
                "description": "Students tend to have higher absenteeism during grade transition years (e.g., entering middle or high school)",
                "enabled": True
            },
            {
                "name": "Special Needs Support",
                "description": "Students with special needs have higher CA rates when support services are insufficient",
                "enabled": True
            },
            {
                "name": "Transportation Issues",
                "description": "Students with long bus trips have higher absence rates, especially during winter months",
                "enabled": True
            },
            {
                "name": "Academic Performance Correlation",
                "description": "Lower academic performance is correlated with higher absenteeism rates",
                "enabled": True
            }
        ]

    # Display and edit patterns
    for i, pattern in enumerate(st.session_state["identified_patterns"]):
        col1, col2 = st.columns([1, 10])

        with col1:
            # Enable/disable pattern
            pattern_enabled = st.checkbox("", value=pattern["enabled"], key=f"pattern_enabled_{i}")
            st.session_state["identified_patterns"][i]["enabled"] = pattern_enabled

        with col2:
            # Display pattern name and description
            st.markdown(f"**{pattern['name']}**")
            st.markdown(pattern["description"])

    # Add new pattern
    with st.expander("Add New Pattern"):
        with st.form(key="add_pattern_form"):
            pattern_name = st.text_input("Pattern Name")
            pattern_description = st.text_area("Pattern Description")

            submit_button = st.form_submit_button("Add Pattern")

            if submit_button and pattern_name and pattern_description:
                st.session_state["identified_patterns"].append({
                    "name": pattern_name,
                    "description": pattern_description,
                    "enabled": True
                })
                st.success(f"Added new pattern: {pattern_name}")

def display_run_prediction_tab():
    """
    Display the run prediction tab
    """
    st.subheader("Run Prediction")

    # Check if prediction data is available
    if "prediction_data" not in st.session_state:
        st.warning("No prediction data available. Please prepare data first.")
        return

    # Check if a trained model is available
    if "best_model" not in st.session_state or "best_model_pipeline" not in st.session_state:
        st.warning("No trained model available. Please train a model first.")
        return

    # Get prediction data and model
    prediction_data = st.session_state["prediction_data"]
    best_model = st.session_state["best_model"]
    best_model_pipeline = st.session_state["best_model_pipeline"]

    # Model information
    st.markdown(f"### Model: {MODEL_CONFIG['models'][best_model]}")

    metrics = st.session_state["best_model_metrics"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
    col2.metric("F1 Score", f"{metrics.get('f1', 0)*100:.2f}%")
    col3.metric("Model Type", MODEL_CONFIG['models'][best_model])

    # Add an explanation about CA definition
    st.markdown("""
    <div style="background-color: #F0F9FF; padding: 15px; border-radius: 5px; margin: 15px 0; border-left: 4px solid #3B82F6;">
        <h4 style="color: #1E40AF; margin-top: 0;">How Chronic Absenteeism (CA) is Determined</h4>
        <p style="margin-bottom: 0;">A student is classified as <strong>Chronically Absent (CA)</strong> if:</p>
        <ul>
            <li>Their <strong>attendance percentage is ≤ 90%</strong> (primary definition)</li>
            <li>OR the model predicts they are at high risk based on multiple factors</li>
        </ul>
        <p>The prediction system prioritizes the 90% attendance threshold for direct classification.</p>
    </div>
    """, unsafe_allow_html=True)

    # Prediction button
    if st.button("Predict CA Risk", key="run_prediction_button"):
        with st.spinner("Running predictions..."):
            try:
                # Get features
                model_categorical = st.session_state["model_selected_categorical"]
                model_numerical = st.session_state["model_selected_numerical"]

                # Select only the features needed for prediction
                features = model_categorical + model_numerical
                X_pred = prediction_data[features]

                # DIRECT CA DETERMINATION: First check if attendance data is available
                # If it is, use the 90% threshold as the primary determinant
                result_data = prediction_data.copy()

                if 'attendance_percentage' in prediction_data.columns:
                    # This is the most reliable way to determine CA status
                    attendance_threshold = 90.0
                    # Apply the rule: CA if attendance percentage ≤ 90%
                    ca_statuses = ["CA" if att <= attendance_threshold else "No-CA"
                                   for att in prediction_data['attendance_percentage']]

                    print(f"Direct CA determination based on attendance threshold: {sum(1 for s in ca_statuses if s == 'CA')} CA students found")

                    # Calculate CA risk scores based on how far below/above threshold
                    ca_risk_scores = []
                    for att in prediction_data['attendance_percentage']:
                        if att <= attendance_threshold:
                            # Below threshold: Higher risk (closer to 1.0)
                            # Scale: 0% attendance → 1.0 risk, 90% attendance → 0.5 risk
                            risk = 1.0 - (att / (attendance_threshold * 2))
                            ca_risk_scores.append(min(1.0, max(0.5, risk)))
                        else:
                            # Above threshold: Lower risk (closer to 0.0)
                            # Scale: 90% attendance → 0.5 risk, 100% attendance → 0.0 risk
                            margin = (att - attendance_threshold) / (100 - attendance_threshold)
                            risk = 0.5 - (margin * 0.5)
                            ca_risk_scores.append(max(0.0, min(0.5, risk)))

                    # Set these values
                    result_data["predicted_ca_status"] = ca_statuses
                    result_data["ca_risk_score"] = ca_risk_scores
                    result_data["prediction_method"] = "Attendance Threshold (Direct)"

                    # Also set predictions variable for display
                    predictions = ca_statuses
                    probabilities = ca_risk_scores
                else:
                    # No attendance data available, need to use the model
                    try:
                        # Try to make model predictions
                        predictions, probabilities = make_prediction(best_model_pipeline, X_pred)

                        # Add predictions to the data
                        result_data["predicted_ca_status"] = predictions

                        if probabilities is not None:
                            result_data["ca_risk_score"] = probabilities

                        result_data["prediction_method"] = "ML Model"
                    except Exception as model_error:
                        # If model prediction fails, log the error and use a fallback approach
                        st.error(f"Model prediction error: {str(model_error)}. Using factor-based risk assessment instead.")
                        import traceback
                        traceback.print_exc()

                        # FALLBACK APPROACH: Use a rule-based approach based on risk factors
                        risk_scores = []
                        risk_factors = {}

                        # Define risk factors we can analyze
                        if 'shelter_status' in X_pred.columns:
                            risk_factors['shelter'] = X_pred['shelter_status'].apply(
                                lambda x: 0.3 if x == "Yes" else 0.0
                            )

                        if 'meal_code' in X_pred.columns:
                            risk_factors['meals'] = X_pred['meal_code'].apply(
                                lambda x: 0.2 if x in ["Free", "Reduced"] else 0.0
                            )

                        if 'suspension_days' in X_pred.columns:
                            risk_factors['suspension'] = X_pred['suspension_days'].apply(
                                lambda x: min(0.2, x * 0.05)  # 0.05 per suspension day, max 0.2
                            )

                        if 'behavior_incidents' in X_pred.columns:
                            risk_factors['behavior'] = X_pred['behavior_incidents'].apply(
                                lambda x: min(0.2, x * 0.04)  # 0.04 per incident, max 0.2
                            )

                        if 'gpa' in X_pred.columns:
                            risk_factors['academics'] = X_pred['gpa'].apply(
                                lambda x: max(0.0, (4.0 - x) * 0.075)  # Lower GPA = higher risk
                            )

                        # Calculate composite risk score
                        if risk_factors:
                            # Create DataFrame from risk factors dictionary
                            risk_df = pd.DataFrame(risk_factors)

                            # Sum across all factors (maximum 1.0)
                            composite_risk = risk_df.sum(axis=1).apply(lambda x: min(1.0, x))

                            # Threshold for CA prediction (0.5 or higher is CA)
                            ca_predictions = composite_risk.apply(lambda x: "CA" if x >= 0.5 else "No-CA")

                            # Set values
                            result_data["predicted_ca_status"] = ca_predictions
                            result_data["ca_risk_score"] = composite_risk
                            result_data["prediction_method"] = "Risk Factor Analysis (Fallback)"

                            # Update variables for display
                            predictions = ca_predictions.values
                            probabilities = composite_risk.values
                        else:
                            # Not enough risk factors to analyze - assign moderate risk to all
                            predictions = ["No-CA"] * len(X_pred)
                            probabilities = [0.3] * len(X_pred)
                            result_data["predicted_ca_status"] = predictions
                            result_data["ca_risk_score"] = probabilities
                            result_data["prediction_method"] = "Default (Insufficient Data)"

                # Store prediction results in session state
                st.session_state["prediction_results"] = result_data

                # Success message with snow animation
                st.success("Predictions completed successfully!")

                # Show snow animation to indicate prediction completion
                st.snow()

                # Add a more visible success banner with animation
                st.markdown("""
                <style>
                @keyframes fade-in-down {
                    0% { opacity: 0; transform: translateY(-20px); }
                    100% { opacity: 1; transform: translateY(0); }
                }

                .prediction-success-banner {
                    background: linear-gradient(135deg, #43A047, #2E7D32);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    margin: 20px 0;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                    animation: fade-in-down 1s ease-out;
                }
                </style>

                <div class="prediction-success-banner">
                    <h2>🔮 Prediction Analysis Complete!</h2>
                    <p>Your chronic absenteeism predictions are ready for review and intervention planning.</p>
                </div>
                """, unsafe_allow_html=True)

                # Count CA students correctly by comparing each value individually
                ca_count = sum(1 for label in result_data["predicted_ca_status"] if label == "CA")
                ca_percentage = (ca_count / len(result_data)) * 100 if len(result_data) > 0 else 0

                # Display with enhanced visuals
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
                padding: 20px; border-radius: 10px; margin: 20px 0; border: 1px solid #FCD34D;">
                    <h3 style="color: #92400E; margin-bottom: 10px;">Prediction Summary</h3>
                    <div style="display: flex; justify-content: space-around; text-align: center;">
                        <div>
                            <h2 style="color: #B45309; font-size: 2rem; margin: 0;">{ca_count}</h2>
                            <p style="color: #92400E;">Predicted CA Students</p>
                        </div>
                        <div>
                            <h2 style="color: #B45309; font-size: 2rem; margin: 0;">{ca_percentage:.1f}%</h2>
                            <p style="color: #92400E;">Predicted CA Percentage</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Display comprehensive breakdown with recommendations
                st.markdown("## Detailed Analysis & Intervention Recommendations")

                tabs = st.tabs([
                    "CA Students (Current)",
                    "Risk Assessment (Non-CA)",
                    "Intervention Recommendations"
                ])

                # Calculate additional metrics
                school_days_per_year = 180  # Standard number of school days

                # TAB 1: Current CA Students Analysis
                with tabs[0]:
                    if ca_count > 0:
                        st.markdown("### Current Chronic Absenteeism Students")

                        # Filter only CA students
                        ca_students = result_data[result_data["predicted_ca_status"] == "CA"].copy()

                        # Calculate days missed for CA students
                        if 'attendance_percentage' in ca_students.columns:
                            ca_students['missed_days'] = school_days_per_year * (1 - (ca_students['attendance_percentage'] / 100))
                            ca_students['missed_days'] = ca_students['missed_days'].round().astype(int)

                            # Categorize based on severity
                            ca_students['severity'] = ca_students['attendance_percentage'].apply(
                                lambda x: "Severe" if x <= 80 else ("High" if x <= 85 else "Moderate")
                            )

                        # Display the CA students table with risk levels
                        st.dataframe(
                            ca_students[[
                                'student_id', 'grade', 'attendance_percentage',
                                'missed_days', 'severity', 'ca_risk_score'
                            ]].sort_values(by='attendance_percentage'),
                            hide_index=True,
                            column_config={
                                'student_id': 'Student ID',
                                'grade': 'Grade',
                                'attendance_percentage': st.column_config.NumberColumn(
                                    'Attendance %',
                                    format="%.1f%%"
                                ),
                                'missed_days': 'School Days Missed',
                                'severity': 'Severity Level',
                                'ca_risk_score': st.column_config.ProgressColumn(
                                    'Risk Score',
                                    format="%.1f%%",
                                    min_value=0,
                                    max_value=1,
                                    help="Risk score from 0-100%"
                                ),
                            },
                            use_container_width=True
                        )

                        # Display severity breakdown
                        if 'severity' in ca_students.columns:
                            severity_counts = ca_students['severity'].value_counts()

                            # Create pie chart
                            fig = px.pie(
                                values=severity_counts.values,
                                names=severity_counts.index,
                                title="CA Severity Distribution",
                                color=severity_counts.index,
                                color_discrete_map={
                                    'Severe': '#DC2626',    # Red
                                    'High': '#EA580C',      # Orange
                                    'Moderate': '#CA8A04'   # Amber
                                }
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No students currently classified as Chronically Absent.")

                # TAB 2: Risk Assessment for Non-CA Students
                with tabs[1]:
                    # Filter only Non-CA students
                    non_ca_students = result_data[result_data["predicted_ca_status"] == "No-CA"].copy()

                    if len(non_ca_students) > 0:
                        st.markdown("### Future CA Risk Assessment for Current Non-CA Students")

                        # Add risk level category based on risk score
                        non_ca_students['risk_level'] = non_ca_students['ca_risk_score'].apply(
                            lambda x: "High" if x >= 0.4 else ("Medium" if x >= 0.2 else "Low")
                        )

                        # Count students by risk level
                        risk_level_counts = non_ca_students['risk_level'].value_counts()

                        col1, col2 = st.columns([1, 1])

                        with col1:
                            # Create risk level distribution chart
                            fig = px.bar(
                                x=risk_level_counts.index,
                                y=risk_level_counts.values,
                                title="Non-CA Students: Future Risk Distribution",
                                labels={'x': 'Risk Level', 'y': 'Number of Students'},
                                color=risk_level_counts.index,
                                color_discrete_map={
                                    'High': '#DC2626',      # Red
                                    'Medium': '#EA580C',    # Orange
                                    'Low': '#16A34A'        # Green
                                }
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            # Display table of high-risk Non-CA students
                            high_risk = non_ca_students[non_ca_students['risk_level'] == 'High']
                            if len(high_risk) > 0:
                                st.markdown("#### High-Risk Non-CA Students")
                                st.markdown("These students have >40% risk of becoming CA in the future")
                                st.dataframe(
                                    high_risk[['student_id', 'grade', 'attendance_percentage', 'ca_risk_score']],
                                    hide_index=True,
                                    column_config={
                                        'student_id': 'Student ID',
                                        'grade': 'Grade',
                                        'attendance_percentage': st.column_config.NumberColumn(
                                            'Current Attendance %',
                                            format="%.1f%%"
                                        ),
                                        'ca_risk_score': st.column_config.ProgressColumn(
                                            'Future CA Risk',
                                            format="%.1f%%",
                                            min_value=0,
                                            max_value=1,
                                            help="Risk score from 0-100%"
                                        )
                                    },
                                    use_container_width=True
                                )
                            else:
                                st.success("No high-risk Non-CA students identified.")
                    else:
                        st.info("No Non-CA students in the dataset.")

                # TAB 3: Intervention Recommendations
                with tabs[2]:
                    st.markdown("### Intervention Recommendations")

                    # Display different intervention strategies based on student categories
                    if ca_count > 0:
                        st.markdown("#### For Current CA Students")

                        # Group students by severity (if available)
                        if 'severity' in ca_students.columns:
                            for severity in ['Severe', 'High', 'Moderate']:
                                if severity in ca_students['severity'].values:
                                    severity_group = ca_students[ca_students['severity'] == severity]

                                    # Display severity-specific recommendations
                                    if severity == 'Severe':
                                        st.markdown("""
                                        <div style="background-color: #FEE2E2; padding: 15px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #DC2626;">
                                            <h4 style="color: #991B1B; margin-top: 0;">Severe CA Students (≤80% Attendance)</h4>
                                            <p>These students have missed significant school time and need immediate intensive intervention.</p>
                                            <ul>
                                                <li><strong>Daily Check-ins</strong>: Implement daily attendance monitoring system</li>
                                                <li><strong>Home Visits</strong>: Schedule home visits to identify barriers to attendance</li>
                                                <li><strong>Individualized Attendance Plan</strong>: Create customized plan addressing specific barriers</li>
                                                <li><strong>Support Services</strong>: Connect with social services, mental health resources, or housing support if SHELTER status indicates need</li>
                                                <li><strong>Transportation Assistance</strong>: Provide transportation options if logistical issues contribute to absences</li>
                                                </ul>
                                            <p><em>Students inthis category will likely need full-year interventionto improve habits, with reduced expectation of achieving Non-CA status this year.</em></div></p>                                        """, unsafe_allow_html=True)
                                    elif severity == 'High':
                                        st.markdown("""
                                        <div style="background-color: #FFEDD5; padding: 15px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #EA580C;">
                                            <h4 style="color: #9A3412; margin-top: 0;">High-Risk CA Students (81-85% Attendance)</h4>
                                            <p>These students have concerning attendance patterns but can improve with targeted support.</p>
                                            <ul>
                                                <li><strong>Weekly Check-ins</strong>: Regular attendance monitoring and support</li>
                                                <li><strong>Attendance Contracts</strong>: Develop formal agreements with goals and incentives</li>
                                                <li><strong>Group Interventions</strong>: Include in attendance support groups</li>
                                                <li><strong>Meal Program Support</strong>: If MEAL-CODE indicates free/reduced lunch, ensure access to nutrition programs</li>
                                                <li><strong>Academic Support</strong>: Provide tutoring to address academic challenges due to absences</li>
                                            </ul>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:  # Moderate
                                        st.markdown("""
                                        <div style="background-color: #FEF3C7; padding: 15px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #CA8A04;">
                                            <h4 style="color: #854D0E; margin-top: 0;">Moderate-Risk CA Students (86-90% Attendance)</h4>
                                            <p>These students are just below the CA threshold and can likely improve with appropriate support.</p>
                                            <ul>
                                                <li><strong>Attendance Awareness</strong>: Education on importance of consistent attendance</li>
                                                <li><strong>Incentive Programs</strong>: Recognition and rewards for attendance improvement</li>
                                                <li><strong>Peer Support</strong>: Connect with student mentors or buddies for accountability</li>
                                                <li><strong>Parent Communication</strong>: Regular updates to parents/guardians on attendance status</li>
                                                <li><strong>School Engagement</strong>: Encourage participation in extracurricular activities to increase school connection</li>
                                            </ul>
                                        </div>
                                        """, unsafe_allow_html=True)

                        # Display specific factor-based recommendations
                        st.markdown("#### Factor-Specific Interventions")

                        factor_cols = st.columns(2)

                        with factor_cols[0]:
                            st.markdown("""
                            <div style="background-color: #F0F9FF; padding: 15px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #0284C7;">
                                <h4 style="color: #0C4A6E; margin-top: 0;">SHELTER Status Interventions</h4>
                                <ul>
                                    <li>Connect families with housing resources and assistance programs</li>
                                    <li>Provide school supplies, uniforms, and hygiene kits</li>
                                    <li>Arrange stable transportation options</li>
                                    <li>Ensure access to before/after school programs</li>
                                    <li>Connect with social workers for wraparound support</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)

                            st.markdown("""
                            <div style="background-color: #F0FDF4; padding: 15px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #16A34A;">
                                <h4 style="color: #166534; margin-top: 0;">Academic Performance Interventions</h4>
                                <ul>
                                    <li>Provide targeted tutoring for missed content</li>
                                    <li>Implement credit recovery options</li>
                                    <li>Create individualized learning plans</li>
                                    <li>Schedule regular academic check-ins</li>
                                    <li>Connect academic improvement to attendance importance</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)

                        with factor_cols[1]:
                            st.markdown("""
                            <div style="background-color: #FEFCE8; padding: 15px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #CA8A04;">
                                <h4 style="color: #854D0E; margin-top: 0;">MEAL-CODE Interventions</h4>
                                <ul>
                                    <li>Ensure enrollment in free/reduced meal programs</li>
                                    <li>Provide information on weekend food backpack programs</li>
                                    <li>Connect with community food resources</li>
                                    <li>Coordinate with food pantries for family assistance</li>
                                    <li>Consider breakfast-in-classroom programs to encourage attendance</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)

                            st.markdown("""
                            <div style="background-color: #F5F3FF; padding: 15px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #7C3AED;">
                                <h4 style="color: #5B21B6; margin-top: 0;">Behavioral/Engagement Interventions</h4>
                                <ul>
                                    <li>Implement positive behavioral support systems</li>
                                    <li>Provide counseling for underlying issues</li>
                                    <li>Create behavior intervention plans</li>
                                    <li>Increase engagement through interest-based activities</li>
                                    <li>Consider mentoring programs for positive relationships</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)

                    if len(non_ca_students) > 0 and 'risk_level' in non_ca_students.columns:
                        st.markdown("#### For At-Risk Non-CA Students")

                        # Check for high-risk Non-CA students
                        high_risk = non_ca_students[non_ca_students['risk_level'] == 'High']
                        if len(high_risk) > 0:
                            st.markdown("""
                            <div style="background-color: #FFF1F2; padding: 15px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #BE123C;">
                                <h4 style="color: #9F1239; margin-top: 0;">High-Risk Prevention Strategies</h4>
                                <p>Proactive interventions for students at high risk of becoming CA:</p>
                                <ul>
                                    <li><strong>Early Warning System</strong>: Monitor attendance weekly for any decline</li>
                                    <li><strong>Prevention Conferences</strong>: Schedule meetings with student and family to discuss attendance importance</li>
                                    <li><strong>Attendance Buddies</strong>: Pair with peers who have strong attendance</li>
                                    <li><strong>Success Plans</strong>: Create personalized plan focusing on attendance goals</li>
                                    <li><strong>Engagement Activities</strong>: Connect to school activities aligning with student interests</li>
                                    <li><strong>Address Specific Factors</strong>: Target interventions based on the risk factors (SHELTER, MEAL-CODE, etc.)</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)

                            # If there are medium-risk students, show recommendations for them
                            medium_risk = non_ca_students[non_ca_students['risk_level'] == 'Medium']
                            if len(medium_risk) > 0:
                                st.markdown("""
                                <div style="background-color: #FFEDD5; padding: 15px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #EA580C;">
                                    <h4 style="color: #9A3412; margin-top: 0;">Medium-Risk Awareness Strategies</h4>
                                    <ul>
                                        <li>Monthly attendance monitoring</li>
                                        <li>Attendance awareness education</li>
                                        <li>Regular communication with families</li>
                                        <li>Positive reinforcement for consistent attendance</li>
                                        <li>Address specific risk factors early</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)

                    # Display school-wide recommendations
                    st.markdown("#### School-Wide Attendance Initiatives")
                    st.markdown("""
                    <div style="background-color: #EFF6FF; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
                        <h4 style="color: #1E40AF; margin-top: 0;">Universal Attendance Strategies</h4>
                        <ol>
                            <li><strong>Attendance Awareness Campaigns</strong>: School-wide initiatives highlighting importance of regular attendance</li>
                            <li><strong>Clear Attendance Policies</strong>: Ensure all stakeholders understand expectations and consequences</li>
                            <li><strong>Positive School Climate</strong>: Create welcoming, engaging environment students want to attend</li>
                            <li><strong>Family Engagement</strong>: Regular communication with families about attendance</li>
                            <li><strong>Attendance Recognition</strong>: Celebrate and reward good and improved attendance</li>
                            <li><strong>Data Monitoring</strong>: Track attendance patterns to identify trends and needs</li>
                            <li><strong>Professional Development</strong>: Train staff on attendance importance and intervention strategies</li>
                            <li><strong>Community Partnerships</strong>: Collaborate with community organizations to address barriers</li>
                        </ol>
                    </div>
                    """, unsafe_allow_html=True)

                # Summary from CA Students and Risk Assessment tabs
                if 'attendance_percentage' in prediction_data.columns:
                    # Count current CA students (from CA Students tab)
                    ca_students = result_data[result_data["predicted_ca_status"] == "CA"].copy()
                    current_ca_count = len(ca_students)

                    # Count high-risk non-CA students (from Risk Assessment tab)
                    non_ca_students = result_data[result_data["predicted_ca_status"] == "No-CA"].copy()
                    if 'ca_risk_score' in non_ca_students.columns:
                        high_risk_non_ca = non_ca_students[non_ca_students['ca_risk_score'] >= 0.4]
                        high_risk_count = len(high_risk_non_ca)
                    else:
                        high_risk_count = 0

                    # Display summary
                    st.markdown("### Current Status and Risk Assessment Summary")
                    summary_cols = st.columns(2)

                    with summary_cols[0]:
                        st.markdown(f"""
                        <div style="background-color: #FEF2F2; padding: 15px; border-radius: 5px; text-align: center;">
                            <h4 style="margin-top: 0;">Current CA Students</h4>
                            <h2 style="color: #DC2626; font-size: 2.5rem; margin: 10px 0;">{current_ca_count}</h2>
                            <p>Students currently identified as CA</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with summary_cols[1]:
                        st.markdown(f"""
                        <div style="background-color: #FFF7ED; padding: 15px; border-radius: 5px; text-align: center;">
                            <h4 style="margin-top: 0;">High-Risk Non-CA Students</h4>
                            <h2 style="color: #EA580C; font-size: 2.5rem; margin: 10px 0;">{high_risk_count}</h2>
                            <p>Non-CA students at high risk (risk score ≥ 40%)</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Add total at-risk population
                    total_at_risk = current_ca_count + high_risk_count
                    st.markdown(f"""
                    <div style="background-color: #EFF6FF; padding: 15px; border-radius: 5px; text-align: center; margin-top: 10px;">
                        <h4 style="margin: 0;">Total At-Risk Population</h4>
                        <h2 style="color: #1E40AF; font-size: 2.5rem; margin: 10px 0;">{total_at_risk}</h2>
                        <p>Combined count of current CA and high-risk students</p>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")
                # Show more detailed error information for debugging
                import traceback
                st.code(traceback.format_exc(), language="python")

def display_single_student_dashboard(student_data):
    """
    Display a detailed dashboard for a single student

    Args:
        student_data (pd.Series): Data for a single student
    """
    st.markdown("""
    <div style="background-color: #f0f7fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    """, unsafe_allow_html=True)

    # Student header with ID and basic info
    st.markdown(f"""
    <h3 style="color: #1E3A8A; margin-bottom: 15px;">
        Student Analysis: {student_data.get('student_id', 'No ID')}
    </h3>
    """, unsafe_allow_html=True)

    # Basic information
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div style="background-color: white; padding: 10px; border-radius: 5px; text-align: center;">
            <h4 style="margin: 0; font-size: 0.9rem; color: #6B7280;">Grade</h4>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 5px 0;">{student_data.get('grade', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background-color: white; padding: 10px; border-radius: 5px; text-align: center;">
            <h4 style="margin: 0; font-size: 0.9rem; color: #6B7280;">School</h4>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 5px 0;">{student_data.get('school', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background-color: white; padding: 10px; border-radius: 5px; text-align: center;">
            <h4 style="margin: 0; font-size: 0.9rem; color: #6B7280;">Academic Year</h4>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 5px 0;">{student_data.get('academic_year', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)

    # Prediction result with gauge chart
    st.subheader("Risk Assessment")

    col1, col2 = st.columns([1, 2])

    # CA status
    ca_status = student_data.get('predicted_ca_status', 'Unknown')
    risk_score = student_data.get('ca_risk_score', 0)

    with col1:
        # Display CA status with appropriate styling
        if ca_status == "CA":
            st.markdown(f"""
            <div style="background-color: #FEE2E2; color: #DC2626; padding: 15px;
                        border-radius: 5px; text-align: center; margin-bottom: 10px;">
                <h3 style="margin: 0;">High Risk - CA</h3>
                <p style="font-size: 0.9rem; margin: 5px 0;">
                    This student is predicted to have Chronic Absenteeism
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #DCFCE7; color: #16A34A; padding: 15px;
                        border-radius: 5px; text-align: center; margin-bottom: 10px;">
                <h3 style="margin: 0;">Low Risk - No CA</h3>
                <p style="font-size: 0.9rem; margin: 5px 0;">
                    This student is not predicted to have Chronic Absenteeism
                </p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        # Create a gauge chart for risk score
        if isinstance(risk_score, (int, float)):
            risk_level = risk_score
        else:
            # If risk_score is a probability series, get the value for CA class
            risk_level = risk_score[1] if len(risk_score) > 1 else 0.5

        # Determine color based on risk level
        if risk_level < 0.33:
            color = "green"
        elif risk_level < 0.67:
            color = "orange"
        else:
            color = "red"

        # Create gauge chart
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_level * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 33], 'color': 'rgba(0, 250, 0, 0.1)'},
                    {'range': [33, 67], 'color': 'rgba(255, 165, 0, 0.1)'},
                    {'range': [67, 100], 'color': 'rgba(255, 0, 0, 0.1)'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))

        gauge_fig.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=50, b=10),
            font={'color': "#1E3A8A", 'family': "Arial"}
        )

        # Use a unique key combining student ID and a random suffix to avoid duplicates
        import random
        unique_suffix = random.randint(1000, 9999)
        st.plotly_chart(gauge_fig, use_container_width=True, key=f"gauge_{student_data.get('student_id', 'unknown')}_{unique_suffix}")

    # Key Indicators section
    st.subheader("Key Indicators")

    # Create 3 columns for the indicators
    col1, col2, col3 = st.columns(3)

    with col1:
        # Academic Performance gauge
        academic_performance = student_data.get('academic_performance', 0)
        academic_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=academic_performance,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Academic Performance", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "royalblue"},
                'steps': [
                    {'range': [0, 60], 'color': 'lightgray'},
                    {'range': [60, 80], 'color': 'lightblue'},
                    {'range': [80, 100], 'color': 'lightgreen'}
                ]
            }
        ))

        academic_fig.update_layout(
            height=180,
            margin=dict(l=10, r=10, t=50, b=10)
        )

        st.plotly_chart(academic_fig, use_container_width=True, key=f"academic_{student_data.get('student_id', 'unknown')}_{random.randint(1000, 9999)}")

    with col2:
        # Attendance Percentage gauge
        attendance_percentage = student_data.get('attendance_percentage', 0)
        attendance_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=attendance_percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Attendance Percentage", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 80], 'color': 'lightgray'},
                    {'range': [80, 90], 'color': 'lightyellow'},
                    {'range': [90, 100], 'color': 'lightgreen'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 2},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))

        attendance_fig.update_layout(
            height=180,
            margin=dict(l=10, r=10, t=50, b=10)
        )

        st.plotly_chart(attendance_fig, use_container_width=True, key=f"attendance_{student_data.get('student_id', 'unknown')}_{random.randint(1000, 9999)}")

    with col3:
        # Absent Days gauge
        absent_days = student_data.get('absent_days', 0)
        max_absent = 40  # Assuming maximum absent days

        absent_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=absent_days,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Absent Days", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, max_absent], 'tickwidth': 1},
                'bar': {'color': "firebrick"},
                'steps': [
                    {'range': [0, 10], 'color': 'lightgreen'},
                    {'range': [10, 20], 'color': 'lightyellow'},
                    {'range': [20, max_absent], 'color': 'lightcoral'}
                ]
            }
        ))

        absent_fig.update_layout(
            height=180,
            margin=dict(l=10, r=10, t=50, b=10)
        )

        st.plotly_chart(absent_fig, use_container_width=True, key=f"absent_{student_data.get('student_id', 'unknown')}_{random.randint(1000, 9999)}")

    # Intervention Recommendations
    st.subheader("Recommended Interventions")

    # Determine intervention categories based on student data
    interventions = []

    # Attendance intervention
    if attendance_percentage < 90:
        interventions.append({
            "category": "Attendance Support",
            "icon": "📅",
            "recommendations": [
                "Regular attendance check-ins with counselor",
                "Personalized attendance improvement plan",
                "Parent/guardian communication strategy"
            ]
        })

    # Academic intervention
    if academic_performance < 70:
        interventions.append({
            "category": "Academic Support",
            "icon": "📚",
            "recommendations": [
                "After-school tutoring program",
                "Study skills workshop",
                "Subject-specific intervention"
            ]
        })

    # Transportation intervention
    if student_data.get('bus_long_trip') == "Yes":
        interventions.append({
            "category": "Transportation Assistance",
            "icon": "🚌",
            "recommendations": [
                "Explore alternative transportation options",
                "Adjust bus route to reduce travel time",
                "Provide materials for productive use of travel time"
            ]
        })

    # Special needs intervention
    if student_data.get('special_need') == "Yes":
        interventions.append({
            "category": "Special Services Support",
            "icon": "🔍",
            "recommendations": [
                "Review and update IEP/504 plan",
                "Specialized attendance accommodations",
                "Additional support services evaluation"
            ]
        })

    # Housing/shelter intervention
    if student_data.get('shelter') in ["S", "ST"]:
        interventions.append({
            "category": "Housing Stability Support",
            "icon": "🏠",
            "recommendations": [
                "Connect with social services coordinator",
                "Provide stable learning environment resources",
                "Transportation assistance program"
            ]
        })

    # Behavioral intervention (if suspended)
    if student_data.get('suspended') == "Yes":
        interventions.append({
            "category": "Behavioral Support",
            "icon": "🤝",
            "recommendations": [
                "Behavioral intervention plan",
                "Counseling services",
                "Restorative practices implementation"
            ]
        })

    # Display interventions in a grid
    if interventions:
        # Create columns for interventions
        cols = st.columns(min(3, len(interventions)))

        for i, intervention in enumerate(interventions):
            with cols[i % len(cols)]:
                st.markdown(f"""
                <div style="background-color: white; padding: 15px; border-radius: 5px; margin-bottom: 10px; min-height: 200px;">
                    <h4 style="color: #1E3A8A; display: flex; align-items: center;">
                        <span style="font-size: 1.5rem; margin-right: 10px;">{intervention['icon']}</span>
                        {intervention['category']}
                    </h4>
                    <ul style="margin-top: 10px;">
                """, unsafe_allow_html=True)

                for rec in intervention['recommendations']:
                    st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)

                st.markdown("</ul></div>", unsafe_allow_html=True)
    else:
        st.info("No specific interventions recommended at this time.")

    st.markdown("</div>", unsafe_allow_html=True)

def display_prediction_results_tab():
    """
    Display the prediction results tab with filters
    """
    # Check if prediction data is available
    if "prediction_results" not in st.session_state:
        st.warning("No prediction data available. Please run predictions first.")
        return

    # Get prediction results
    results = st.session_state["prediction_results"]

    # Add filters in an expander
    with st.expander("Filter Results", expanded=True):
        filter_cols = st.columns([1, 1, 1, 1])

        # School filter
        if 'school' in results.columns:
            with filter_cols[0]:
                selected_schools = st.multiselect(
                    "School",
                    options=sorted(results['school'].unique()),
                    default=sorted(results['school'].unique())
                )

        # Grade filter
        if 'grade' in results.columns:
            with filter_cols[1]:
                selected_grades = st.multiselect(
                    "Grade",
                    options=sorted(results['grade'].unique()),
                    default=sorted(results['grade'].unique())
                )

        # Risk category filter
        if 'risk_category' in results.columns:
            with filter_cols[2]:
                selected_risk = st.multiselect(
                    "Risk Category",
                    options=sorted(results['risk_category'].unique()),
                    default=sorted(results['risk_category'].unique())
                )

        # CA Status filter
        if 'predicted_ca_status' in results.columns:
            with filter_cols[3]:
                selected_ca_status = st.multiselect(
                    "CA Status",
                    options=sorted(results['predicted_ca_status'].unique()),
                    default=sorted(results['predicted_ca_status'].unique())
                )

        # Apply filters
        filtered_results = results.copy()
        if 'school' in results.columns:
            filtered_results = filtered_results[filtered_results['school'].isin(selected_schools)]
        if 'grade' in results.columns:
            filtered_results = filtered_results[filtered_results['grade'].isin(selected_grades)]
        if 'risk_category' in results.columns:
            filtered_results = filtered_results[filtered_results['risk_category'].isin(selected_risk)]
        if 'predicted_ca_status' in results.columns:
            filtered_results = filtered_results[filtered_results['predicted_ca_status'].isin(selected_ca_status)]

    # Summary statistics
    st.subheader("Prediction Summary")

    # Count CA and No-CA predictions
    ca_count = sum(filtered_results["predicted_ca_status"] == "CA")
    noca_count = sum(filtered_results["predicted_ca_status"] == "No-CA")
    total_count = len(filtered_results)

    ca_percentage = (ca_count / total_count) * 100

    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", total_count)
    col2.metric("Predicted CA Students", ca_count)
    col3.metric("CA Percentage", f"{ca_percentage:.2f}%")

    # Add reset button for predictions
    if st.button("🔄 Reset Predictions", help="Clear all prediction results and start over"):
        if "prediction_results" in st.session_state:
            del st.session_state["prediction_results"]
        st.success("Predictions have been reset!")
        st.experimental_rerun()

    # Show results by school if available
    if "school" in filtered_results.columns:
        st.subheader("Results by School")

        # Group by school
        school_summary = filtered_results.groupby("school")["predicted_ca_status"].apply(
            lambda x: sum(x == "CA") / len(x) * 100
        ).reset_index()
        school_summary.columns = ["School", "CA Percentage"]

        # Sort by CA percentage (descending)
        school_summary = school_summary.sort_values("CA Percentage", ascending=False)

        # Create bar chart
        fig = px.bar(
            school_summary,
            x="School",
            y="CA Percentage",
            color="CA Percentage",
            color_continuous_scale="RdYlGn_r",
            title="Predicted CA Percentage by School"
        )

        fig.update_layout(xaxis_title="School", yaxis_title="CA Percentage (%)")

        st.plotly_chart(fig, use_container_width=True, key="school_summary_chart")

    # Show results by grade if available
    if "grade" in filtered_results.columns:
        st.subheader("Results by Grade")

        # Group by grade
        grade_summary = filtered_results.groupby("grade")["predicted_ca_status"].apply(
            lambda x: sum(x == "CA") / len(x) * 100
        ).reset_index()
        grade_summary.columns = ["Grade", "CA Percentage"]

        # Sort by grade
        grade_summary = grade_summary.sort_values("Grade")

        # Create bar chart
        fig = px.bar(
            grade_summary,
            x="Grade",
            y="CA Percentage",
            color="CA Percentage",
            color_continuous_scale="RdYlGn_r",
            title="Predicted CA Percentage by Grade"
        )

        fig.update_layout(xaxis_title="Grade", yaxis_title="CA Percentage (%)")

        st.plotly_chart(fig, use_container_width=True, key="grade_summary_chart")

    # Bubble chart of academic performance vs attendance
    if "academic_performance" in filtered_results.columns and "attendance_percentage" in filtered_results.columns:
        st.subheader("Academic Performance vs Attendance")

        bubble_fig = plot_bubble_chart(
            filtered_results,
            x="academic_performance",
            y="attendance_percentage",
            size="absent_days" if "absent_days" in filtered_results.columns else None,
            color="predicted_ca_status"
        )

        st.plotly_chart(bubble_fig, use_container_width=True, key="performance_attendance_bubble")

    # Display full results table
    st.subheader("Full Prediction Results")

    # Add risk category if risk score is available
    if "ca_risk_score" in filtered_results.columns:
        filtered_results["risk_category"] = filtered_results["ca_risk_score"].apply(
            lambda x: "High Risk" if x >= 0.75 else
                     ("Medium Risk" if x >= 0.5 else
                     ("Low-Medium Risk" if x >= 0.25 else "Low Risk"))
        )

    # Display the results table
    st.dataframe(filtered_results, use_container_width=True)

    # Download button for results
    csv = filtered_results.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="ca_prediction_results.csv">Download Prediction Results as CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

    # Single student analysis
    st.subheader("Single Student Analysis")

    # Get student IDs if available
    if "student_id" in filtered_results.columns:
        student_ids = filtered_results["student_id"].tolist()

        # Select a student
        selected_student_id = st.selectbox(
            "Select Student ID",
            options=student_ids,
            key="select_student_analysis"
        )

        # Get student data
        student_data = filtered_results[filtered_results["student_id"] == selected_student_id].iloc[0]

        # Display the student dashboard
        display_single_student_dashboard(student_data)

        # Display student details
        with st.container():
            st.markdown("### Student Details")

            # Create columns for student information            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"**Student ID:** {student_data['student_id']}")
                st.markdown(f"**School:** {student_data.get('school', 'N/A')}")
                st.markdown(f"**Grade:** {student_data.get('grade', 'N/A')}")

            with col2:
                st.markdown(f"**Gender:** {student_data.get('gender', 'N/A')}")
                st.markdown(f"**Meal Code:** {student_data.get('meal_code', 'N/A')}")
                st.markdown(f"**Special Needs:** {student_data.get('special_need', 'N/A')}")

            with col3:
                st.markdown(f"**Academic Performance:** {student_data.get('academic_performance', 'N/A'):.1f}%")
                attendance = student_data.get('attendance_percentage')
                st.markdown(f"**Attendance:** {attendance:.1f}%" if isinstance(attendance, (int, float)) else "**Attendance:** N/A")
                st.markdown(f"**Absent Days:** {student_data.get('absent_days', 'N/A')}")

            # Display prediction result
            st.markdown("### Prediction Result")

            # Risk level
            if "ca_risk_score" in student_data:
                risk_score = student_data["ca_risk_score"]
                risk_category = "High Risk" if risk_score >= 0.75 else (
                    "Medium Risk" if risk_score >= 0.5 else (
                    "Low-Medium Risk" if risk_score >= 0.25 else "Low Risk"))
                risk_color_class = "risk-high" if risk_score >= 0.75 else ("risk-medium" if risk_score >= 0.5 else "risk-low")

                st.markdown(f"""
                <div class="prediction-card">
                    <h4>CA Prediction: <span class="{risk_color_class}">{student_data['predicted_ca_status']}</span></h4>
                    <h4>Risk Assessment Score: <span class="{risk_color_class}">{risk_score:.2f}</span></h4>
                    <h4>Risk Category: <span class="{risk_color_class}">{risk_category}</span></h4>
                    <p><small>Risk Assessment shows raw probability (0-1), Category provides interpretation</small></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-card">
                    <h4>CA Prediction: {student_data['predicted_ca_status']}</h4>
                </div>
                """, unsafe_allow_html=True)

            # Contributing factors
            st.markdown("### Contributing Factors")

            # Get the enabled patterns
            if "identified_patterns" in st.session_state:
                enabled_patterns = [p for p in st.session_state["identified_patterns"] if p["enabled"]]

                if enabled_patterns:
                    st.markdown("The following patterns may be contributing to this student's absenteeism risk:")

                    for pattern in enabled_patterns:
                        # Check if pattern applies to this student
                        applies = False

                        # More flexible pattern matching
                        if pattern["name"] == "Grade Transition":
                            grade = student_data.get("grade")
                            applies = isinstance(grade, (int, float)) and grade in [6, 9]
                        elif pattern["name"] == "Special Needs Support":
                            special_need = str(student_data.get("special_need", "")).upper()
                            applies = special_need in ["YES", "Y", "TRUE", "1"]
                        elif pattern["name"] == "Transportation Issues":
                            bus_trip = str(student_data.get("bus_long_trip", "")).upper()
                            applies = bus_trip in ["YES", "Y", "TRUE", "1"]
                        elif pattern["name"] == "Academic Performance Correlation":
                            perf = student_data.get("academic_performance")
                            applies = isinstance(perf, (int, float)) and perf < 70

                        # Display pattern if it applies
                        if applies:
                            st.markdown(f"- **{pattern['name']}**: {pattern['description']}")

def main():
    st.title("Chronic Absenteeism Predictions")
    st.write("Predict at-risk students and visualize patterns using trained machine learning models")

    # Create tabs for different sections
    tabs = st.tabs([
        "Prepare Current Data",
        "Patterns & Correlations",
        "Run Prediction",
        "Prediction Results",
        "Student Analysis"
    ])

    # Tab 1: Prepare Current Data
    with tabs[0]:
        display_prepare_data_tab()

    # Tab 2: Patterns & Correlations
    with tabs[1]:
        display_patterns_dashboard()

    # Tab 3: Run Prediction
    with tabs[2]:
        display_run_prediction_tab()

    # Tab 4: Prediction Results
    with tabs[3]:
        display_prediction_results_tab()

    # Tab 5: Student Analysis
    with tabs[4]:
        st.subheader("Individual Student Analysis")
        if "prediction_results" in st.session_state:
            results = st.session_state["prediction_results"]

            # Add filters in an expander
            with st.expander("Filter Students", expanded=True):
                col1, col2, col3 = st.columns(3)

                # School filter
                with col1:
                    if 'school' in results.columns:
                        schools = sorted(results['school'].unique())
                        selected_school = st.multiselect("School", schools, default=schools, key="student_analysis_school")
                        results = results[results['school'].isin(selected_school)]

                # Grade filter
                with col2:
                    if 'grade' in results.columns:
                        grades = sorted(results['grade'].unique())
                        selected_grade = st.multiselect("Grade", grades, default=grades, key="student_analysis_grade")
                        results = results[results['grade'].isin(selected_grade)]

                # CA Status filter
                with col3:
                    if 'predicted_ca_status' in results.columns:
                        ca_statuses = sorted(results['predicted_ca_status'].unique())
                        selected_status = st.multiselect("CA Status", ca_statuses, default=ca_statuses, key="student_analysis_status")
                        results = results[results['predicted_ca_status'].isin(selected_status)]

                # Additional filters row
                col4, col5, col6 = st.columns(3)

                # Attendance range filter
                with col4:
                    if 'attendance_percentage' in results.columns:
                        att_min = float(results['attendance_percentage'].min())
                        att_max = float(results['attendance_percentage'].max())
                        att_range = st.slider("Attendance Range (%)", 
                                            min_value=att_min,
                                            max_value=att_max,
                                            value=(att_min, att_max))
                        results = results[
                            (results['attendance_percentage'] >= att_range[0]) & 
                            (results['attendance_percentage'] <= att_range[1])
                        ]

                # Academic performance filter
                with col5:
                    if 'academic_performance' in results.columns:
                        acad_min = float(results['academic_performance'].min())
                        acad_max = float(results['academic_performance'].max())
                        acad_range = st.slider("Academic Performance Range", 
                                             min_value=acad_min,
                                             max_value=acad_max,
                                             value=(acad_min, acad_max))
                        results = results[
                            (results['academic_performance'] >= acad_range[0]) & 
                            (results['academic_performance'] <= acad_range[1])
                        ]

                # Risk score filter
                with col6:
                    if 'ca_risk_score' in results.columns:
                        risk_min = float(results['ca_risk_score'].min())
                        risk_max = float(results['ca_risk_score'].max())
                        risk_range = st.slider("Risk Score Range", 
                                             min_value=risk_min,
                                             max_value=risk_max,
                                             value=(risk_min, risk_max))
                        results = results[
                            (results['ca_risk_score'] >= risk_range[0]) & 
                            (results['ca_risk_score'] <= risk_range[1])
                        ]

            # Display filtered results count
            st.info(f"Showing {len(results)} students based on selected filters")

            if "student_id" in results.columns:
                student_ids = results["student_id"].tolist()
                if student_ids:
                    selected_student = st.selectbox("Select Student ID", student_ids)
                if selected_student:
                    student_data = results[results["student_id"] == selected_student].iloc[0]
                    display_single_student_dashboard(student_data)
            else:
                st.warning("No student IDs found in prediction results")
        else:
            st.info("Please run predictions first to analyze individual students")

if __name__ == "__main__":
    main()