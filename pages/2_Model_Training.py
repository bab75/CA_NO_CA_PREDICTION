"""
Model Training Page for the CA Prediction System
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import pickle
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import roc_curve, auc

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import (
    preprocess_data, 
    train_model, 
    evaluate_model,
    get_feature_importance,
    save_model
)
from utils.visualizations import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
    plot_metric_comparison
)
from config import MODEL_CONFIG

# Set page config
st.set_page_config(
    page_title="Model Training - CA Prediction System",
    page_icon="📊",
    layout="wide"
)

# Add CSS for styling
st.markdown("""
<style>
    .performance-metrics {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .hyperparameter-container {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .model-explanation {
        margin-top: 20px;
        padding: 15px;
        background-color: #f5f7fb;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

def display_data_configuration_tab():
    """
    Display the data configuration tab for model training
    """
    st.subheader("Data Source Selection")
    
    # Radio button for data source
    data_source = st.radio(
        "Select Data Source",
        options=["Generated Historical Data", "Upload Data"],
        index=0,
        key="model_data_source"
    )
    
    # Default training data
    training_data = None
    
    if data_source == "Generated Historical Data":
        # Check if historical data exists in session state
        if "historical_data" in st.session_state:
            training_data = st.session_state["historical_data"]
            st.success(f"Using generated historical data with {len(training_data)} records")
        else:
            st.warning("No historical data found. Please generate data in the Data Preparation page first.")
            return None
    else:
        # File uploader for training data
        uploaded_file = st.file_uploader(
            "Upload training data (CSV or Excel)",
            type=["csv", "xlsx"],
            key="model_training_data_upload"
        )
        
        if uploaded_file is not None:
            try:
                # Determine file type
                file_extension = uploaded_file.name.split(".")[-1].lower()
                
                # Read the file
                if file_extension == "csv":
                    training_data = pd.read_csv(uploaded_file)
                elif file_extension == "xlsx":
                    training_data = pd.read_excel(uploaded_file)
                
                st.success(f"Successfully loaded data with {len(training_data)} records")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return None
        else:
            st.info("Please upload a data file")
            return None
    
    # Ensure we have training data
    if training_data is None:
        return None
    
    # Display data
    st.subheader("Data Preview")
    st.dataframe(training_data.head(5), use_container_width=True)
    
    # Feature selection
    st.subheader("Feature Selection")
    
    # Target variable
    # Get all column names to support multiple datasets
    all_columns = list(training_data.columns)
    
    # Categorize columns for better selection as potential targets
    binary_cols = []
    categorical_cols = []
    possible_target_cols = []
    
    # First identify columns that are most likely to be targets
    # Analyze columns to categorize by potential target type
    # Create containers for different types of potential targets
    prediction_targets = {
        "chronic_absenteeism": [],
        "dropout": [],
        "academic_risk": [],
        "behavioral": [],
        "other_binary": []
    }

    # Detailed analysis of columns for better target identification
    for col in all_columns:
        col_lower = col.lower()
        
        # Chronic Absenteeism related targets
        if any(keyword in col_lower for keyword in ["ca_status", "chronic_absenteeism", "absent", "attendance"]):
            prediction_targets["chronic_absenteeism"].append(col)
            possible_target_cols.append(col)
        
        # Dropout related targets
        elif any(keyword in col_lower for keyword in ["dropout", "retention", "persist"]):
            prediction_targets["dropout"].append(col)
            possible_target_cols.append(col)
            
        # Academic risk targets
        elif any(keyword in col_lower for keyword in ["academic", "grade", "fail", "gpa"]):
            prediction_targets["academic_risk"].append(col)
            possible_target_cols.append(col)
            
        # Behavioral targets
        elif any(keyword in col_lower for keyword in ["behavior", "conduct", "discipline"]):
            prediction_targets["behavioral"].append(col)
            possible_target_cols.append(col)
            
        # General target-related columns
        elif any(keyword in col_lower for keyword in ["target", "predict", "outcome", "result", "class", "risk"]):
            possible_target_cols.append(col)
            
        # Check if it's a binary column (good target candidate)
        elif training_data[col].nunique() == 2:
            if col not in ["gender", "special_need", "shelter", "transfer", "suspended"]:  # Skip demographic columns
                binary_cols.append(col)
                prediction_targets["other_binary"].append(col)
    
    # Prepare target options grouped by category
    target_categories = []
    for category, cols in prediction_targets.items():
        if cols:
            target_categories.append({
                "name": category.replace("_", " ").title(),
                "columns": cols
            })
    
    # Add general binary columns if found
    if binary_cols and not any(col in binary_cols for cols in prediction_targets.values() for col in cols):
        target_categories.append({
            "name": "Other Binary Columns",
            "columns": binary_cols
        })
    
    # Combine all potential targets, prioritizing the most likely ones
    target_options = possible_target_cols + [col for col in binary_cols if col not in possible_target_cols]
    
    # If we still don't have options, add all columns as potential targets
    if not target_options:
        target_options = all_columns
    
    # Initialize session state values for target column if not already set
    if "model_target_column" not in st.session_state:
        if "ca_status" in training_data.columns:
            st.session_state["model_target_column"] = "ca_status"
        elif possible_target_cols:
            st.session_state["model_target_column"] = possible_target_cols[0]
        elif binary_cols:
            st.session_state["model_target_column"] = binary_cols[0]
        elif target_options:
            st.session_state["model_target_column"] = target_options[0]
        else:
            st.session_state["model_target_column"] = all_columns[0]
    
    # Display target selection with improved UI and detailed explanations
    st.markdown("""
    <div style="background-color: #f0f7fa; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
        <h4 style="color: #1E3A8A;">What do you want to predict?</h4>
        <p>Select a target variable for your prediction model. This is a two-step process:</p>
        <ol style="margin-top: 10px; margin-left: 20px;">
            <li><strong>Target Category</strong>: First, select the general category of what you want to predict (e.g., Chronic Absenteeism, Academic Risk)</li>
            <li><strong>Specific Target</strong>: Then, choose the specific variable within that category to use as your prediction target</li>
        </ol>
        <div style="background-color: rgba(59, 130, 246, 0.1); padding: 10px; border-left: 4px solid #3B82F6; margin-top: 10px;">
            <p style="margin: 0;"><strong>How to choose:</strong></p>
            <ul style="margin-top: 5px; margin-bottom: 0;">
                <li><strong>For binary classification</strong>: Choose categorical columns like "ca_status" (values: "CA" or "No-CA")</li>
                <li><strong>For regression</strong>: Choose numerical columns like "attendance_percentage" (values: 0-100)</li>
            </ul>
            <p style="margin-top: 5px; font-style: italic; font-size: 0.9em;"><strong>IMPORTANT:</strong> For predicting Chronic Absenteeism, choose "ca_status" (categorical) as your target rather than "attendance_percentage" (numerical). The "ca_status" column contains "CA" and "No-CA" values which work better for classification models.</p>
            <p style="margin-top: 5px; color: #DC2626; font-weight: bold;">Error Warning: If you select "attendance_percentage" as your target, you may encounter the error "The least populated class in y has only 1 member" because percentage values are unique (continuous) rather than categorical.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # If we have categorized targets, show them in a more organized way
    if target_categories:
        # Two-step selection: first category, then column
        col1, col2 = st.columns([1, 2])
        
        with col1:
            category_names = [cat["name"] for cat in target_categories]
            if "Other Columns" not in category_names:
                category_names.append("Other Columns")
            
            selected_category = st.radio(
                "Target Category",
                options=category_names,
                key="target_category"
            )
        
        with col2:
            # Find columns for the selected category
            if selected_category == "Other Columns":
                category_columns = [col for col in all_columns if col not in target_options]
            else:
                category_idx = next((i for i, cat in enumerate(target_categories) if cat["name"] == selected_category), None)
                category_columns = target_categories[category_idx]["columns"] if category_idx is not None else []
            
            if not category_columns:
                category_columns = target_options
            
            # Use the value from session state for the widget
            target_column = st.selectbox(
                f"Select {selected_category} Target",
                options=category_columns,
                index=category_columns.index(st.session_state["model_target_column"]) if st.session_state["model_target_column"] in category_columns else 0,
                key="model_target_column"
            )
    else:
        # Fallback to simple dropdown if categories aren't available
        target_column = st.selectbox(
            "Select Target Column",
            options=target_options,
            index=target_options.index(st.session_state["model_target_column"]) if st.session_state["model_target_column"] in target_options else 0,
            key="model_target_column"
        )
    
    # Show the values in the selected target column
    if target_column:
        unique_values = training_data[target_column].unique()
        st.markdown(f"**Target values:** {', '.join(str(v) for v in unique_values)}")
        
        # If binary, show the distribution
        if len(unique_values) == 2:
            col1, col2 = st.columns(2)
            val_counts = training_data[target_column].value_counts()
            
            with col1:
                st.metric(f"{unique_values[0]}", f"{val_counts.iloc[0]} ({val_counts.iloc[0]/len(training_data)*100:.1f}%)")
            
            with col2:
                st.metric(f"{unique_values[1]}", f"{val_counts.iloc[1]} ({val_counts.iloc[1]/len(training_data)*100:.1f}%)")
    
    
    # Feature categories
    categorical_features = training_data.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_features = training_data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    # Remove target from features
    if target_column in categorical_features:
        categorical_features.remove(target_column)
    if target_column in numerical_features:
        numerical_features.remove(target_column)
    
    # Remove ID columns from features (they shouldn't be used for training)
    id_cols = [col for col in training_data.columns if "id" in col.lower() or "identifier" in col.lower()]
    for col in id_cols:
        if col in categorical_features:
            categorical_features.remove(col)
        if col in numerical_features:
            numerical_features.remove(col)
    
    # Initialize session state for categorical and numerical features
    if "model_selected_categorical" not in st.session_state:
        st.session_state["model_selected_categorical"] = categorical_features
    if "model_selected_numerical" not in st.session_state:
        st.session_state["model_selected_numerical"] = numerical_features
    
    # Feature selection
    st.subheader("Select Features for Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Categorical Features")
        selected_categorical = st.multiselect(
            "Select Categorical Features",
            options=categorical_features,
            default=[f for f in st.session_state["model_selected_categorical"] if f in categorical_features],
            key="model_selected_categorical"
        )
    
    with col2:
        st.write("Numerical Features")
        selected_numerical = st.multiselect(
            "Select Numerical Features",
            options=numerical_features,
            default=[f for f in st.session_state["model_selected_numerical"] if f in numerical_features],
            key="model_selected_numerical"
        )
    
    # Combine selected features
    selected_features = selected_categorical + selected_numerical
    
    if not selected_features:
        st.warning("Please select at least one feature for training")
        return None
    
    # Store training data in session state
    st.session_state["model_training_data"] = training_data
    
    return training_data

def display_model_selection_tab():
    """
    Display the model selection and training tab
    """
    # Check if training data exists in session state
    if "model_training_data" not in st.session_state:
        st.warning("Please configure data first")
        return
    
    # Get data from session state
    training_data = st.session_state["model_training_data"]
    target_column = st.session_state["model_target_column"]
    selected_categorical = st.session_state["model_selected_categorical"]
    selected_numerical = st.session_state["model_selected_numerical"]
    
    # Model selection with educational information
    #st.subheader("Model Selection")
    with st.expander("Model Selection Guide: How to use"):
        # Information about models with links
        st.markdown("""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
        <h4 style="color: #1E3A8A;">Choose Machine Learning Models</h4>
        <p>Choose machine learning models to predict chronic absenteeism. Each model has unique strengths:</p>
        <ul>
            <li><strong>Logistic Regression</strong>: Models the probability of absenteeism using a linear relationship. Best for interpretable results.
              <br><a href="https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression" target="_blank">Learn More</a></li>
            <li><strong>Random Forest</strong>: Combines multiple decision trees to improve accuracy and handle complex patterns. Robust to overfitting.
              <br><a href="https://scikit-learn.org/stable/modules/ensemble.html#random-forests" target="_blank">Learn More</a></li>
            <li><strong>Decision Tree</strong>: Splits data into branches based on feature values. Easy to interpret but may overfit.
              <br><a href="https://scikit-learn.org/stable/modules/tree.html" target="_blank">Learn More</a></li>
            <li><strong>SVM (Support Vector Machine)</strong>: Finds the optimal boundary to separate classes. Effective for non-linear data.
              <br><a href="https://scikit-learn.org/stable/modules/svm.html" target="_blank">Learn More</a></li>
            <li><strong>Gradient Boosting</strong>: Builds trees sequentially to correct errors. Powerful for predictive accuracy.
              <br><a href="https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting" target="_blank">Learn More</a></li>
            <li><strong>Neural Network</strong>: Models complex relationships with layered nodes. Suitable for large datasets but requires tuning.
              <br><a href="https://scikit-learn.org/stable/modules/neural_networks_supervised.html" target="_blank">Learn More</a></li>
        </ul>
        <p>Select multiple models to compare their performance. Use hyperparameter tuning for optimized results.
        <br>For a deeper dive, read <a href="https://towardsdatascience.com/the-7-most-common-machine-learning-models-8e8d6c0e1c5c" target="_blank">this guide on machine learning models</a>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for model selection
    col1, col2, col3 = st.columns(3)
    
    model_selections = {}
    
    with col1:
        model_selections["logistic_regression"] = st.checkbox(
            "Logistic Regression", 
            value=True,
            key="model_select_logistic_regression",
            help="Best for interpretable results and understanding feature importance"
        )
        model_selections["decision_tree"] = st.checkbox(
            "Decision Tree", 
            value=False,
            key="model_select_decision_tree",
            help="Easy to interpret but may overfit to training data"
        )
    
    with col2:
        model_selections["random_forest"] = st.checkbox(
            "Random Forest", 
            value=True,
            key="model_select_random_forest",
            help="Combines multiple decision trees for improved accuracy"
        )
        model_selections["gradient_boosting"] = st.checkbox(
            "Gradient Boosting", 
            value=False,
            key="model_select_gradient_boosting",
            help="Powerful algorithm with high predictive accuracy"
        )
    
    with col3:
        model_selections["svm"] = st.checkbox(
            "Support Vector Machine", 
            value=False,
            key="model_select_svm",
            help="Effective for non-linear data patterns"
        )
        model_selections["neural_network"] = st.checkbox(
            "Neural Network", 
            value=False,
            key="model_select_neural_network",
            help="Complex model suitable for large datasets"
        )
    
    # Check if at least one model is selected
    selected_models = [model for model, selected in model_selections.items() if selected]
    
    if not selected_models:
        st.warning("Please select at least one model for training")
        return
    
    # Verify target column has sufficient data for training
    target_values_counts = training_data[target_column].value_counts()
    if len(target_values_counts) < 2:
        st.error(f"Error: The target column '{target_column}' has only one unique value. A target must have at least two different values for classification.")
        st.markdown("""
        <div style="background-color: #FFEBEE; border-left: 5px solid #F44336; padding: 10px; margin-top: 10px;">
            <strong>How to fix this issue:</strong>
            <ul>
                <li>Choose a different target column with at least two unique values</li>
                <li>If using a numerical target like "attendance_percentage", try using "ca_status" instead</li>
                <li>Generate a larger dataset with more varied target values</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    elif any(count < 2 for count in target_values_counts.values):
        st.warning(f"Warning: The target column '{target_column}' has classes with very few samples. This may affect model performance.")
        st.markdown("""
        <div style="background-color: #FFF8E1; border-left: 5px solid #FFC107; padding: 10px; margin-top: 10px;">
            <strong>Recommended actions:</strong>
            <ul>
                <li>Generate more balanced data or choose a different target column</li>
                <li>Use a stratified sampling approach (automatically applied by the system)</li>
                <li>Consider using a different evaluation metric that works well with imbalanced classes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Hyperparameter tuning
    st.subheader("Hyperparameter Tuning")
    
    with st.expander("Advanced Hyperparameter Settings"):
        hyperparameters = {}
        
        # Logistic Regression hyperparameters
        if model_selections["logistic_regression"]:
            st.markdown("### Logistic Regression")
            col1, col2 = st.columns(2)
            
            with col1:
                C = st.select_slider(
                    "Regularization strength (C)",
                    options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    value=1.0,
                    key="lr_C"
                )
            
            with col2:
                solver = st.selectbox(
                    "Solver",
                    options=["liblinear", "lbfgs"],
                    index=0,
                    key="lr_solver"
                )
            
            hyperparameters["logistic_regression"] = {
                "classifier__C": C,
                "classifier__solver": solver,
                "classifier__penalty": "l2"
            }
        
        # Random Forest hyperparameters
        if model_selections["random_forest"]:
            st.markdown("### Random Forest")
            col1, col2 = st.columns(2)
            
            with col1:
                n_estimators = st.slider(
                    "Number of trees",
                    min_value=50,
                    max_value=500,
                    value=100,
                    step=50,
                    key="rf_n_estimators"
                )
                
                min_samples_split = st.slider(
                    "Minimum samples to split",
                    min_value=2,
                    max_value=10,
                    value=2,
                    key="rf_min_samples_split"
                )
            
            with col2:
                max_depth = st.slider(
                    "Maximum depth",
                    min_value=5,
                    max_value=30,
                    value=None,
                    key="rf_max_depth"
                )
                
                min_samples_leaf = st.slider(
                    "Minimum samples per leaf",
                    min_value=1,
                    max_value=10,
                    value=1,
                    key="rf_min_samples_leaf"
                )
            
            hyperparameters["random_forest"] = {
                "classifier__n_estimators": n_estimators,
                "classifier__max_depth": max_depth,
                "classifier__min_samples_split": min_samples_split,
                "classifier__min_samples_leaf": min_samples_leaf
            }
        
        # Decision Tree hyperparameters
        if model_selections["decision_tree"]:
            st.markdown("### Decision Tree")
            col1, col2 = st.columns(2)
            
            with col1:
                max_depth = st.slider(
                    "Maximum depth",
                    min_value=5,
                    max_value=30,
                    value=None,
                    key="dt_max_depth"
                )
                
                min_samples_split = st.slider(
                    "Minimum samples to split",
                    min_value=2,
                    max_value=10,
                    value=2,
                    key="dt_min_samples_split"
                )
            
            with col2:
                min_samples_leaf = st.slider(
                    "Minimum samples per leaf",
                    min_value=1,
                    max_value=10,
                    value=1,
                    key="dt_min_samples_leaf"
                )
                
                criterion = st.selectbox(
                    "Criterion",
                    options=["gini", "entropy"],
                    index=0,
                    key="dt_criterion"
                )
            
            hyperparameters["decision_tree"] = {
                "classifier__max_depth": max_depth,
                "classifier__min_samples_split": min_samples_split,
                "classifier__min_samples_leaf": min_samples_leaf,
                "classifier__criterion": criterion
            }
        
        # SVM hyperparameters
        if model_selections["svm"]:
            st.markdown("### Support Vector Machine")
            col1, col2 = st.columns(2)
            
            with col1:
                C = st.select_slider(
                    "Regularization parameter (C)",
                    options=[0.1, 1.0, 10.0, 100.0],
                    value=1.0,
                    key="svm_C"
                )
                
                kernel = st.selectbox(
                    "Kernel",
                    options=["linear", "rbf", "poly"],
                    index=1,
                    key="svm_kernel"
                )
            
            with col2:
                gamma = st.selectbox(
                    "Kernel coefficient (gamma)",
                    options=["scale", "auto"],
                    index=0,
                    key="svm_gamma"
                )
            
            hyperparameters["svm"] = {
                "classifier__C": C,
                "classifier__kernel": kernel,
                "classifier__gamma": gamma
            }
        
        # Gradient Boosting hyperparameters
        if model_selections["gradient_boosting"]:
            st.markdown("### Gradient Boosting")
            col1, col2 = st.columns(2)
            
            with col1:
                n_estimators = st.slider(
                    "Number of boosting stages",
                    min_value=50,
                    max_value=500,
                    value=100,
                    step=50,
                    key="gb_n_estimators"
                )
                
                max_depth = st.slider(
                    "Maximum depth",
                    min_value=3,
                    max_value=10,
                    value=3,
                    key="gb_max_depth"
                )
            
            with col2:
                learning_rate = st.select_slider(
                    "Learning rate",
                    options=[0.01, 0.05, 0.1, 0.2],
                    value=0.1,
                    key="gb_learning_rate"
                )
                
                subsample = st.slider(
                    "Subsample ratio",
                    min_value=0.5,
                    max_value=1.0,
                    value=1.0,
                    step=0.1,
                    key="gb_subsample"
                )
            
            hyperparameters["gradient_boosting"] = {
                "classifier__n_estimators": n_estimators,
                "classifier__learning_rate": learning_rate,
                "classifier__max_depth": max_depth,
                "classifier__subsample": subsample
            }
        
        # Neural Network hyperparameters
        if model_selections["neural_network"]:
            st.markdown("### Neural Network")
            col1, col2 = st.columns(2)
            
            with col1:
                hidden_layer_size = st.selectbox(
                    "Hidden layer size",
                    options=["(50,)", "(100,)", "(50, 50)", "(100, 50)", "(100, 100)"],
                    index=1,
                    key="nn_hidden_layer_size"
                )
                
                activation = st.selectbox(
                    "Activation function",
                    options=["relu", "tanh"],
                    index=0,
                    key="nn_activation"
                )
            
            with col2:
                alpha = st.select_slider(
                    "L2 regularization parameter",
                    options=[0.0001, 0.001, 0.01],
                    value=0.0001,
                    key="nn_alpha"
                )
                
                learning_rate = st.selectbox(
                    "Learning rate",
                    options=["constant", "adaptive"],
                    index=1,
                    key="nn_learning_rate"
                )
            
            # Convert string representation of tuple to actual tuple
            hidden_layer_size_eval = eval(hidden_layer_size)
            
            hyperparameters["neural_network"] = {
                "classifier__hidden_layer_sizes": hidden_layer_size_eval,
                "classifier__activation": activation,
                "classifier__alpha": alpha,
                "classifier__learning_rate": learning_rate
            }
    
    # Training configuration
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider(
            "Test set size (%)",
            min_value=10,
            max_value=40,
            value=20,
            key="test_size"
        ) / 100
        
        cv_folds = st.slider(
            "Cross-validation folds",
            min_value=3,
            max_value=10,
            value=5,
            key="cv_folds"
        )
    
    with col2:
        evaluation_metric = st.selectbox(
            "Primary evaluation metric",
            options=["accuracy", "precision", "recall", "f1", "roc_auc"],
            index=3,  # Default to F1
            key="evaluation_metric"
        )
    
    # Add space between the options and training button
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Training section with animation
    st.markdown("""
    <style>
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.03); }
        100% { transform: scale(1); }
    }
    .training-ready-container {
        background: linear-gradient(to right, #e8f4f8, #d0e8f2);
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
        margin-bottom: 15px;
        border-left: 4px solid #4b9fd5;
        transition: all 0.3s ease;
    }
    .training-ready-container:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .animated-pulse {
        animation: pulse 1.5s infinite ease-in-out;
    }
    .train-button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 15px;
        border: none;
        border-radius: 5px;
        transition: all 0.3s;
    }
    .train-button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    </style>
    <div class="training-ready-container">
        <h3>Ready to Start Training?</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Checkbox for training readiness (placed above button)
    ready_to_train = st.checkbox(
        "✓ I'm ready to train the selected models",
        value=False,
        key="ready_to_train",
        help="Check this box to enable the training button"
    )
    
    # Training button - add animation when ready
    if ready_to_train:
        st.markdown("""
        <style>
        .train-button-container {
            margin-top: 15px;
            text-align: center;
        }
        </style>
        <div class="train-button-container animated-pulse">
            Ready to build your predictive model!
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Train Selected Models", key="train_button"):
            # Store training parameters in session state
            st.session_state["model_hyperparameters"] = hyperparameters
            st.session_state["model_test_size"] = test_size
            st.session_state["model_cv_folds"] = cv_folds
            st.session_state["model_evaluation_metric"] = evaluation_metric
            
            # Train models
            train_models(
                training_data, 
                target_column, 
                selected_categorical, 
                selected_numerical, 
                selected_models, 
                hyperparameters, 
                test_size, 
                cv_folds, 
                evaluation_metric
            )
    else:
        st.info("✓ Check 'I'm ready to train the selected models' above to enable the training button")

def train_models(
    data, 
    target_col, 
    categorical_cols, 
    numerical_cols, 
    models, 
    hyperparameters, 
    test_size, 
    cv_folds, 
    metric
):
    """
    Train selected models with given parameters
    
    Args:
        data (pd.DataFrame): Training data
        target_col (str): Target column name
        categorical_cols (list): Categorical feature columns
        numerical_cols (list): Numerical feature columns
        models (list): List of model names to train
        hyperparameters (dict): Hyperparameters for each model
        test_size (float): Test set size proportion
        cv_folds (int): Number of cross-validation folds
        metric (str): Evaluation metric
    """
    # Show progress message
    progress_placeholder = st.empty()
    progress_placeholder.info("Preprocessing data...")
    
    try:
        # Preprocess data
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
            data,
            target_col=target_col,
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols,
            test_size=test_size,
            random_state=MODEL_CONFIG["default_random_state"]
        )
        
        # Initialize dictionaries to store results
        trained_models = {}
        model_metrics = {}
        feature_importances = {}
        training_times = {}
        
        # Total number of models to train
        total_models = len(models)
        
        # Train each selected model
        for i, model_name in enumerate(models):
            model_display_name = MODEL_CONFIG["models"][model_name]
            progress_placeholder.info(f"Training {model_display_name} ({i+1}/{total_models})...")
            
            # Get hyperparameters for this model
            model_hyperparams = hyperparameters.get(model_name, {})
            
            # Train the model
            pipeline, training_time = train_model(
                X_train,
                y_train,
                preprocessor,
                model_name,
                hyperparams=model_hyperparams,
                cv=cv_folds,
                metric=metric
            )
            
            # Evaluate the model
            metrics = evaluate_model(pipeline, X_test, y_test)
            
            # Get feature importance if available
            feature_names = categorical_cols + numerical_cols
            importance = get_feature_importance(pipeline, feature_names)
            
            # Store results
            trained_models[model_name] = pipeline
            model_metrics[model_name] = metrics
            feature_importances[model_name] = importance
            training_times[model_name] = training_time
            
            # Save the model
            model_path = save_model(pipeline, model_name, metrics, importance)
            st.session_state[f"model_path_{model_name}"] = model_path
        
        # Store results in session state
        st.session_state["trained_models"] = trained_models
        st.session_state["model_metrics"] = model_metrics
        st.session_state["feature_importances"] = feature_importances
        st.session_state["training_times"] = training_times
        st.session_state["model_X_test"] = X_test
        st.session_state["model_y_test"] = y_test
        
        # Show success message with enhanced visual effects
        progress_placeholder.success("Training completed successfully!")
        
        # Display success celebration animation with balloons
        st.balloons()
        
        # Add a confetti animation overlay
        st.markdown("""
        <style>
        @keyframes confetti-fall {
            0% { transform: translateY(-100vh) rotate(0deg); opacity: 1; }
            100% { transform: translateY(100vh) rotate(360deg); opacity: 0; }
        }
        
        .confetti {
            position: fixed;
            width: 10px;
            height: 10px;
            background-color: #f00;
            opacity: 0;
            z-index: 9999;
            animation: confetti-fall 5s linear forwards;
        }
        
        .success-banner {
            background: linear-gradient(135deg, #4CAF50, #2E7D32);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin: 20px 0;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.03); }
            100% { transform: scale(1); }
        }
        </style>
        
        <div class="success-banner">
            <h2>🎉 Model Training Complete! 🎉</h2>
            <p>Your models have been successfully trained and are ready to make predictions.</p>
        </div>
        
        <script>
        function createConfetti() {
            const colors = ['#f00', '#0f0', '#00f', '#ff0', '#f0f', '#0ff'];
            const confettiCount = 100;
            
            for (let i = 0; i < confettiCount; i++) {
                setTimeout(() => {
                    const confetti = document.createElement('div');
                    confetti.classList.add('confetti');
                    confetti.style.left = Math.random() * 100 + 'vw';
                    confetti.style.width = (Math.random() * 10 + 5) + 'px';
                    confetti.style.height = (Math.random() * 10 + 5) + 'px';
                    confetti.style.background = colors[Math.floor(Math.random() * colors.length)];
                    confetti.style.animationDuration = (Math.random() * 3 + 2) + 's';
                    
                    document.body.appendChild(confetti);
                    
                    setTimeout(() => {
                        confetti.remove();
                    }, 5000);
                }, i * 50);
            }
        }
        
        // Call the function when the page loads
        createConfetti();
        </script>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        progress_placeholder.error(f"Error during training: {str(e)}")
        st.error(f"Details: {str(e)}")

def display_training_results_tab():
    """
    Display the training results tab
    """
    # Check if models have been trained
    if "trained_models" not in st.session_state:
        st.info("No trained models found. Please train models first.")
        return
        
    # Initialize session state variables if they don't exist
    required_keys = ["model_metrics", "feature_importances", "training_times", "model_X_test", "model_y_test"]
    for key in required_keys:
        if key not in st.session_state:
            st.session_state[key] = {}
    
    # Get results from session state
    trained_models = st.session_state["trained_models"]
    model_metrics = st.session_state["model_metrics"]
    feature_importances = st.session_state["feature_importances"]
    training_times = st.session_state["training_times"]
    X_test = st.session_state["model_X_test"]
    y_test = st.session_state["model_y_test"]
    
    # If model_metrics is empty but trained_models exists, show appropriate message
    if not model_metrics and trained_models:
        st.warning("Model metrics are not available. Please try retraining the models.")
        return
    
    # Display performance metrics
    st.subheader("Model Performance Comparison")
    
    # Determine if we're dealing with classification or regression metrics
    first_metrics = next(iter(model_metrics.values())) if model_metrics else {}
    is_regression = "r2_score" in first_metrics
    
    # Create a summary table of metrics based on model type
    if is_regression:
        # Regression metrics table
        metrics_df = pd.DataFrame({
            "Model": [MODEL_CONFIG["models"][model] for model in model_metrics.keys()],
            "R² Score": [metrics.get("r2_score", 0) for metrics in model_metrics.values()],
            "RMSE": [metrics.get("root_mean_squared_error", 0) for metrics in model_metrics.values()],
            "MAE": [metrics.get("mean_absolute_error", 0) for metrics in model_metrics.values()],
            "Explained Var": [metrics.get("explained_variance", 0) for metrics in model_metrics.values()],
            "Training Time (s)": [round(time, 2) for time in training_times.values()]
        })
        
        # Format the regression metrics
        for col in ["R² Score", "Explained Var"]:
            metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
        
        for col in ["RMSE", "MAE"]:
            metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
    else:
        # Classification metrics table
        metrics_df = pd.DataFrame({
            "Model": [MODEL_CONFIG["models"][model] for model in model_metrics.keys()],
            "Accuracy": [metrics.get("accuracy", 0) for metrics in model_metrics.values()],
            "Precision": [metrics.get("precision", 0) for metrics in model_metrics.values()],
            "Recall": [metrics.get("recall", 0) for metrics in model_metrics.values()],
            "F1 Score": [metrics.get("f1", 0) for metrics in model_metrics.values()],
            "ROC AUC": [metrics.get("roc_auc", 0) for metrics in model_metrics.values()],
            "Training Time (s)": [round(time, 2) for time in training_times.values()]
        })
        
        # Format the classification metrics as percentages
        for col in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]:
            metrics_df[col] = metrics_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
    
    # Display the metrics table
    st.dataframe(metrics_df, use_container_width=True)
    
    # Plot metric comparison
    st.subheader("Metrics Comparison Chart")
    
    # Select metric to compare based on model type
    if is_regression:
        metric_to_compare = st.selectbox(
            "Select Metric to Compare",
            options=["r2_score", "root_mean_squared_error", "mean_absolute_error", "explained_variance"],
            index=0,  # Default to R²
            key="compare_metric"
        )
    else:
        metric_to_compare = st.selectbox(
            "Select Metric to Compare",
            options=["accuracy", "precision", "recall", "f1", "roc_auc"],
            index=3,  # Default to F1
            key="compare_metric"
        )
    
    # Plot comparison chart
    comparison_fig = plot_metric_comparison(model_metrics, metric_to_compare)
    st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Select a model to view detailed results
    st.subheader("Detailed Model Results")
    
    selected_model = st.selectbox(
        "Select Model",
        options=list(trained_models.keys()),
        format_func=lambda x: MODEL_CONFIG["models"][x],
        key="selected_model_details"
    )
    
    # Get metrics for the selected model
    metrics = model_metrics[selected_model]
    
    # Display detailed metrics based on model type
    cols = st.columns(5)
    
    # Check if regression or classification metrics
    if "r2_score" in metrics:
        # Regression metrics
        cols[0].metric("R² Score", f"{metrics.get('r2_score', 0):.4f}")
        cols[1].metric("RMSE", f"{metrics.get('root_mean_squared_error', 0):.4f}")
        cols[2].metric("MAE", f"{metrics.get('mean_absolute_error', 0):.4f}")
        cols[3].metric("Explained Var", f"{metrics.get('explained_variance', 0):.4f}" if "explained_variance" in metrics else "N/A")
        cols[4].metric("Training Time", f"{training_times.get(selected_model, 0):.2f}s")
    else:
        # Classification metrics
        cols[0].metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
        cols[1].metric("Precision", f"{metrics.get('precision', 0)*100:.2f}%")
        cols[2].metric("Recall", f"{metrics.get('recall', 0)*100:.2f}%")
        cols[3].metric("F1 Score", f"{metrics.get('f1', 0)*100:.2f}%")
        cols[4].metric("ROC AUC", f"{metrics.get('roc_auc', 0)*100:.2f}%" if "roc_auc" in metrics else "N/A")
    
    # Create columns for visualizations
    col1, col2 = st.columns(2)
    
    # Check if regression or classification
    is_regression = "r2_score" in metrics
    
    with col1:
        if is_regression:
            # For regression, show actual vs predicted scatter plot
            st.subheader("Actual vs Predicted Values")
            
            # Get the pipeline and predictions
            pipeline = trained_models[selected_model]
            y_pred = pipeline.predict(X_test)
            
            # Create a dataframe with actual and predicted values
            pred_df = pd.DataFrame({
                "Actual": y_test,
                "Predicted": y_pred
            })
            
            # Create scatter plot with plotly
            import plotly.express as px
            fig = px.scatter(pred_df, x="Actual", y="Predicted", 
                           title="Actual vs Predicted Values",
                           labels={"Actual": "Actual Values", "Predicted": "Predicted Values"},
                           opacity=0.7)
            
            # Add perfect prediction line
            min_val = min(pred_df["Actual"].min(), pred_df["Predicted"].min())
            max_val = max(pred_df["Actual"].max(), pred_df["Predicted"].max())
            fig.add_scatter(x=[min_val, max_val], y=[min_val, max_val], 
                          mode="lines", name="Perfect Prediction", 
                          line=dict(color="red", dash="dash"))
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # For classification, show confusion matrix
            st.subheader("Confusion Matrix")
            conf_matrix = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
            conf_matrix_fig = plot_confusion_matrix(conf_matrix)
            st.plotly_chart(conf_matrix_fig, use_container_width=True, key="conf_matrix")
    
    with col2:
        # Plot feature importance if available
        st.subheader("Feature Importance")
        importance = feature_importances.get(selected_model)
        
        if importance:
            importance_fig = plot_feature_importance(importance)
            st.plotly_chart(importance_fig, use_container_width=True, key="feature_importance")
        else:
            st.info("Feature importance not available for this model")
    
    # Plot ROC curve if available
    if "roc_auc" in metrics:
        st.subheader("ROC Curve")
        
        # Get the pipeline
        pipeline = trained_models[selected_model]
        
        # Make probability predictions
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            
            # Calculate ROC curve points
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            roc_fig = plot_roc_curve(fpr, tpr, roc_auc)
            st.plotly_chart(roc_fig, use_container_width=True)
        except:
            st.info("ROC curve could not be generated for this model")
    
    # Pattern Discovery section
    st.subheader("Pattern Discovery")
    
    # Get the pipeline
    pipeline = trained_models[selected_model]
    
    # Make predictions on test set
    y_pred = pipeline.predict(X_test)
    
    # Get original feature names
    feature_names = st.session_state["model_selected_categorical"] + st.session_state["model_selected_numerical"]
    
    # Display examples section
    with st.expander("View Example Patterns"):
        if is_regression:
            # For regression, show examples with different error ranges
            
            # Calculate absolute error
            errors = np.abs(y_test - y_pred)
            
            # Create a dataframe with predictions and errors
            pred_error_df = pd.DataFrame({
                "Actual": y_test.values,
                "Predicted": y_pred,
                "Error": errors
            })
            
            # Sort by error magnitude
            pred_error_df = pred_error_df.sort_values("Error")
            
            # Display low, medium and high error examples
            col1, col2 = st.columns(2)
            
            # Make sure we have data before proceeding
            if len(pred_error_df) > 0:
                # Calculate number of examples to show (up to 5)
                num_examples = min(5, len(pred_error_df))
                
                with col1:
                    st.markdown("### Best Predictions (Lowest Error)")
                    if num_examples > 0:
                        try:
                            best_df = X_test.iloc[pred_error_df.index[:num_examples]].reset_index(drop=True)
                            best_df["Actual"] = pred_error_df["Actual"].values[:num_examples]
                            best_df["Predicted"] = pred_error_df["Predicted"].values[:num_examples]
                            best_df["Error"] = pred_error_df["Error"].values[:num_examples]
                            st.dataframe(best_df, use_container_width=True)
                        except IndexError:
                            st.warning("Unable to display best predictions - index out of bounds.")
                    else:
                        st.info("No prediction data available to display.")
                
                with col2:
                    st.markdown("### Worst Predictions (Highest Error)")
                    if num_examples > 0:
                        try:
                            worst_indices = pred_error_df.index[-num_examples:] if num_examples > 0 else []
                            worst_df = X_test.iloc[worst_indices].reset_index(drop=True)
                            worst_df["Actual"] = pred_error_df["Actual"].values[-num_examples:]
                            worst_df["Predicted"] = pred_error_df["Predicted"].values[-num_examples:]
                            worst_df["Error"] = pred_error_df["Error"].values[-num_examples:]
                            st.dataframe(worst_df, use_container_width=True)
                        except IndexError:
                            st.warning("Unable to display worst predictions - index out of bounds.")
                    else:
                        st.info("No prediction data available to display.")
            else:
                st.info("No prediction data available to display examples.")
            
            # Show distribution of errors
            st.markdown("### Error Distribution")
            error_fig = px.histogram(pred_error_df, x="Error", nbins=20,
                                   title="Distribution of Prediction Errors",
                                   labels={"Error": "Absolute Error"})
            st.plotly_chart(error_fig, use_container_width=True)
            
        else:
            # For classification, show examples of correct and incorrect classifications
            
            # Get indices of correct and incorrect predictions
            correct_indices = np.where(y_pred == y_test)[0]
            incorrect_indices = np.where(y_pred != y_test)[0]
            
            # Check if there are any correct/incorrect predictions
            col1, col2 = st.columns(2)
            
            with col1:
                if len(correct_indices) > 0:
                    st.markdown("### Correct Classifications")
                    # Sample a few correct predictions
                    try:
                        sample_correct = np.random.choice(correct_indices, min(5, len(correct_indices)), replace=False)
                        
                        # Get the original data (before preprocessing)
                        orig_X_test = X_test.iloc[sample_correct].reset_index(drop=True)
                        orig_y_test = y_test.iloc[sample_correct].reset_index(drop=True)
                        
                        # Create a dataframe with the sample
                        sample_df = orig_X_test.copy()
                        sample_df["Actual"] = orig_y_test
                        sample_df["Predicted"] = y_pred[sample_correct]
                        
                        # Display the sample
                        st.dataframe(sample_df, use_container_width=True)
                    except (IndexError, ValueError) as e:
                        st.warning(f"Unable to display correct classifications: {str(e)}")
                else:
                    st.info("No correct classifications found in the test set")
            
            with col2:
                if len(incorrect_indices) > 0:
                    st.markdown("### Incorrect Classifications")
                    # Sample a few incorrect predictions
                    try:
                        sample_incorrect = np.random.choice(incorrect_indices, min(5, len(incorrect_indices)), replace=False)
                        
                        # Get the original data (before preprocessing)
                        orig_X_test = X_test.iloc[sample_incorrect].reset_index(drop=True)
                        orig_y_test = y_test.iloc[sample_incorrect].reset_index(drop=True)
                        
                        # Create a dataframe with the sample
                        sample_df = orig_X_test.copy()
                        sample_df["Actual"] = orig_y_test
                        sample_df["Predicted"] = y_pred[sample_incorrect]
                        
                        # Display the sample
                        st.dataframe(sample_df, use_container_width=True)
                    except (IndexError, ValueError) as e:
                        st.warning(f"Unable to display incorrect classifications: {str(e)}")
                else:
                    st.info("No incorrect classifications found in the test set")
    
    # Save as best model
    st.subheader("Save as Best Model")
    
    if st.button("Set as Current Best Model for Predictions"):
        # Store the selected model as the best model
        st.session_state["best_model"] = selected_model
        st.session_state["best_model_pipeline"] = trained_models[selected_model]
        st.session_state["best_model_metrics"] = metrics
        st.session_state["best_model_feature_importance"] = feature_importances.get(selected_model)
        
        st.success(f"Set {MODEL_CONFIG['models'][selected_model]} as the current best model for predictions!")

def display_documentation_tab():
    """
    Display the documentation tab
    """
    st.subheader("Model Training Documentation")
    
    st.markdown("""
    ## Model Training Module Documentation

    This module allows you to train and evaluate machine learning models for predicting Chronic Absenteeism (CA) based on student data.

    ### Workflow

    1. **Data Configuration**
        - Select the data source (generated historical data or upload your own)
        - Preview the data and select features for training
        - Choose the target column (usually "ca_status")

    2. **Model Selection & Training**
        - Select one or more machine learning models to train
        - Adjust hyperparameters for each selected model (optional)
        - Configure training parameters like test set size and evaluation metric
        - Train the selected models on the data

    3. **Training Results**
        - View performance metrics for all trained models
        - Compare models using various metrics (accuracy, precision, recall, F1 score, ROC AUC)
        - Explore detailed results for individual models:
            - Confusion matrix
            - Feature importance
            - ROC curve
            - Example patterns
        - Set the best performing model for making predictions

    ### Available Models

    1. **Logistic Regression**
        - Simple linear model for binary classification
        - Fast to train and interpret
        - Works well for linearly separable data

    2. **Random Forest**
        - Ensemble of decision trees
        - Handles non-linear relationships well
        - Robust to overfitting

    3. **Decision Tree**
        - Simple tree-based model
        - Highly interpretable
        - Can capture non-linear patterns

    4. **Support Vector Machine (SVM)**
        - Creates a hyperplane to separate classes
        - Works well in high-dimensional spaces
        - Effective when classes are separable

    5. **Gradient Boosting**
        - Ensemble method that builds trees sequentially
        - Often achieves high accuracy
        - Can be more prone to overfitting

    6. **Neural Network**
        - Multi-layer perceptron classifier
        - Can capture complex patterns
        - Requires more data and tuning

    ### Performance Metrics
    
    #### Classification Metrics (for categorical targets like "CA Status")
    - **Accuracy**: Overall correctness of predictions
    - **Precision**: Proportion of positive identifications that were actually correct
    - **Recall**: Proportion of actual positives that were identified correctly
    - **F1 Score**: Harmonic mean of precision and recall
    - **ROC AUC**: Area under the ROC curve, measures the model's ability to discriminate between classes
    
    #### Regression Metrics (for continuous numeric targets)
    - **R² Score**: Proportion of variance in the dependent variable predictable from the independent variables
    - **RMSE (Root Mean Squared Error)**: Square root of the average of squared differences between predicted and actual values
    - **MAE (Mean Absolute Error)**: Average of absolute differences between predicted and actual values
    - **Explained Variance**: Proportion of variance in the dependent variable that is explained by the model

    ### Tips for Model Training

    - Include a balanced set of features for optimal performance
    - Compare multiple models to find the best one for your data
    - Consider the trade-off between precision and recall based on your goals
    - Experiment with different hyperparameters to improve performance
    - Look at feature importance to understand key factors affecting CA
    """)

def reset_session_state():
    """
    Reset all model training related session state variables
    """
    keys_to_clear = [
        "model_training_data", "model_target_column", "model_selected_categorical", 
        "model_selected_numerical", "model_metrics", "model_pipelines", "feature_importances",
        "best_model", "best_model_pipeline", "best_model_metrics", "training_results"
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    # Reset all model selection checkboxes
    for model in ["logistic_regression", "random_forest", "decision_tree", 
                 "svm", "gradient_boosting", "neural_network"]:
        if f"model_select_{model}" in st.session_state:
            del st.session_state[f"model_select_{model}"]
            
    # Also clear any prediction results
    if "prediction_results" in st.session_state:
        del st.session_state["prediction_results"]
    
    return True

def main():
    # Custom CSS for animations and styling
    st.markdown("""
    <style>
        @keyframes gradient {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        
        .banner {
            background: linear-gradient(-45deg, #3B82F6, #2563EB, #1E40AF, #1E3A8A);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        
        .banner h1 {
            font-size: 2.5rem;
            margin: 0;
            padding: 0;
        }
        
        .banner p {
            font-size: 1.1rem;
            margin-top: 10px;
            opacity: 0.9;
        }
        
        .icon-container {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .pulse {
            animation: pulse 2s infinite;
            border-radius: 50%;
            margin-right: 15px;
        }
        
        @keyframes pulse {
            0% {
                box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.4);
            }
            70% {
                box-shadow: 0 0 0 10px rgba(255, 255, 255, 0);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(255, 255, 255, 0);
            }
        }
        
        .tab-content {
            animation: fadeIn 0.6s ease-in-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(90deg, #3B82F6, #2563EB);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .stButton > button:hover {
            background: linear-gradient(90deg, #2563EB, #1E40AF);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Animated header banner
    st.markdown("""
    <div class="banner">
        <div class="icon-container">
            <div class="pulse">
                <img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjQwIiBoZWlnaHQ9IjQwIiBmaWxsPSJ3aGl0ZSI+PHBhdGggZD0iTTE1IDUuNWMxLjEgMCAyLS45IDItMnMtLjktMi0yLTItMiAuOS0yIDIgLjkgMiAyIDJ6Ii8+PHBhdGggZD0iTTEyIDIwLjVjLTEuMSAwLTIgLjktMiAycy45IDIgMiAyIDItLjkgMi0yLS45LTItMi0yeiIvPjxwYXRoIGQ9Ik0xOCAxMmMtLjkgMC0xLjcuNC0yLjIgMWwtMi41LTEuNGMuMS0uMy4yLS42LjItMSAwLS40LS4xLS43LS4yLTFsMi41LTEuNGMuNS42IDEuMyAxIDIuMiAxIDEuNyAwIDMtMS4zIDMtM3MtMS4zLTMtMy0zLTMgMS4zLTMgM2MwIC40LjEuNy4yIDFsLTIuNSAxLjRjLS41LS42LTEuMy0xLTIuMi0xLTEuNyAwLTMgMS4zLTMgM3MxLjMgMyAzIDNjLjkgMCAxLjctLjQgMi4yLTFsMi41IDEuNGMtLjEuMy0uMi42LS4yIDEgMCAxLjcgMS4zIDMgMyAzczMtMS4zIDMtMy0xLjMtMy0zLTN6Ii8+PC9zdmc+" width="40" height="40">
            </div>
            <h1>Model Training & Selection</h1>
        </div>
        <p>Train, compare, and evaluate machine learning models to predict chronic absenteeism patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load icon (just for reference in sidebar but use our banner for main display)
    st.image("assets/model_icon.svg", width=0)
    
    # Add a sidebar reset button
    with st.sidebar:
        st.subheader("Reset Training")
        if st.button("🔄 Reset All Training Data", help="Clear all training data, models, and results"):
            reset = reset_session_state()
            if reset:
                st.success("All training data and models have been reset!")
                
        st.markdown("---")
        
        # System status
        st.subheader("System Status")
        
        # Check if models are trained
        if "model_metrics" in st.session_state and st.session_state["model_metrics"]:
            models_count = len(st.session_state["model_metrics"])
            st.success(f"✅ {models_count} models trained")
        else:
            st.warning("⚠️ No models trained")
        
        # Check if data is loaded
        if "model_training_data" in st.session_state:
            data_shape = st.session_state["model_training_data"].shape
            st.info(f"📊 Data loaded: {data_shape[0]} rows × {data_shape[1]} columns")
        else:
            st.warning("⚠️ No data loaded")
        
        # Check for predictions
        if "prediction_results" in st.session_state:
            pred_count = len(st.session_state["prediction_results"])
            st.success(f"✅ Predictions made for {pred_count} students")
        else:
            st.warning("⚠️ No predictions made")
    
    # Tabs for different sections
    tabs = st.tabs([
        "Data Configuration", 
        "Model Selection & Training", 
        "Training Results", 
        "Documentation"
    ])
    
    # Data Configuration tab
    with tabs[0]:
        display_data_configuration_tab()
    
    # Model Selection & Training tab
    with tabs[1]:
        display_model_selection_tab()
    
    # Training Results tab
    with tabs[2]:
        display_training_results_tab()
    
    # Documentation tab
    with tabs[3]:
        display_documentation_tab()
        
    # Add a discovered patterns section at the bottom
    if "best_model" in st.session_state and "feature_importances" in st.session_state:
        st.subheader("Discovered Patterns")
        
        st.markdown("""
        <div style="background-color: #f0f7fa; padding: 15px; border-radius: 5px; margin-top: 20px;">
            <h4 style="color: #1E3A8A;">Key Patterns Discovered</h4>
            <p>Based on the model training and feature importance analysis, the system has discovered the following patterns:</p>
        """, unsafe_allow_html=True)
        
        # Get feature importances from the best model
        best_model = st.session_state["best_model"]
        feature_importances = st.session_state["feature_importances"].get(best_model, {})
        
        if feature_importances:
            # Sort features by importance
            sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
            
            # Display top patterns
            st.markdown("<ul>", unsafe_allow_html=True)
            for feature, importance in sorted_features[:5]:
                feature_name = feature.replace("_", " ").title()
                st.markdown(f"""
                <li><strong>{feature_name}</strong> is a significant predictor of chronic absenteeism 
                (importance score: {importance:.4f})</li>
                """, unsafe_allow_html=True)
            
            # Add some potential insights based on common features
            common_insights = {
                "academic_performance": "Students with lower academic performance are more likely to have chronic absenteeism",
                "grade": "There appears to be a grade-based pattern in chronic absenteeism rates",
                "special_need": "Special needs status has a significant impact on attendance patterns",
                "bus_long_trip": "Long bus trips correlate with higher absenteeism rates",
                "meal_code": "Socioeconomic factors (indicated by meal codes) influence attendance patterns",
                "gender": "There are gender-based differences in attendance patterns",
                "shelter": "Housing status shows a correlation with attendance rates"
            }
            
            # Display additional insights based on important features
            for feature, _ in sorted_features[:5]:
                if feature in common_insights:
                    st.markdown(f"""
                    <li><strong>Insight:</strong> {common_insights[feature]}</li>
                    """, unsafe_allow_html=True)
            
            st.markdown("</ul>", unsafe_allow_html=True)
        else:
            st.info("No pattern data available. Please train models to discover patterns.")
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
