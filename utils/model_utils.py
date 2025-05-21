"""
Utility module for machine learning model operations in the CA Prediction System
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Import ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import joblib
import os
import sys
import pickle
import time
from datetime import datetime

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG

def get_model_instance(model_name):
    """
    Get a model instance based on the model name

    Args:
        model_name (str): Name of the model to instantiate

    Returns:
        object: Model instance
    """
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=MODEL_CONFIG["default_random_state"]),
        "random_forest": RandomForestClassifier(random_state=MODEL_CONFIG["default_random_state"]),
        "decision_tree": DecisionTreeClassifier(random_state=MODEL_CONFIG["default_random_state"]),
        "svm": SVC(probability=True, random_state=MODEL_CONFIG["default_random_state"]),
        "gradient_boosting": GradientBoostingClassifier(random_state=MODEL_CONFIG["default_random_state"]),
        "neural_network": MLPClassifier(max_iter=500, random_state=MODEL_CONFIG["default_random_state"])
    }

    return models.get(model_name)

def get_hyperparameter_grid(model_name):
    """
    Get hyperparameter grid for a specific model

    Args:
        model_name (str): Name of the model

    Returns:
        dict: Hyperparameter grid
    """
    param_grids = {
        "logistic_regression": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__solver": ["liblinear", "lbfgs"],
            "classifier__penalty": ["l2"]
        },
        "random_forest": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [None, 10, 20, 30],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4]
        },
        "decision_tree": {
            "classifier__max_depth": [None, 10, 20, 30],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__criterion": ["gini", "entropy"]
        },
        "svm": {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__kernel": ["linear", "rbf", "poly"],
            "classifier__gamma": ["scale", "auto"]
        },
        "gradient_boosting": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__learning_rate": [0.01, 0.1, 0.2],
            "classifier__max_depth": [3, 5, 7],
            "classifier__subsample": [0.8, 0.9, 1.0]
        },
        "neural_network": {
            "classifier__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "classifier__activation": ["relu", "tanh"],
            "classifier__alpha": [0.0001, 0.001, 0.01],
            "classifier__learning_rate": ["constant", "adaptive"]
        }
    }

    return param_grids.get(model_name, {})

def preprocess_data(df, target_col="ca_status", categorical_cols=None, numerical_cols=None, test_size=0.2, random_state=42):
    """
    Preprocess the dataframe and split into train/test sets with balanced feature groups

    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        categorical_cols (list): List of categorical column names
        numerical_cols (list): List of numerical column names
        test_size (float): Test set size proportion
        random_state (int): Random state for reproducibility

    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor)
    """
    # Load feature group weights from system config
    import json
    with open('config/system_config.json', 'r') as f:
        config = json.load(f)

    feature_groups = config.get('feature_groups', {})
    weights = config.get('weights', {})

    # Clone the dataframe to avoid modifying the original
    df = df.copy()

    # Ensure target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    # Process target column based on its data type and unique values
    if df[target_col].dtype == 'object':
        # For "CA" vs "No-CA" classification (or similar binary text labels)
        if len(df[target_col].unique()) == 2:
            # Get the two unique values
            unique_vals = list(df[target_col].unique())

            # Special handling for CA values - always map CA to 1 and No-CA to 0
            if "CA" in unique_vals and "No-CA" in unique_vals:
                print("Converting CA/No-CA to 1/0 for better model training")
                df[target_col] = (df[target_col] == "CA").astype(int)
            else:
                # For other binary values, convert to 0/1 (first value is positive class)
                df[target_col] = (df[target_col] == unique_vals[0]).astype(int)
        else:
            # For categorical variables with more than 2 classes
            # Try to convert to numeric if possible (for regression)
            try:
                df[target_col] = pd.to_numeric(df[target_col])
            except:
                # If not numeric, keep as categorical for classification
                # No need to convert - let the model handle it
                pass
    elif pd.api.types.is_numeric_dtype(df[target_col]):
        # For numeric targets with only 2 unique values (binary classification)
        if len(df[target_col].unique()) == 2:
            # Make sure it's 0/1
            unique_vals = sorted(df[target_col].unique())
            df[target_col] = (df[target_col] == unique_vals[1]).astype(int)
        # For continuous variables (regression) or multi-class, keep as is

    # Auto-detect column types if not provided
    if categorical_cols is None and numerical_cols is None:
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        # Remove the target column from features
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)

        # Remove any ID columns that should not be used as features
        id_cols = [col for col in df.columns if "id" in col.lower() or "identifier" in col.lower()]
        for col in id_cols:
            if col in categorical_cols:
                categorical_cols.remove(col)
            if col in numerical_cols:
                numerical_cols.remove(col)

    # Balance feature groups using weights
    for group, features in feature_groups.items():
        group_weight = weights.get(group, 1.0)
        for feature in features:
            if feature in df.columns:
                if feature in numerical_cols:
                    # Scale numerical features by group weight
                    df[feature] = df[feature] * group_weight

    # Create column transformers for preprocessing
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Split data
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Check if target variable has enough samples in each class for stratification
    # (To fix "The least populated class has only 1 member" error)
    unique_classes = np.unique(y)

    try:
        # First attempt to get class counts safely
        if pd.api.types.is_integer_dtype(y):
            # For numeric classes
            class_counts = np.bincount(y)
            min_class_count = min(class_counts) if len(class_counts) > 0 else 0
            print(f"Class distribution: {dict(zip(range(len(class_counts)), class_counts))}")
        else:
            # For string or other class types
            class_counts = y.value_counts()
            min_class_count = class_counts.min() if not class_counts.empty else 0
            print(f"Class distribution: {class_counts.to_dict()}")
    except Exception as e:
        print(f"Warning: Error calculating class distribution: {e}")
        min_class_count = 0

    # If we have multiple classes with at least 2 samples per class, use stratified split
    if len(unique_classes) > 1 and min_class_count >= 2:
        print(f"Using stratified split with {len(unique_classes)} classes (min {min_class_count} samples)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        # If we have class imbalance or single class, use regular split
        print(f"Using regular split (non-stratified) due to class imbalance or single class")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    # Log split results
    train_counts = pd.Series(y_train).value_counts()
    test_counts = pd.Series(y_test).value_counts()
    print(f"Training set class distribution: {train_counts.to_dict()}")
    print(f"Testing set class distribution: {test_counts.to_dict()}")

    return X_train, X_test, y_train, y_test, preprocessor

def train_model(X_train, y_train, preprocessor, model_name, hyperparams=None, cv=5, metric='f1'):
    """
    Train a machine learning model with optional hyperparameter tuning

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        preprocessor (ColumnTransformer): Data preprocessor
        model_name (str): Name of the model to train
        hyperparams (dict, optional): Custom hyperparameters
        cv (int): Number of cross-validation folds
        metric (str): Scoring metric for hyperparameter tuning

    Returns:
        tuple: (trained_pipeline, training_time)
    """
    # Check if we're dealing with a regression or classification problem
    unique_values = np.unique(y_train)

    if pd.api.types.is_numeric_dtype(y_train):
        # If numeric with only a few unique values, treat as classification
        if len(unique_values) <= 10:
            # Classification - use classifier models
            model_type = "classifier"
        else:
            # More than 10 unique values - treat as regression
            model_type = "regressor"
            # Change metric if needed
            if metric == 'f1':
                metric = 'r2'
    else:
        # Non-numeric target is always classification
        model_type = "classifier"

    # Get model instance
    model = get_model_instance(model_name)

    if model is None:
        raise ValueError(f"Unknown model: {model_name}")

    # Create pipeline with appropriate step name (classifier or regressor)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        (model_type, model)  # Use dynamic step name based on problem type
    ])

    start_time = time.time()

    # Initialize param_grid variable
    param_grid = None

    # Fix hyperparameter keys to match model_type
    if hyperparams:
        # Convert classifier__ to regressor__ if needed
        if model_type == "regressor":
            valid_hyperparams = {}
            for param, value in hyperparams.items():
                if param.startswith("classifier__"):
                    # Replace classifier__ with regressor__
                    new_param = param.replace("classifier__", "regressor__")
                    valid_hyperparams[new_param] = value
                else:
                    valid_hyperparams[param] = value
            hyperparams = valid_hyperparams

    # Try fitting the model with appropriate error handling
    try:
        # If hyperparams provided, use them directly
        if hyperparams:
            # Filter to include only valid hyperparams for this model
            step_prefix = f"{model_type}__"
            valid_hyperparams = {}
            for param, value in hyperparams.items():
                if param.startswith(step_prefix):
                    valid_hyperparams[param] = value

            if valid_hyperparams:
                pipeline.set_params(**valid_hyperparams)
                # Fit the model with the specified hyperparameters
                pipeline.fit(X_train, y_train)
        # Otherwise use grid search for hyperparameter tuning
        else:
            # Get hyperparameter grid and adjust param names if needed
            orig_param_grid = get_hyperparameter_grid(model_name)

            if orig_param_grid and model_type == "regressor":
                # Convert classifier__ keys to regressor__
                param_grid = {}
                for param, values in orig_param_grid.items():
                    if param.startswith("classifier__"):
                        new_param = param.replace("classifier__", "regressor__")
                        param_grid[new_param] = values
                    else:
                        param_grid[param] = values
            else:
                param_grid = orig_param_grid

            if param_grid:
                try:
                    grid_search = GridSearchCV(
                        pipeline,
                        param_grid=param_grid,
                        cv=cv,
                        scoring=metric,
                        n_jobs=-1,
                        verbose=1
                    )

                    grid_search.fit(X_train, y_train)
                    pipeline = grid_search.best_estimator_
                except Exception as e:
                    print(f"Error during grid search: {str(e)}")
                    # Fallback to direct fitting with default parameters
                    pipeline.fit(X_train, y_train)
            else:
                # If no param grid, just fit the model
                pipeline.fit(X_train, y_train)

    except Exception as e:
        error_msg = str(e)
        print(f"Error during model training: {error_msg}")

        # Special handling for different error types
        if "Unknown label type: continuous" in error_msg:
            # We're trying to use a classifier on regression data
            # Try switching to a regressor for continuous targets
            try:
                # Create a regression version of the model if possible
                if model_name == "linear_regression" or model_name == "logistic_regression":
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                elif model_name == "random_forest":
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(random_state=42)
                elif model_name == "gradient_boosting":
                    from sklearn.ensemble import GradientBoostingRegressor
                    model = GradientBoostingRegressor(random_state=42)
                elif model_name == "decision_tree":
                    from sklearn.tree import DecisionTreeRegressor
                    model = DecisionTreeRegressor(random_state=42)
                elif model_name == "neural_network":
                    # Change to regressor
                    model.activation = 'identity'  # or 'relu'
                else:
                    raise ValueError(f"Cannot convert {model_name} to a regressor")

                # Create new pipeline with regressor
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', model)
                ])

                # Fit the regression model
                pipeline.fit(X_train, y_train)
                print(f"Successfully switched to regression model for continuous target")

            except Exception as fallback_error:
                print(f"Failed to create regression model: {str(fallback_error)}")
                raise ValueError(f"Error training model. Target variable might be continuous (numeric) but you're using a classifier model. Try selecting a different target column that has categorical values (e.g., 'ca_status').") from e
        else:
            # Re-raise the original error with more context
            raise ValueError(f"Error training model: {error_msg}") from e

    # Ensure the model is fitted
    predict_method = "predict"
    if not hasattr(pipeline, predict_method) or not hasattr(pipeline.steps[-1][1], predict_method):
        raise ValueError(f"Invalid pipeline or model: {model_name}")

    training_time = time.time() - start_time

    return pipeline, training_time

def evaluate_model(pipeline, X_test, y_test):
    """
    Evaluate a trained model on test data

    Args:
        pipeline (Pipeline): Trained scikit-learn pipeline
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target

    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Determine if we're dealing with classification or regression
    is_classification = True
    is_binary = False

    # Check if pipeline has a regressor (for regression) or classifier (for classification)
    if hasattr(pipeline, 'named_steps') and 'regressor' in pipeline.named_steps:
        is_classification = False

    # Process classification metrics
    if is_classification:
        # For ROC AUC, we need probability predictions
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            has_proba = True
        except:
            has_proba = False
            y_prob = None

        # Check if we're dealing with binary or multiclass classification
        unique_classes = np.unique(y_test)
        is_binary = len(unique_classes) == 2

        try:
            # Calculate metrics with appropriate average setting for multiclass
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }

            # For binary classification
            if is_binary:
                metrics.update({
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0)
                })
            # For multiclass classification
            else:
                metrics.update({
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                })

            # ROC AUC can only be calculated for binary classification
            if has_proba and is_binary:
                metrics['roc_auc'] = roc_auc_score(y_test, y_prob)

        except Exception as e:
            # Fallback metrics if there's an error
            print(f"Error calculating classification metrics: {str(e)}")
            metrics = {
                'error': str(e),
                'accuracy': np.mean(y_pred == y_test)
            }

    # Process regression metrics
    else:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        try:
            # Calculate regression metrics
            metrics = {
                'mean_squared_error': mean_squared_error(y_test, y_pred),
                'root_mean_squared_error': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mean_absolute_error': mean_absolute_error(y_test, y_pred),
                'r2_score': r2_score(y_test, y_pred)
            }

            # Add explained variance score if available
            try:
                from sklearn.metrics import explained_variance_score
                metrics['explained_variance'] = explained_variance_score(y_test, y_pred)
            except:
                pass

        except Exception as e:
            # Fallback metrics if there's an error
            print(f"Error calculating regression metrics: {str(e)}")
            metrics = {
                'error': str(e),
                'mean_squared_error': np.mean((y_test - y_pred) ** 2)
            }

    return metrics

def get_feature_importance(pipeline, feature_names):
    """
    Extract feature importance from a trained model if available

    Args:
        pipeline (Pipeline): Trained scikit-learn pipeline
        feature_names (list): List of feature names

    Returns:
        dict: Dictionary mapping feature names to importance scores, or None if not available
    """
    # Check if it's a classifier or regressor
    is_classifier = 'classifier' in pipeline.named_steps
    is_regressor = 'regressor' in pipeline.named_steps

    # Get the model from the pipeline (classifier or regressor)
    if is_classifier:
        model = pipeline.named_steps.get('classifier')
    elif is_regressor:
        model = pipeline.named_steps.get('regressor')
    else:
        # No model found
        return None

    # Get the preprocessor from the pipeline
    preprocessor = pipeline.named_steps.get('preprocessor')

    # Check if preprocessor exists and extract transformed feature names
    if preprocessor:
        # Try to get the one-hot encoded feature names
        try:
            # Get categorical features after one-hot encoding
            onehotencoder = preprocessor.named_transformers_.get('cat').named_steps.get('onehot')
            if onehotencoder:
                cat_features = preprocessor.transformers_[1][2]  # Get categorical column names
                if hasattr(onehotencoder, 'get_feature_names_out'):
                    # For newer sklearn versions
                    cat_feature_names = onehotencoder.get_feature_names_out(cat_features)
                else:
                    # For older sklearn versions
                    try:
                        cat_feature_names = [f"{col}_{val}" for col in cat_features 
                                           for val in onehotencoder.categories_[list(cat_features).index(col)]]
                    except:
                        # If columns are already processed somehow
                        cat_feature_names = [f"cat_{i}" for i in range(len(cat_features))]

                # Get numerical feature names
                num_features = preprocessor.transformers_[0][2]  # Get numerical column names

                # Combine numerical and one-hot encoded feature names
                transformed_feature_names = list(num_features) + list(cat_feature_names)
            else:
                transformed_feature_names = feature_names
        except Exception as e:
            # Fallback to original feature names
            print(f"Error extracting feature names: {str(e)}")
            transformed_feature_names = feature_names
    else:
        transformed_feature_names = feature_names

    # Check if model has feature_importances_ or coef_ attribute
    importances = None

    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models
        if len(model.coef_.shape) > 1 and model.coef_.shape[0] > 1:
            # For multi-class models, use the mean absolute importance across all classes
            importances = np.mean(np.abs(model.coef_), axis=0)
        else:
            # For binary classification or regression
            importances = np.abs(model.coef_).flatten()

    if importances is not None:
        # Ensure the feature names and importances have the same length
        if len(transformed_feature_names) != len(importances):
            # Trim or pad as needed
            min_len = min(len(transformed_feature_names), len(importances))
            transformed_feature_names = transformed_feature_names[:min_len]
            importances = importances[:min_len]

        # Create a dictionary mapping feature names to importance scores
        feature_importance = dict(zip(transformed_feature_names, importances))

        # Sort by importance (descending)
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

        return feature_importance

    return None

def save_model(pipeline, model_name, metrics, feature_importance=None):
    """
    Save a trained model to disk

    Args:
        pipeline (Pipeline): Trained scikit-learn pipeline
        model_name (str): Name of the model
        metrics (dict): Evaluation metrics
        feature_importance (dict, optional): Feature importance scores

    Returns:
        str: Path to the saved model
    """
    # Create the models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Create a timestamp for the model filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a model info dictionary
    model_info = {
        "pipeline": pipeline,
        "model_name": model_name,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "timestamp": timestamp
    }

    # Save the model info
    model_path = f"models/{model_name}_{timestamp}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_info, f)

    return model_path

def load_model(model_path):
    """
    Load a trained model from disk

    Args:
        model_path (str): Path to the saved model

    Returns:
        dict: Model info dictionary
    """
    with open(model_path, "rb") as f:
        model_info = pickle.load(f)

    return model_info

def make_prediction(pipeline, X):
    """
    Make predictions using a trained model

    Args:
        pipeline (Pipeline): Trained scikit-learn pipeline
        X (pd.DataFrame): Features for prediction

    Returns:
        tuple: (predictions, probabilities)
    """
    # Clone the dataframe to avoid modifying the original
    X = X.copy()

    if pipeline is None:
        print("Error: Model pipeline is None")
        return np.array(["No-CA"] * len(X)), np.zeros(len(X))

    # Check if model exists and what type it is
    is_classifier = hasattr(pipeline, 'named_steps') and 'classifier' in pipeline.named_steps
    is_regressor = hasattr(pipeline, 'named_steps') and 'regressor' in pipeline.named_steps

    # Check if 'attendance_percentage' is in the dataframe
    # This is for direct CA determination based on the threshold
    has_attendance = 'attendance_percentage' in X.columns

    try:
        # Make model predictions
        predictions = pipeline.predict(X)

        # For classification models, try to get class probabilities
        if is_classifier:
            try:
                # For binary classification, get probability of positive class
                probabilities = pipeline.predict_proba(X)[:, 1]
            except Exception as e:
                print(f"Warning - Unable to get prediction probabilities: {str(e)}")
                # Use a confidence score based on prediction (1.0 for positive, 0.0 for negative)
                probabilities = np.array([1.0 if p == 1 or p == True or p == "CA" else 0.0 for p in predictions])
        # For regression models, no probabilities
        else:
            # Convert regression predictions to probabilities as best as possible
            # For attendance_percentage predictions - higher is better (less likely to be CA)
            if predictions.dtype in [np.float64, np.float32]:
                # Normalize to 0-1 range for probability-like score
                # For attendance, higher is better, so inverse the probability
                probabilities = 1.0 - (predictions / 100)  # Assuming percentage scale
            else:
                # Fallback
                probabilities = np.array([0.5] * len(predictions))

        # First get ML model predictions for future risk
        ml_predictions = predictions
        ml_probabilities = probabilities

        # Then check current CA status using attendance
        if has_attendance:
            attendance_threshold = 90.0
            current_ca_status = np.array([
                "CA" if row['attendance_percentage'] <= attendance_threshold else "No-CA" 
                for _, row in X.iterrows()
            ])

            # Hybrid approach combining attendance rules and ML predictions:
            # 1. Students with attendance ≤ 90% are automatically CA
            # 2. For others, combine attendance trend with ML prediction
            final_predictions = []
            final_probabilities = []

            for i, (att_status, ml_pred, ml_prob, att_pct) in enumerate(zip(
                current_ca_status, ml_predictions, ml_probabilities, X['attendance_percentage'])):

                if att_status == "CA":  # Current attendance ≤ 90%
                    final_predictions.append("CA")
                    final_probabilities.append(max(0.8, ml_prob))  # High confidence for current CA
                else:
                    # For non-CA students, consider both attendance trend and ML prediction
                    # Attendance weight increases as it gets closer to threshold
                    att_weight = max(0, (90 - att_pct) / 20)  # Higher weight as attendance drops
                    ml_weight = 1 - att_weight

                    # Combine predictions
                    combined_prob = (att_weight * (90 - att_pct)/10) + (ml_weight * ml_prob)
                    final_probabilities.append(combined_prob)
                    final_predictions.append("CA" if combined_prob >= 0.5 else "No-CA")

            final_predictions = np.array(final_predictions)
            probabilities = np.array(final_probabilities)

            # Add prediction reason to help with interpretability
            prediction_reasons = []
            for i, (att, pred, prob) in enumerate(zip(current_ca_status, final_predictions, probabilities)):
                if att == "CA":
                    prediction_reasons.append("Current attendance below threshold")
                else:
                    if pred == "CA":
                        # Get top risk factors
                        risk_factors = []
                        if 'academic_performance' in X.columns and X.iloc[i]['academic_performance'] < 70:
                            risk_factors.append("Low academic performance")
                        if 'behavior_incidents' in X.columns and X.iloc[i]['behavior_incidents'] > 2:
                            risk_factors.append("Behavioral concerns")
                        if 'suspension_days' in X.columns and X.iloc[i]['suspension_days'] > 0:
                            risk_factors.append("Previous suspensions")
                        prediction_reasons.append(f"Future risk factors: {', '.join(risk_factors)}")
                    else:
                        prediction_reasons.append("No significant risk factors")

            return final_predictions, probabilities, prediction_reasons

        # For model-based predictions only
        if isinstance(predictions[0], (np.integer, np.floating, np.bool_, bool, int, float)):
            string_predictions = np.array([
                "CA" if (p == 1 or p == True) else "No-CA" for p in predictions
            ])
            return string_predictions, probabilities

        # Already string format (assuming "CA"/"No-CA")
        return predictions, probabilities

    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        # Handle the error gracefully
        if len(X) > 0:
            # Log the stacktrace for better debugging
            import traceback
            traceback.print_exc()

            # Default predictions: If attendance column exists, use it directly
            if has_attendance:
                ca_threshold = 90.0
                final_predictions = np.array([
                    "CA" if row['attendance_percentage'] <= ca_threshold else "No-CA" 
                    for _, row in X.iterrows()
                ])
                probabilities = np.array([
                    0.9 if row['attendance_percentage'] <= ca_threshold else 0.1
                    for _, row in X.iterrows()
                ])

                print(f"Fallback method: Using attendance threshold of {ca_threshold}% to determine CA status")
                print(f"Found {sum(1 for p in final_predictions if p == 'CA')} CA students")

                return final_predictions, probabilities

            # No attendance column, return safe fallback
            predictions = np.array(["No-CA"] * len(X))
            probabilities = np.zeros(len(X))
            return predictions, probabilities
        else:
            # Empty input
            return np.array([]), np.array([])

    # This should never be reached, but just in case
    # Return safe values rather than None to prevent NoneType errors
    return np.array(["No-CA"] * len(X)), np.zeros(len(X))

def analyze_predictions(result_data):
    """
    Analyze predictions to count and categorize CA students

    Args:
        result_data (pd.DataFrame): DataFrame containing prediction results

    Returns:
        dict: Analysis of predictions
    """
    # Ensure the necessary columns exist
    required_columns = ['predicted_ca_status', 'prediction_method', 'attendance_percentage']
    if not all(col in result_data.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in result_data.columns]
        print(f"Warning: Missing columns in result_data: {missing_cols}")
        return {
            'total_students': 0,
            'total_ca_students': 0,
            'attendance_based_ca_count': 0,
            'model_ca_count': 0
        }

    # Total number of students
    total_students = len(result_data)

    # Total number of students predicted as CA
    total_ca_students = sum(1 for status in result_data['predicted_ca_status'] if status == 'CA')

    # Attendance-based calculation (students with attendance <= 90%)
    attendance_based_ca_count = sum(1 for status, method, att in zip(
        result_data['predicted_ca_status'],
        result_data['prediction_method'],
        result_data['attendance_percentage']
    ) if status == 'CA' and method == 'Attendance' and att <= 90.0)

    # Model-based calculation (excluding attendance-based determinations)
    model_ca_count = sum(1 for status, method, att in zip(
        result_data['predicted_ca_status'],
        result_data['prediction_method'],
        result_data['attendance_percentage']
    ) if status == 'CA' and method == 'ML Model' and att > 90.0)

    return {
        'total_students': total_students,
        'total_ca_students': total_ca_students,
        'attendance_based_ca_count': attendance_based_ca_count,
        'model_ca_count': model_ca_count
    }