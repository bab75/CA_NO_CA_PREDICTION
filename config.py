"""
Configuration settings for the Chronic Absenteeism Prediction System
"""

# Data generation configuration
DATA_CONFIG = {
    "min_students": 100,
    "max_students": 5000,
    "min_year": 2021,
    "max_year": 2025,
    "prefix_historical": "H",
    "prefix_current": "C",
    "min_grade": 1,
    "max_grade": 12,
    "gender_options": ["M", "F", "O"],
    "meal_code_options": ["Free", "Reduced", "Paid", "1", "2", "3", "4", "5"],
    "shelter_options": ["NS", "ST", "S"],
    "min_present_days": 1,
    "max_present_days": 200,
    "min_absent_days": 1,
    "max_absent_days": 200,
    "ca_threshold": 90  # Percentage, below which student is marked as chronically absent
}

# Model configuration
MODEL_CONFIG = {
    "models": {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "decision_tree": "Decision Tree",
        "svm": "Support Vector Machine",
        "gradient_boosting": "Gradient Boosting",
        "neural_network": "Neural Network"
    },
    "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
    "default_test_size": 0.2,
    "default_random_state": 42
}

# Visualization configuration
VIZ_CONFIG = {
    "color_palette": {
        "primary": "#4e89ae",
        "secondary": "#43658b",
        "ca_color": "#ff6b6b",
        "no_ca_color": "#51cf66",
        "highlight": "#ffd43b",
        "text": "#262730"
    },
    "bubble_plot": {
        "min_size": 5,
        "max_size": 30,
        "alpha": 0.7
    }
}

# Path configuration
PATH_CONFIG = {
    "models_dir": "models",
    "data_dir": "data",
    "assets_dir": "assets"
}

# Default values for generation
DEFAULT_VALUES = {
    "num_students": 500,
    "num_schools": 5,
    "school_prefix": "School",
    "academic_years": [2021, 2022]
}

# Dropdown options
DROPDOWN_OPTIONS = {
    "special_need_options": ["Yes", "No", "Learning Disability", "Physical Disability", "ADHD", "Autism"],
    "bus_trip_options": ["Yes", "No", "Long Distance", "Short Distance"],
    "transfer_options": ["Yes", "No", "Within District", "Outside District"],
    "suspended_options": ["Yes", "No", "Multiple Times", "Once"],
    "dropout_options": ["Yes", "No", "At Risk"]
}
