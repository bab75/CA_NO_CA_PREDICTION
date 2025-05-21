
"""
System configuration management for the CA Prediction System
"""
import json
import os
from pathlib import Path

DEFAULT_WEIGHTS = {
    "attendance": 0.35,
    "academic": 0.25,
    "behavioral": 0.20,
    "demographic": 0.10,
    "transportation": 0.10
}

DEFAULT_THRESHOLDS = {
    "ca_attendance": 90.0,
    "academic_risk": 70.0,
    "behavioral_risk": 3,
    "transportation_risk": 60
}

class SystemConfig:
    def __init__(self):
        self.config_file = Path("config/system_config.json")
        self.weights = DEFAULT_WEIGHTS.copy()
        self.thresholds = DEFAULT_THRESHOLDS.copy()
        self.load_config()

    def save_config(self):
        """Save current configuration to file"""
        self.config_file.parent.mkdir(exist_ok=True)
        config = {
            "weights": self.weights,
            "thresholds": self.thresholds
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)

    def load_config(self):
        """Load configuration from file"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.weights = config.get('weights', DEFAULT_WEIGHTS)
                self.thresholds = config.get('thresholds', DEFAULT_THRESHOLDS)

    def update_weights(self, new_weights):
        """Update pattern weights"""
        self.weights.update(new_weights)
        self.save_config()

    def update_thresholds(self, new_thresholds):
        """Update risk thresholds"""
        self.thresholds.update(new_thresholds)
        self.save_config()
