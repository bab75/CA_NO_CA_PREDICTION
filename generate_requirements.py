"""
Script to generate requirements.txt for GitHub and Streamlit Cloud deployment
"""
import os

# List of required packages with versions
requirements = [
    "joblib>=1.5.0",
    "matplotlib>=3.10.1",
    "numpy>=2.2.5",
    "pandas>=2.2.3",
    "plotly>=6.0.1",
    "scikit-learn>=1.6.1",
    "seaborn>=0.13.2",
    "streamlit>=1.45.0",
]

# Write requirements to a file in a different name to avoid restrictions
with open("requirements_for_github.txt", "w") as f:
    for req in requirements:
        f.write(f"{req}\n")

print("Generated 'requirements_for_github.txt' file successfully!")
print("Please rename it to 'requirements.txt' after downloading.")