"""
Utility module for generating synthetic student data for the CA Prediction System
"""

import pandas as pd
import numpy as np
import random
import string
from datetime import datetime
import sys
import os

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG, DROPDOWN_OPTIONS, DEFAULT_VALUES

def generate_school_names(base_name, num_schools):
    """
    Generate a list of school names based on a pattern.
    
    Args:
        base_name (str): Base name pattern provided by user (e.g., "10U1", "A-SCHOOL")
        num_schools (int): Number of schools to generate
        
    Returns:
        list: List of generated school names
    """
    school_names = []
    
    # Check if the base name has a numeric suffix
    if base_name and base_name[-1].isdigit():
        prefix = base_name[:-1]
        start_num = int(base_name[-1])
        for i in range(start_num, start_num + num_schools):
            school_names.append(f"{prefix}{i}")
    
    # Check if the base name starts with a letter and has a hyphen
    elif base_name and "-" in base_name and base_name[0].isalpha():
        parts = base_name.split("-")
        # Fix: Check if parts[0] is a single character before using ord()
        if len(parts[0]) == 1:
            prefix_letter = ord(parts[0])
            suffix = parts[1] if len(parts) > 1 else "SCHOOL"
            
            for i in range(num_schools):
                school_names.append(f"{chr(prefix_letter + i)}-{suffix}")
        else:
            # Handle case when prefix is not a single character
            prefix = parts[0]
            suffix = parts[1] if len(parts) > 1 else "SCHOOL"
            
            for i in range(1, num_schools + 1):
                school_names.append(f"{prefix}-{suffix}{i}")
    
    # Default pattern generation
    else:
        for i in range(1, num_schools + 1):
            school_names.append(f"{base_name}{i}")
            
    return school_names

def generate_student_ids(num_students, prefix="H", year=None):
    """
    Generate unique student IDs with specified prefix
    
    Args:
        num_students (int): Number of student IDs to generate
        prefix (str): Prefix for student IDs (H for historical, C for current)
        year (int, optional): Year to include in the ID
        
    Returns:
        list: List of generated student IDs
    """
    student_ids = []
    year_suffix = f"{str(year)[-2:]}" if year else ""
    
    for i in range(1, num_students + 1):
        if year:
            student_ids.append(f"{prefix}{year_suffix}{i:04d}")
        else:
            student_ids.append(f"{prefix}{i:04d}")
            
    return student_ids

def calculate_attendance_metrics(present_days, absent_days):
    """
    Calculate attendance percentage and CA status
    
    Args:
        present_days (int): Number of days present
        absent_days (int): Number of days absent
        
    Returns:
        tuple: (total_days, attendance_percentage, ca_status)
    """
    total_days = present_days + absent_days
    attendance_percentage = (present_days / total_days) * 100 if total_days > 0 else 0
    ca_status = "CA" if attendance_percentage <= DATA_CONFIG["ca_threshold"] else "No-CA"
    
    return total_days, attendance_percentage, ca_status

def generate_student_data(
    num_students=DEFAULT_VALUES["num_students"],
    academic_years=DEFAULT_VALUES["academic_years"],
    schools=None,
    num_schools=DEFAULT_VALUES["num_schools"],
    school_base_name=DEFAULT_VALUES["school_prefix"],
    grades=None,
    gender_distribution=None,
    meal_code_distribution=None,
    academic_performance_range=(50, 100),
    present_days_range=(150, 180),
    absent_days_range=(0, 30),
    shelter_distribution=None,
    special_needs_distribution=None,
    bus_trip_distribution=None,
    transfer_distribution=None,
    suspended_distribution=None,
    dropout_distribution=None,
    custom_fields=None,
    include_student_id=True,
    selected_columns=None,
    is_historical=True
):
    """
    Generate synthetic student data with customizable attributes
    
    Args:
        num_students (int): Number of students to generate
        academic_years (list): List of academic years to include
        schools (list, optional): List of school names
        num_schools (int): Number of schools to generate if schools is None
        school_base_name (str): Base name for generating school names
        grades (list, optional): List of grades to include
        gender_distribution (dict, optional): Distribution for gender values
        meal_code_distribution (dict, optional): Distribution for meal code values
        academic_performance_range (tuple): Range for academic performance values
        present_days_range (tuple): Range for present days
        absent_days_range (tuple): Range for absent days
        shelter_distribution (dict, optional): Distribution for shelter values
        special_needs_distribution (dict, optional): Distribution for special needs values
        bus_trip_distribution (dict, optional): Distribution for bus trip values
        transfer_distribution (dict, optional): Distribution for transfer values
        suspended_distribution (dict, optional): Distribution for suspended values
        dropout_distribution (dict, optional): Distribution for dropout values
        custom_fields (dict, optional): Custom fields with their possible values
        include_student_id (bool): Whether to include student ID
        selected_columns (list, optional): Columns to include in the output
        is_historical (bool): Whether the data is historical or current
        
    Returns:
        pd.DataFrame: Generated student data
    """
    if not schools:
        schools = generate_school_names(school_base_name, num_schools)
    
    if not grades:
        grades = list(range(DATA_CONFIG["min_grade"], DATA_CONFIG["max_grade"] + 1))
    
    # Set default distributions if not provided
    if not gender_distribution:
        gender_distribution = {"M": 0.48, "F": 0.48, "O": 0.04}
    
    if not meal_code_distribution:
        meal_code_distribution = {"Free": 0.4, "Reduced": 0.2, "Paid": 0.4}
        
    if not shelter_distribution:
        shelter_distribution = {"NS": 0.8, "ST": 0.15, "S": 0.05}
        
    if not special_needs_distribution:
        special_needs_distribution = {"No": 0.85, "Yes": 0.15}
        
    if not bus_trip_distribution:
        bus_trip_distribution = {"No": 0.6, "Yes": 0.4}
        
    if not transfer_distribution:
        transfer_distribution = {"No": 0.9, "Yes": 0.1}
        
    if not suspended_distribution:
        suspended_distribution = {"No": 0.95, "Yes": 0.05}
        
    if not dropout_distribution:
        dropout_distribution = {"No": 0.98, "Yes": 0.02}
    
    all_data = []
    
    # Helper function for weighted random choice
    def weighted_choice(distribution):
        choices, weights = zip(*distribution.items())
        return np.random.choice(choices, p=weights)
    
    # Generate data for each academic year
    for year in academic_years:
        # For keeping track of student progression if multiple years
        student_grade_map = {}
        
        prefix = DATA_CONFIG["prefix_historical"] if is_historical else DATA_CONFIG["prefix_current"]
        student_ids = generate_student_ids(num_students, prefix, year)
        
        for i in range(num_students):
            student_data = {}
            
            # Student ID
            if include_student_id:
                student_data["student_id"] = student_ids[i]
            
            # Basic attributes
            student_data["academic_year"] = year
            student_data["school"] = np.random.choice(schools)
            
            # Check if student exists in previous years (for grade progression)
            student_id_base = student_ids[i][1:] if include_student_id else None
            if student_id_base in student_grade_map:
                # Progress the grade by 1 from previous year
                prev_grade = student_grade_map[student_id_base]
                if prev_grade < DATA_CONFIG["max_grade"]:
                    student_data["grade"] = prev_grade + 1
                else:
                    student_data["grade"] = prev_grade  # Stay in the same grade
            else:
                student_data["grade"] = np.random.choice(grades)
            
            # Update the grade map for this student
            if include_student_id:
                student_grade_map[student_id_base] = student_data["grade"]
            
            # Demographic attributes
            student_data["gender"] = weighted_choice(gender_distribution)
            student_data["meal_code"] = weighted_choice(meal_code_distribution)
            
            # Academic attributes
            student_data["academic_performance"] = np.random.uniform(*academic_performance_range)
            student_data["present_days"] = np.random.randint(*present_days_range)
            student_data["absent_days"] = np.random.randint(*absent_days_range)
            
            # Calculate attendance metrics
            total_days, attendance_percentage, ca_status = calculate_attendance_metrics(
                student_data["present_days"], student_data["absent_days"]
            )
            
            student_data["total_days"] = total_days
            student_data["attendance_percentage"] = attendance_percentage
            student_data["ca_status"] = ca_status
            
            # Additional attributes
            student_data["shelter"] = weighted_choice(shelter_distribution)
            student_data["special_need"] = weighted_choice(special_needs_distribution)
            student_data["bus_long_trip"] = weighted_choice(bus_trip_distribution)
            student_data["enrolled_transfer_schools"] = weighted_choice(transfer_distribution)
            student_data["suspended"] = weighted_choice(suspended_distribution)
            student_data["dropout_status"] = weighted_choice(dropout_distribution)
            
            # Add custom fields if provided
            if custom_fields:
                for field, values in custom_fields.items():
                    # Ensure the field name doesn't overwrite existing fields
                    if field not in student_data:
                        # If values is a dictionary, use it as a distribution
                        if isinstance(values, dict):
                            student_data[field] = weighted_choice(values)
                        # Otherwise treat values as a list of possible values
                        else:
                            student_data[field] = np.random.choice(values)
            
            all_data.append(student_data)
    
    # Create dataframe
    df = pd.DataFrame(all_data)
    
    # Select only the requested columns if specified
    if selected_columns:
        # Make sure all columns exist
        valid_columns = [col for col in selected_columns if col in df.columns]
        df = df[valid_columns]
    
    return df

def sample_data(num_rows=5):
    """
    Generate a small sample dataframe for UI display
    
    Args:
        num_rows (int): Number of rows to generate
        
    Returns:
        pd.DataFrame: Sample dataframe
    """
    return generate_student_data(
        num_students=num_rows,
        academic_years=[2023],
        num_schools=2,
        school_base_name="School-",
        grades=[9, 10, 11, 12],
        is_historical=True
    )
