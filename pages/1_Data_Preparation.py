import streamlit as st

# Set page config must be the first streamlit command
st.set_page_config(
    page_title="Data Preparation - CA Prediction System",
    page_icon="📊",
    layout="wide"
)


#The code has been modified to make the CA status optional and handle cases #where it's not present in the uploaded data.


#Data Preparation Page for the CA Prediction System


import pandas as pd
import numpy as np
import os
import sys
import json
import base64
from datetime import datetime
from io import BytesIO

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_generator import (
    generate_student_data, 
    generate_school_names
)
from config import DATA_CONFIG, DROPDOWN_OPTIONS, DEFAULT_VALUES

# Add CSS for styling
st.markdown("""
<style>
    .distribution-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .distribution-item {
        flex: 1;
        min-width: 200px;
    }
    .preview-container {
        margin-top: 20px;
    }
    .custom-field {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

def display_data_tab(data_type="historical"):
    """
    Display the data generation tab UI

    Args:
        data_type (str): Type of data to generate (historical or current)
    """

    column_prefix = data_type
    is_historical = (data_type == "historical")

    # Basic settings
    st.subheader(f"{data_type.title()} Data Generation")

    col1, col2, col3 = st.columns(3)

    with col1:
        num_students = st.number_input(
            "Number of Students",
            min_value=DATA_CONFIG["min_students"],
            max_value=DATA_CONFIG["max_students"],
            value=DEFAULT_VALUES["num_students"],
            step=100,
            key=f"{column_prefix}_num_students"
        )

    with col2:
        # Use the current year as default for current data, min_year for historical
        default_year = DATA_CONFIG["max_year"] if not is_historical else DATA_CONFIG["min_year"]

        # Ensure default_year is within valid range
        if default_year < DATA_CONFIG["min_year"]:
            default_year = DATA_CONFIG["min_year"]
        if default_year > DATA_CONFIG["max_year"]:
            default_year = DATA_CONFIG["max_year"]

        year_from = st.number_input(
            "Academic Year (From)",
            min_value=DATA_CONFIG["min_year"],
            max_value=DATA_CONFIG["max_year"],
            value=default_year,
            step=1,
            key=f"{column_prefix}_year_from"
        )

    with col3:
        # For current data, keep the same year by default. For historical data, use a range.
        default_to_year = default_year if not is_historical else min(default_year + 1, DATA_CONFIG["max_year"])

        # Ensure default_to_year is not less than year_from
        if default_to_year < year_from:
            default_to_year = year_from

        year_to = st.number_input(
            "Academic Year (To)",
            min_value=year_from,
            max_value=DATA_CONFIG["max_year"],
            value=default_to_year,
            step=1,
            key=f"{column_prefix}_year_to"
        )

    # Academic years range
    academic_years = list(range(year_from, year_to + 1))

    # School settings
    st.subheader("School Settings")

    col1, col2 = st.columns(2)

    with col1:
        school_base_name = st.text_input(
            "School Name Pattern",
            value="School-",
            help="The base name for schools (e.g., '10U', 'A-SCHOOL')",
            key=f"{column_prefix}_school_base_name"
        )

    with col2:
        num_schools = st.number_input(
            "Number of Schools",
            min_value=1,
            max_value=50,
            value=DEFAULT_VALUES["num_schools"],
            step=1,
            key=f"{column_prefix}_num_schools"
        )

    # School names
    school_names = generate_school_names(school_base_name, num_schools)
    st.write("Generated School Names:", ", ".join(school_names))

    # Student attributes
    st.subheader("Student Attributes")

    # Use tabs for better organization
    tabs = st.tabs([
        "Basic Attributes", 
        "Attendance", 
        "Additional Attributes", 
        "Column Selection", 
        "Custom Fields"
    ])

    # Basic attributes tab
    with tabs[0]:
        col1, col2 = st.columns(2)

        with col1:
            # Grade selection
            grade_selection = st.multiselect(
                "Grades",
                options=list(range(DATA_CONFIG["min_grade"], DATA_CONFIG["max_grade"] + 1)),
                default=list(range(DATA_CONFIG["min_grade"], DATA_CONFIG["max_grade"] + 1)),
                key=f"{column_prefix}_grades"
            )

            # Gender distribution
            st.write("Gender Distribution")
            gender_male = st.slider(
                "Male (%)",
                min_value=0,
                max_value=100,
                value=48,
                step=1,
                key=f"{column_prefix}_gender_male"
            )

            gender_female = st.slider(
                "Female (%)",
                min_value=0,
                max_value=100,
                value=48,
                step=1,
                key=f"{column_prefix}_gender_female"
            )

            # Adjust the other gender percentage to make the total 100%
            gender_other = 100 - gender_male - gender_female
            gender_other = max(0, gender_other)
            st.write(f"Other (%): {gender_other}")

            # Create gender distribution dictionary
            gender_distribution = {
                "M": gender_male / 100,
                "F": gender_female / 100,
                "O": gender_other / 100
            }

        with col2:
            # Meal code distribution
            st.write("Meal Code Distribution")
            meal_free = st.slider(
                "Free (%)",
                min_value=0,
                max_value=100,
                value=40,
                step=1,
                key=f"{column_prefix}_meal_free"
            )

            meal_reduced = st.slider(
                "Reduced (%)",
                min_value=0,
                max_value=100,
                value=20,
                step=1,
                key=f"{column_prefix}_meal_reduced"
            )

            # Adjust the paid percentage to make the total 100%
            meal_paid = 100 - meal_free - meal_reduced
            meal_paid = max(0, meal_paid)
            st.write(f"Paid (%): {meal_paid}")

            # Create meal code distribution dictionary
            meal_code_distribution = {
                "Free": meal_free / 100,
                "Reduced": meal_reduced / 100,
                "Paid": meal_paid / 100
            }

            # Academic performance range
            st.write("Academic Performance Range")
            academic_min, academic_max = st.slider(
                "Range (%)",
                min_value=1,
                max_value=100,
                value=(50, 100),
                step=1,
                key=f"{column_prefix}_academic_range"
            )

    # Attendance tab
    with tabs[1]:
        col1, col2 = st.columns(2)

        with col1:
            # Present days range
            st.write("Present Days Range")
            present_min, present_max = st.slider(
                "Range (days)",
                min_value=DATA_CONFIG["min_present_days"],
                max_value=DATA_CONFIG["max_present_days"],
                value=(150, 180),
                step=1,
                key=f"{column_prefix}_present_range"
            )

        with col2:
            # Absent days range
            st.write("Absent Days Range")
            absent_min, absent_max = st.slider(
                "Range (days)",
                min_value=DATA_CONFIG["min_absent_days"],
                max_value=DATA_CONFIG["max_absent_days"],
                value=(0, 30),
                step=1,
                key=f"{column_prefix}_absent_range"
            )

        # Display attendance metrics
        st.write("""
        Attendance metrics will be calculated automatically:
        - Total Days = Present Days + Absent Days
        - Attendance % = (Present Days / Total Days) * 100
        - CA Status = "CA" if Attendance % ≤ 90%, else "No-CA"
        """)

        # CA threshold information
        st.info(f"Current CA threshold is set to {DATA_CONFIG['ca_threshold']}%. Students with attendance below this threshold will be marked as Chronically Absent (CA).")

    # Additional attributes tab
    with tabs[2]:
        col1, col2 = st.columns(2)

        with col1:
            # Shelter distribution
            st.write("Shelter Distribution")
            shelter_ns = st.slider(
                "NS - No Shelter (%)",
                min_value=0,
                max_value=100,
                value=80,
                step=1,
                key=f"{column_prefix}_shelter_ns"
            )

            shelter_st = st.slider(
                "ST - Short Term (%)",
                min_value=0,
                max_value=100,
                value=15,
                step=1,
                key=f"{column_prefix}_shelter_st"
            )

            # Adjust the S percentage to make the total 100%
            shelter_s = 100 - shelter_ns - shelter_st
            shelter_s = max(0, shelter_s)
            st.write(f"S - Shelter (%): {shelter_s}")

            # Create shelter distribution dictionary
            shelter_distribution = {
                "NS": shelter_ns / 100,
                "ST": shelter_st / 100,
                "S": shelter_s / 100
            }

            # Special needs
            st.write("Special Needs")
            special_needs_options = DROPDOWN_OPTIONS["special_need_options"]
            special_needs_yes = st.slider(
                "Students with Special Needs (%)",
                min_value=0,
                max_value=100,
                value=15,
                step=1,
                key=f"{column_prefix}_special_needs_yes"
            )

            special_needs_no = 100 - special_needs_yes

            # Create special needs distribution dictionary
            special_needs_distribution = {
                "Yes": special_needs_yes / 100,
                "No": special_needs_no / 100
            }

        with col2:
            # Bus long trip
            st.write("Bus Long Trip")
            bus_trip_yes = st.slider(
                "Students with Long Bus Trips (%)",
                min_value=0,
                max_value=100,
                value=40,
                step=1,
                key=f"{column_prefix}_bus_trip_yes"
            )

            bus_trip_no = 100 - bus_trip_yes

            # Create bus trip distribution dictionary
            bus_trip_distribution = {
                "Yes": bus_trip_yes / 100,
                "No": bus_trip_no / 100
            }

            # Transfer schools
            st.write("Transfer Schools")
            transfer_yes = st.slider(
                "Students with Transfers (%)",
                min_value=0,
                max_value=100,
                value=10,
                step=1,
                key=f"{column_prefix}_transfer_yes"
            )

            transfer_no = 100 - transfer_yes

            # Create transfer distribution dictionary
            transfer_distribution = {
                "Yes": transfer_yes / 100,
                "No": transfer_no / 100
            }

            # Suspended
            st.write("Suspended")
            suspended_yes = st.slider(
                "Suspended Students (%)",
                min_value=0,
                max_value=100,
                value=5,
                step=1,
                key=f"{column_prefix}_suspended_yes"
            )

            suspended_no = 100 - suspended_yes

            # Create suspended distribution dictionary
            suspended_distribution = {
                "Yes": suspended_yes / 100,
                "No": suspended_no / 100
            }

            # Dropout status
            st.write("Dropout Status")
            dropout_yes = st.slider(
                "Dropout Students (%)",
                min_value=0,
                max_value=100,
                value=2,
                step=1,
                key=f"{column_prefix}_dropout_yes"
            )

            dropout_no = 100 - dropout_yes

            # Create dropout distribution dictionary
            dropout_distribution = {
                "Yes": dropout_yes / 100,
                "No": dropout_no / 100
            }

    # Column selection tab
    with tabs[3]:
        st.write("Select columns to include in the generated dataset:")

        # Default columns
        default_columns = [
            "student_id", 
            "academic_year", 
            "school", 
            "grade", 
            "gender", 
            "meal_code", 
            "academic_performance", 
            "present_days", 
            "absent_days", 
            "total_days", 
            "attendance_percentage", 
            "ca_status", 
            "shelter", 
            "special_need", 
            "bus_long_trip", 
            "enrolled_transfer_schools", 
            "suspended", 
            "dropout_status"
        ]

        # Add custom fields if any
        if f"{column_prefix}_custom_fields" in st.session_state and st.session_state[f"{column_prefix}_custom_fields"]:
            custom_field_names = [field["name"] for field in st.session_state[f"{column_prefix}_custom_fields"]]
            default_columns.extend(custom_field_names)

        # Column selection
        selected_columns = st.multiselect(
            "Columns",
            options=default_columns,
            default=default_columns,
            key=f"{column_prefix}_selected_columns"
        )

        # Student ID options
        include_student_id = "student_id" in selected_columns

        if include_student_id:
            st.write(f"Student IDs will be prefixed with '{DATA_CONFIG['prefix_historical'] if is_historical else DATA_CONFIG['prefix_current']}'")

    # Custom fields tab
    with tabs[4]:
        st.write("Add custom fields to the dataset:")

        # Initialize custom fields if not already in session state
        if f"{column_prefix}_custom_fields" not in st.session_state:
            st.session_state[f"{column_prefix}_custom_fields"] = []

        # Form for adding new custom fields
        with st.form(key=f"{column_prefix}_add_custom_field_form"):
            st.subheader("Add New Custom Field")

            # Use different key naming to avoid session state conflicts
            field_name = st.text_input("Field Name", key=f"input_{column_prefix}_field_name")
            field_values = st.text_input(
                "Possible Values (comma-separated)",
                help="Enter comma-separated values, e.g., 'Bus, Walk, Bike'",
                key=f"input_{column_prefix}_field_values"
            )

            # Submit button for the form
            submit_button = st.form_submit_button("Add Field")

            if submit_button:
                if field_name and field_values:
                    # Parse values
                    values_list = [value.strip() for value in field_values.split(",") if value.strip()]

                    if values_list:
                        # Add the new custom field
                        st.session_state[f"{column_prefix}_custom_fields"].append({
                            "name": field_name,
                            "values": values_list
                        })

                        # No need to clear the form since we're using different keys

                        st.success(f"Added custom field: {field_name}")
                    else:
                        st.error("Please enter at least one value for the field")
                else:
                    st.error("Please enter both field name and values")

        # Display existing custom fields
        if st.session_state[f"{column_prefix}_custom_fields"]:
            st.subheader("Custom Fields")

            for i, field in enumerate(st.session_state[f"{column_prefix}_custom_fields"]):
                with st.container():
                    st.markdown(f"""
                    <div class="custom-field">
                        <strong>{field['name']}</strong>: {', '.join(field['values'])}
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button(f"Remove", key=f"{column_prefix}_remove_field_{i}"):
                        st.session_state[f"{column_prefix}_custom_fields"].pop(i)
                        st.rerun()
        else:
            st.info("No custom fields added yet")

    # Generate data button
    if st.button(f"Generate {data_type.title()} Data", key=f"generate_{column_prefix}_data"):
        with st.spinner(f"Generating {data_type.title()} Data..."):
            try:
                # Prepare custom fields
                custom_fields = {}
                if f"{column_prefix}_custom_fields" in st.session_state and st.session_state[f"{column_prefix}_custom_fields"]:
                    for field in st.session_state[f"{column_prefix}_custom_fields"]:
                        custom_fields[field["name"]] = field["values"]

                # Generate the dataset
                df = generate_student_data(
                    num_students=num_students,
                    academic_years=academic_years,
                    schools=school_names,
                    grades=grade_selection,
                    gender_distribution=gender_distribution,
                    meal_code_distribution=meal_code_distribution,
                    academic_performance_range=(academic_min, academic_max),
                    present_days_range=(present_min, present_max),
                    absent_days_range=(absent_min, absent_max),
                    shelter_distribution=shelter_distribution,
                    special_needs_distribution=special_needs_distribution,
                    bus_trip_distribution=bus_trip_distribution,
                    transfer_distribution=transfer_distribution,
                    suspended_distribution=suspended_distribution,
                    dropout_distribution=dropout_distribution,
                    custom_fields=custom_fields,
                    include_student_id=include_student_id,
                    selected_columns=selected_columns,
                    is_historical=is_historical
                )

                # Store the generated data in session state
                st.session_state[f"{column_prefix}_data"] = df

                # Success message
                st.success(f"Generated {len(df)} records of {data_type} data!")

                # Preview the data
                st.subheader("Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

                # Download button
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{data_type}_student_data.csv">Download {data_type.title()} Data as CSV</a>'
                st.markdown(href, unsafe_allow_html=True)

                # Stats
                st.subheader("Data Statistics")
                if "ca_status" in df.columns:
                    ca_count = df[df["ca_status"] == "CA"].shape[0]
                    ca_percentage = (ca_count / len(df)) * 100
                else:
                    ca_count = 0
                    ca_percentage = 0

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Records", len(df))
                col2.metric("CA Students", ca_count)
                col3.metric("CA Percentage", f"{ca_percentage:.2f}%")

            except Exception as e:
                st.error(f"Error generating data: {str(e)}")

    # Display stored data if available
    elif f"{column_prefix}_data" in st.session_state:
        df = st.session_state[f"{column_prefix}_data"]

        # Preview the data
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # Download button
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{data_type}_student_data.csv">Download {data_type.title()} Data as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

        # Stats
        if f"{column_prefix}_data" in st.session_state:
            df = st.session_state[f"{column_prefix}_data"]
            if not df.empty and "ca_status" in df.columns:
                st.subheader("Data Statistics")
                if "ca_status" in df.columns:
                    ca_count = df[df["ca_status"] == "CA"].shape[0]
                    ca_percentage = (ca_count / len(df)) * 100
                else:
                    ca_count = 0
                    ca_percentage = 0

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Records", len(df))
                col2.metric("CA Students", ca_count)
                col3.metric("CA Percentage", f"{ca_percentage:.2f}%")

def main():
    # Custom CSS for animations and styling
    st.markdown("""
    <style>
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translate3d(0, 40px, 0);
            }
            to {
                opacity: 1;
                transform: translate3d(0, 0, 0);
            }
        }

        .data-prep-header {
            background: linear-gradient(120deg, #4F46E5, #7C3AED);
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            animation: fadeInUp 0.8s ease-out;
        }

        .data-prep-header h1 {
            font-size: 2.5rem;
            margin: 0;
            padding: 0;
        }

        .data-prep-header p {
            font-size: 1.1rem;
            margin-top: 10px;
            opacity: 0.9;
        }

        .data-icon {
            display: inline-block;
            margin-right: 15px;
            vertical-align: middle;
        }

        /* Tab panel animations */
        .stTabs [data-baseweb="tab-panel"] {
            animation: fadeIn 0.5s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Button styling */
        .stButton > button {
            background: linear-gradient(90deg, #4F46E5, #7C3AED);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        .stButton > button:hover {
            background: linear-gradient(90deg, #7C3AED, #6D28D9);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }
    </style>
    """, unsafe_allow_html=True)

    # Custom header with animation
    st.markdown("""
    <div class="data-prep-header">
        <div class="data-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="white">
                <path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/>
            </svg>
        </div>
        <h1>Data Preparation</h1>
        <p>Generate or upload data for training prediction models and analyzing student attendance patterns</p>
    </div>
    """, unsafe_allow_html=True)

    # Hide the original icon (we're using our custom header)
    st.image("assets/absenteeism_icon.svg", width=0)

    # Tabs for historical and current data
    data_tabs = st.tabs(["Historical Data Generation", "Current Year Data Generation", "Upload Data"])

    # Historical data tab
    with data_tabs[0]:
        display_data_tab("historical")

    # Current year data tab
    with data_tabs[1]:
        display_data_tab("current")

    # Upload data tab
    with data_tabs[2]:
        st.subheader("Upload Your Own Data")

        # File uploader
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx"],
            help="Upload your own dataset for analysis",
            key="upload_data"
        )

        if uploaded_file is not None:
            try:
                # Determine file type
                file_extension = uploaded_file.name.split(".")[-1].lower()

                # Read the file
                if file_extension == "csv":
                    df = pd.read_csv(uploaded_file)
                elif file_extension == "xlsx":
                    df = pd.read_excel(uploaded_file)

                # Store the data in session state
                st.session_state["uploaded_data"] = df

                # Preview the data
                st.subheader("Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

                # Stats
                st.subheader("Data Statistics")
                col1, col2 = st.columns(2)
                col1.metric("Total Records", len(df))
                col2.metric("Number of Columns", len(df.columns))

                # Check if CA status column exists
                if "ca_status" in df.columns:
                    ca_count = df[df["ca_status"] == "CA"].shape[0]
                    ca_percentage = (ca_count / len(df)) * 100
                else:
                    ca_count = 0
                    ca_percentage = 0

                col1, col2 = st.columns(2)
                col1.metric("CA Students", ca_count)
                col2.metric("CA Percentage", f"{ca_percentage:.2f}%")

                # Option to save as historical or current data
                st.subheader("Save Data As")
                col1, col2, col3 = st.columns([1, 1, 2])

                with col1:
                    if st.button("Save as Historical Data"):
                        st.session_state["historical_data"] = df
                        st.success("Data saved as historical data!")

                with col2:
                    if st.button("Save as Current Year Data"):
                        st.session_state["current_data"] = df
                        st.success("Data saved as current year data!")

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    # Documentation
    with st.expander("Documentation: How to Prepare Data"):
        st.markdown("""
        ## Data Preparation Module Documentation

        This module allows you to generate synthetic student data for training and testing machine learning models for Chronic Absenteeism (CA) prediction.

        ### Historical Data Generation
        Use this tab to generate historical student data. This data will be used for training your machine learning models.

        ### Current Year Data Generation
        Use this tab to generate current year student data. This data will be used for making predictions.

        ### Upload Data
        If you have your own dataset, you can upload it here and use it instead of generating synthetic data.

        ### Key Features
        1. **Flexible Data Generation**: Control the number of students, academic years, and schools.
        2. **Customizable Attributes**: Set distributions for various student attributes like gender, meal codes, and special needs.
        3. **Automatic Calculations**: Total days, attendance percentage, and CA status are calculated automatically.
        4. **Custom Fields**: Add your own custom fields with specified values.
        5. **Column Selection**: Choose which columns to include in the generated dataset.
        6. **Student ID Prefixing**: Historical data uses "H" prefix, current data uses "C" prefix.

        ### Data Generation Process
        1. Configure the parameters for data generation using the provided controls.
        2. Click the "Generate Data" button to create the dataset.
        3. Preview the generated data and download it as a CSV file if needed.
        4. The data will also be stored in the app's session state for use in other modules.

        ### Tips
        - Make sure the historical and current data have compatible structures for the ML models.
        - Include enough CA and non-CA examples in your training data.
        - Use realistic parameter values for better model performance.
        - Add custom fields to test additional factors that might influence absenteeism.
        """)

if __name__ == "__main__":
    main()