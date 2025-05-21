import streamlit as st
import os
import sys
import pandas as pd

# Add the current directory to path so we can import our utility modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_generator import sample_data

# Load SVG icons for better UI
def load_svg(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        # Fallback to an empty SVG if file not found
        return f'<svg width="24" height="24" viewBox="0 0 24 24"></svg>'

# Load SVG icons from files
DATA_PREP_ICON = load_svg('assets/data_prep_icon.svg')
MODEL_TRAIN_ICON = load_svg('assets/model_icon.svg')
PREDICTION_ICON = load_svg('assets/absenteeism_icon.svg')
ANALYSIS_ICON = load_svg('assets/analytics_icon.svg')
STUDENT_ICON = load_svg('assets/student_icon.svg')
GAUGE_ICON = load_svg('assets/gauge_icon.svg')
INTERVENTION_ICON = load_svg('assets/intervention_icon.svg')

# Add CSS for styling 
st.set_page_config(
    page_title="Chronic Absenteeism Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #1E3A8A;
    }
    .subtitle {
        font-size: 1.5rem;
        font-weight: 400;
        margin-bottom: 2rem;
        color: #4B5563;
    }
    .feature-card {
        background-color: #F9FAFB;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #E5E7EB;
        transition: all 0.3s;
    }
    .feature-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        transform: translateY(-5px);
    }
    .feature-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
    }
    .feature-icon {
        margin-right: 0.5rem;
    }
    .feature-description {
        color: #4B5563;
    }
    .data-preview {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #F3F4F6;
    }
    .stExpander {
        border: 1px solid #E5E7EB;
        border-radius: 0.5rem;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
        text-align: center;
        font-size: 0.875rem;
        color: #6B7280;
    }
    /* Fix for SVG rendering in Streamlit */
    svg {
        display: inline-block;
        vertical-align: middle;
    }
    
    /* Mini cards */
    .mini-card {
        padding: 1rem;
        min-height: 150px;
    }
    
    /* Feature links */
    .feature-actions {
        margin-top: 1rem;
        text-align: right;
    }
    
    .feature-link {
        color: #1E3A8A;
        text-decoration: none;
        font-weight: 500;
        border-bottom: 1px solid transparent;
        transition: border-color 0.3s;
        padding-bottom: 2px;
    }
    
    .feature-link:hover {
        border-bottom-color: #1E3A8A;
    }
    
    /* Gauge styles */
    .gauge-container {
        padding: 1rem;
        background-color: white;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Risk categories */
    .risk-high {
        color: #DC2626;
        font-weight: 600;
    }
    
    .risk-medium {
        color: #F59E0B;
        font-weight: 600;
    }
    
    .risk-low {
        color: #10B981;
        font-weight: 600;
    }
    
    /* Intervention cards */
    .intervention-suggestion {
        background-color: #F0F9FF;
        border-left: 4px solid #0EA5E9;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.25rem;
    }
    
    /* Prediction card */
    .prediction-card {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)

# Main page content
def reset_session_state():
    """
    Reset all session state variables to clear the application state
    """
    # Get all keys in session state
    keys_to_remove = [key for key in st.session_state.keys()]
    
    # Remove each key
    for key in keys_to_remove:
        # Skip certain keys that are used for UI state if needed
        if key == "_pages":  # Don't remove Streamlit's internal page tracking
            continue
        del st.session_state[key]
    
    # Show success message
    st.success("Application state has been reset successfully!")
    
def main():
    # Add sidebar with user guide and reset button
    with st.sidebar:
        st.title("Application Controls")
        
        # Comprehensive User Guide
        st.markdown("### 📚 User Guide")
        
        with st.expander("Step 1: Data Preparation", expanded=False):
            st.markdown("""
            #### Data Preparation Guide
            
            1. **Generate Historical Data**:
               - Choose the number of students (100-5000)
               - Select academic years (2021-2025)
               - Configure distributions for gender, meal codes, etc.
               - Click "Generate Data" to create training dataset
               
            2. **Generate Current Year Data**:
               - Similar settings, but for predicting current students
               - Student IDs with "C" prefix for current year data
               
            3. **Upload Custom Data**:
               - Alternative to generation: upload CSV files
               - Must include required columns (explained in UI)
               
            4. **Data Preview**:
               - Review generated data in tables
               - Check column distributions
            """)
        
        with st.expander("Step 2: Model Training", expanded=False):
            st.markdown("""
            #### Model Training Guide
            
            1. **Data Configuration**:
               - Select data source for training
               - **Target Category**: Choose what to predict (e.g., Chronic Absenteeism)
               - **Select Specific Target**: Choose specific column (e.g., "ca_status" or "attendance_percentage")
               - Select features to include in model
               
            2. **Model Selection**:
               - Choose one or more ML models to train
               - Adjust hyperparameters if needed
               - Configure test size and cross-validation
               
            3. **Training Results**:
               - Compare model performance metrics
               - Check confusion matrix and feature importance
               - Set best model for predictions
               
            > **Important**: For predicting chronic absenteeism, choose "ca_status" as target (categorical) rather than "attendance_percentage" (numerical) for better results.
            """)
        
        with st.expander("Step 3: Predictions", expanded=False):
            st.markdown("""
            #### Predictions Guide
            
            1. **Prepare Current Data**:
               - Load current year student data
               - Review data patterns and distributions
               
            2. **Run Predictions**:
               - Use your trained model to predict chronic absenteeism
               - System applies selected model to current data
               - Snow animation shows when prediction completes!
               
            3. **View Results**:
               - Check prediction summary (CA count and percentage)
               - Explore results by school, grade, risk levels
               - Review individual student dashboards
               - Get intervention recommendations
            """)
        
        with st.expander("Step 4: Advanced Analysis", expanded=False):
            st.markdown("""
            #### Advanced Analysis Guide
            
            1. **Feature Importance**:
               - Understand which factors most influence CA
               - See feature weights for selected model
               
            2. **Risk Distribution**:
               - View risk levels across different dimensions
               - Compare schools, grades, demographic groups
               
            3. **Correlation Analysis**:
               - Explore relationships between attendance and other factors
               - Find patterns in academic performance vs. attendance
               
            4. **Temporal Trends**:
               - Track patterns across academic years
               - Identify seasonal attendance fluctuations
            """)
        
        with st.expander("Troubleshooting", expanded=False):
            st.markdown("""
            #### Common Issues & Solutions
            
            1. **Error: "Target column has only one unique value"**
               - Choose a different target column with 2+ values
               - Generate more diverse data
            
            2. **Warning: "Classes with few samples"**
               - Generate more balanced data
               - Use different evaluation metrics
            
            3. **Prediction shows 0 CA students**
               - Verify your model was trained correctly
               - Check target selection is appropriate
            
            4. **Model performance is poor**
               - Include more relevant features
               - Try different models
               - Generate larger training dataset
            """)
        
        # Add divider
        st.markdown("---")
        
        # Add reset button
        if st.button("🔄 Clear & Reset All", help="Reset all data and start fresh"):
            reset_session_state()
            st.rerun()
    
    # Add more dynamic CSS animations
    st.markdown("""
    <style>
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .logo-container {
            animation: float 6s ease-in-out infinite;
        }
        
        .main-header {
            background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
            border-radius: 12px;
            padding: 25px;
            color: white;
            margin-bottom: 30px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            position: relative;
            overflow: hidden;
        }
        
        .main-header::before {
            content: "";
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 60%);
            animation: pulse 10s infinite;
        }
        
        .main-header-title {
            font-size: 2.75rem;
            font-weight: 700;
            margin: 0;
            position: relative;
            z-index: 1;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
        }
        
        .main-header-subtitle {
            font-size: 1.4rem;
            font-weight: 400;
            opacity: 0.9;
            margin-top: 10px;
            position: relative;
            z-index: 1;
        }
        
        /* Button animations */
        .nav-buttons {
            margin-top: 15px;
        }
        
        .nav-button {
            display: inline-block;
            background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            padding: 8px 16px;
            margin-right: 10px;
            border-radius: 4px;
            text-decoration: none;
            transition: all 0.3s;
            font-weight: 500;
        }
        
        .nav-button:hover {
            background: rgba(255,255,255,0.3);
            transform: translateY(-2px);
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }
        
        /* Updated feature card hover effects */
        .feature-card {
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        
        .feature-card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # New animated header with logo
    st.markdown("""
    <div class="main-header">
        <div class="row">
            <div class="column" style="width: 20%; float: left;">
                <div class="logo-container">
                    <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8Y2lyY2xlIGN4PSIxMDAiIGN5PSIxMDAiIHI9IjkwIiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjgiIC8+CiAgPHBhdGggZD0iTTQ1LDEzMCBRMTAwLDUwIDE1NSwxMzAiIGZpbGw9Im5vbmUiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iOCIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiAvPgogIDxjaXJjbGUgY3g9IjcwIiBjeT0iODAiIHI9IjE1IiBmaWxsPSJ3aGl0ZSIgLz4KICA8Y2lyY2xlIGN4PSIxMzAiIGN5PSI4MCIgcj0iMTUiIGZpbGw9IndoaXRlIiAvPgo8L3N2Zz4=" alt="Logo" width="150" />
                </div>
            </div>
            <div class="column" style="width: 80%; float: left;">
                <h1 class="main-header-title">Chronic Absenteeism Prediction</h1>
                <p class="main-header-subtitle">Track, Predict, and Intervene - A Complete Solution for Educational Success</p>
                <div class="nav-buttons">
                    <a href="/1_Data_Preparation" class="nav-button">Data Preparation</a>
                    <a href="/2_Model_Training" class="nav-button">Train Models</a>
                    <a href="/3_Predictions" class="nav-button">Predictions</a>
                    <a href="/4_Advanced_Analysis" class="nav-button">Advanced Analysis</a>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hidden original logo (for compatibility)
    st.image("assets/logo.svg", width=0, output_format="SVG")
    
    # Create a dashboard introduction with animation
    st.markdown("""
    <div style="background: linear-gradient(135deg, #E0F2FE 0%, #EFF6FF 100%); 
    padding: 20px; border-radius: 10px; margin: 20px 0; border: 1px solid #BFDBFE;
    animation: fadeIn 1s ease-in-out;">
        <h3 style="color: #1E40AF; margin-bottom: 10px;">Welcome to the Future of Attendance Management</h3>
        <p style="color: #1E3A8A; line-height: 1.6;">
            This comprehensive system helps education professionals predict and analyze chronic absenteeism 
            patterns, track student progression across years, and develop targeted intervention strategies.
            Built with advanced analytics and machine learning to provide actionable insights.
        </p>
    </div>
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Feature cards with SVG icons
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-title">
                <span class="feature-icon">{DATA_PREP_ICON}</span>
                Data Preparation
            </div>
            <div class="feature-description">
                Generate synthetic student data with customizable attributes or upload your existing datasets. 
                Configure detailed student profiles including attendance patterns, demographic information, and academic performance.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-title">
                <span class="feature-icon">{PREDICTION_ICON}</span>
                Predictions & Intervention
            </div>
            <div class="feature-description">
                Apply trained models to current student data to identify at-risk students and 
                develop targeted intervention strategies. View predictions by school, grade, and other dimensions.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-title">
                <span class="feature-icon">{MODEL_TRAIN_ICON}</span>
                Model Training
            </div>
            <div class="feature-description">
                Train and optimize multiple machine learning models to accurately predict chronic absenteeism risks. 
                Compare model performance and select the most effective predictors for your student population.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-title">
                <span class="feature-icon">{ANALYSIS_ICON}</span>
                Advanced Analysis
            </div>
            <div class="feature-description">
                Explore detailed visualizations and analytics to understand attendance patterns, risk factors, 
                and correlations. Identify trends across schools, grades, and demographic groups.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # Additional feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="feature-card mini-card">
            <div class="feature-title">
                <span class="feature-icon">{STUDENT_ICON}</span>
                Student Tracking
            </div>
            <div class="feature-description">
                Monitor student progression across academic years with advanced grade tracking and trend analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="feature-card mini-card">
            <div class="feature-title">
                <span class="feature-icon">{GAUGE_ICON}</span>
                Risk Assessment
            </div>
            <div class="feature-description">
                Visual risk meters and gauges help quickly identify at-risk students requiring intervention.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="feature-card mini-card">
            <div class="feature-title">
                <span class="feature-icon">{INTERVENTION_ICON}</span>
                Intervention Planning
            </div>
            <div class="feature-description">
                Generate tailored intervention strategies based on student-specific risk factors and patterns.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display sample data preview
    st.markdown('<div class="data-preview">', unsafe_allow_html=True)
    st.subheader("Sample Student Data Preview")
    
    try:
        sample_df = sample_data(10)
        st.dataframe(sample_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating sample data: {str(e)}")
        # Fallback to an empty dataframe with the expected columns
        cols = ["student_id", "academic_year", "school", "grade", "gender", "meal_code", 
                "academic_performance", "attendance_percentage", "ca_status"]
        sample_df = pd.DataFrame(columns=cols)
        st.info("Sample data could not be generated. Please navigate to the Data Preparation page to generate data.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Instructions for use
    with st.expander("📋 How to use this application"):
        st.markdown("""
        ### Getting Started Guide
        
        Follow these steps to make the most of the Chronic Absenteeism Prediction System:
        
        1. **Data Preparation** ➡️ Begin by generating historical student data or uploading your existing dataset
           - Configure student attributes including attendance patterns and demographic details
           - Generate data for multiple academic years to enable longitudinal analysis
        
        2. **Model Training** ➡️ Train machine learning models to predict chronic absenteeism
           - Select relevant features for prediction
           - Compare different models to find the most accurate predictor
           - Tune hyperparameters to optimize model performance
        
        3. **Predictions** ➡️ Apply your trained models to current year data
           - Identify students at risk of chronic absenteeism
           - View prediction results across different dimensions (school, grade, etc.)
           - Develop targeted intervention strategies based on identified patterns
        
        4. **Advanced Analysis** ➡️ Explore detailed visualizations and analytics
           - Analyze correlations between attendance and other factors
           - Identify attendance patterns across different student populations
           - Track student progression across academic years
        
        All data generated or uploaded is processed locally and is not stored permanently unless you 
        explicitly save it. Models can be trained and saved for future predictions.
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>© 2025 Chronic Absenteeism Prediction & Student Progression System | Last Updated: May 4, 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
