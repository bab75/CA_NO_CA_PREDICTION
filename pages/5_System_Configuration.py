
"""
System Configuration Page for the CA Prediction System
"""
import streamlit as st
from utils.system_setup import SystemConfig

def main():
    st.title("System Configuration")
    
    system_config = SystemConfig()
    
    st.header("Pattern Weights")
    st.info("Adjust the importance of different factors in risk calculation")
    
    # Weight adjustment
    weights = {}
    cols = st.columns(2)
    with cols[0]:
        weights['attendance'] = st.slider("Attendance Weight", 0.0, 1.0, 
                                        system_config.weights['attendance'])
        weights['academic'] = st.slider("Academic Weight", 0.0, 1.0, 
                                      system_config.weights['academic'])
        weights['behavioral'] = st.slider("Behavioral Weight", 0.0, 1.0, 
                                        system_config.weights['behavioral'])
    with cols[1]:
        weights['demographic'] = st.slider("Demographic Weight", 0.0, 1.0, 
                                         system_config.weights['demographic'])
        weights['transportation'] = st.slider("Transportation Weight", 0.0, 1.0, 
                                            system_config.weights['transportation'])

    # Thresholds configuration
    st.header("Risk Thresholds")
    st.info("Configure thresholds for risk categorization")
    
    thresholds = {}
    cols = st.columns(2)
    with cols[0]:
        thresholds['ca_attendance'] = st.number_input(
            "CA Attendance Threshold (%)", 
            50.0, 100.0, 
            system_config.thresholds['ca_attendance'])
        thresholds['academic_risk'] = st.number_input(
            "Academic Risk Threshold (%)", 
            0.0, 100.0, 
            system_config.thresholds['academic_risk'])
    with cols[1]:
        thresholds['behavioral_risk'] = st.number_input(
            "Behavioral Risk Threshold (incidents)", 
            0, 10, 
            system_config.thresholds['behavioral_risk'])
        thresholds['transportation_risk'] = st.number_input(
            "Transportation Risk Threshold (minutes)", 
            0, 120, 
            system_config.thresholds['transportation_risk'])

    # Save changes
    if st.button("Save Configuration"):
        system_config.update_weights(weights)
        system_config.update_thresholds(thresholds)
        st.success("Configuration saved successfully!")

    # Documentation
    with st.expander("Configuration Guide"):
        st.markdown("""
        ### Pattern Weights
        Adjust these weights to control how different factors influence the overall risk assessment:
        - **Attendance Weight**: Impact of attendance history
        - **Academic Weight**: Impact of academic performance
        - **Behavioral Weight**: Impact of behavioral incidents
        - **Demographic Weight**: Impact of demographic factors
        - **Transportation Weight**: Impact of transportation factors

        ### Risk Thresholds
        Configure thresholds that determine risk levels:
        - **CA Attendance**: % below which a student is considered chronically absent
        - **Academic Risk**: Grade % below which academic risk is flagged
        - **Behavioral Risk**: Number of incidents that indicates behavioral risk
        - **Transportation Risk**: Travel time that indicates transportation risk
        """)

if __name__ == "__main__":
    main()
