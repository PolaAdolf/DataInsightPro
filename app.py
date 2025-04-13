import streamlit as st
import pandas as pd
import os
import numpy as np
from utils.data_loader import load_data
from utils.data_validator import validate_data
from analytics.descriptive import perform_descriptive_analytics
from analytics.diagnostic import perform_diagnostic_analytics
from analytics.predictive import perform_predictive_analytics
from analytics.prescriptive import perform_prescriptive_analytics
from analytics.cognitive import perform_cognitive_analytics
from utils.work_plan import generate_work_plan

st.set_page_config(
    page_title="Comprehensive Data Analytics Platform",
    page_icon="📊",
    layout="wide",
)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "upload"

def reset_app():
    """Reset the application state"""
    st.session_state.data = None
    st.session_state.file_uploaded = False
    st.session_state.validation_results = None
    st.session_state.current_tab = "upload"

def main():
    st.title("Comprehensive Data Analytics Platform")
    st.write("Upload your data and get insights across five analytics dimensions")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        
        # Only show analytics tabs if data is uploaded and validated
        if st.session_state.file_uploaded and st.session_state.validation_results['is_valid']:
            tabs = {
                "upload": "📂 Upload Data",
                "descriptive": "📊 Descriptive Analytics",
                "diagnostic": "🔍 Diagnostic Analytics",
                "predictive": "🔮 Predictive Analytics",
                "prescriptive": "🧠 Prescriptive Analytics",
                "cognitive": "💡 Cognitive Analytics",
                "work_plan": "📋 Work Plan"
            }
        else:
            tabs = {"upload": "📂 Upload Data"}
        
        for tab_id, tab_name in tabs.items():
            if st.button(tab_name, key=f"btn_{tab_id}"):
                st.session_state.current_tab = tab_id
        
        st.divider()
        if st.button("🔄 Reset Application"):
            reset_app()
            st.rerun()
    
    # Main content area
    if st.session_state.current_tab == "upload":
        render_upload_tab()
    elif st.session_state.current_tab == "descriptive":
        render_descriptive_tab()
    elif st.session_state.current_tab == "diagnostic":
        render_diagnostic_tab()
    elif st.session_state.current_tab == "predictive":
        render_predictive_tab()
    elif st.session_state.current_tab == "prescriptive":
        render_prescriptive_tab()
    elif st.session_state.current_tab == "cognitive":
        render_cognitive_tab()
    elif st.session_state.current_tab == "work_plan":
        render_work_plan_tab()

def render_upload_tab():
    st.header("Data Upload")
    
    with st.expander("Upload Instructions", expanded=not st.session_state.file_uploaded):
        st.markdown("""
        ### How to upload your data
        1. Prepare your data file in CSV or Excel (XLSX) format
        2. Click the 'Browse files' button below
        3. Select your file from your computer
        4. The system will automatically validate your data
        5. Once validated, you can explore different analytics views
        
        **Note**: Your data should have headers in the first row and be well-structured for best results
        """)
    
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # Load the data
            df = load_data(uploaded_file)
            
            # Store in session state
            st.session_state.data = df
            st.session_state.file_uploaded = True
            
            # Validate the data
            st.session_state.validation_results = validate_data(df)
            
            # Display data preview
            st.header("Data Preview")
            st.dataframe(df.head(10))
            
            # Display validation results
            st.header("Data Validation Results")
            
            if st.session_state.validation_results['is_valid']:
                st.success("✅ Your data is valid and ready for analysis!")
                
                st.markdown("### Data Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Rows:** {df.shape[0]}")
                    st.write(f"**Columns:** {df.shape[1]}")
                    st.write(f"**Numeric Features:** {len(st.session_state.validation_results['numeric_columns'])}")
                    st.write(f"**Categorical Features:** {len(st.session_state.validation_results['categorical_columns'])}")
                
                with col2:
                    st.write(f"**Date Features:** {len(st.session_state.validation_results['date_columns'])}")
                    st.write(f"**Missing Values:** {df.isna().sum().sum()} ({df.isna().sum().sum() / (df.shape[0] * df.shape[1]):.2%})")
                    st.write(f"**Data Types:** {', '.join(df.dtypes.astype(str).unique())}")
                
                # Proceed to analytics button
                if st.button("Proceed to Analytics", key="proceed_btn"):
                    st.session_state.current_tab = "descriptive"
                    st.rerun()
            else:
                st.error("❌ There are issues with your data:")
                for issue in st.session_state.validation_results['issues']:
                    st.warning(issue)
                
                st.markdown("""
                ### Recommendations:
                - Check for missing values and fill them appropriately
                - Ensure data types are consistent
                - Remove duplicates if present
                - Fix any other highlighted issues and upload again
                """)
                
        except Exception as e:
            st.error(f"Error processing your file: {str(e)}")
            st.session_state.file_uploaded = False

def render_descriptive_tab():
    st.header("📊 Descriptive Analytics")
    
    with st.expander("What is Descriptive Analytics?"):
        st.markdown("""
        **Descriptive Analytics** summarizes historical data to provide insights about what has happened. It includes:
        - Statistical summaries (mean, median, mode, etc.)
        - Distribution analysis
        - Pattern identification in historical data
        - Data visualization of past trends
        
        This type of analytics answers the question: **"What happened?"**
        """)
    
    if st.session_state.data is not None:
        perform_descriptive_analytics(st.session_state.data, st.session_state.validation_results)
    else:
        st.warning("Please upload a file first.")

def render_diagnostic_tab():
    st.header("🔍 Diagnostic Analytics")
    
    with st.expander("What is Diagnostic Analytics?"):
        st.markdown("""
        **Diagnostic Analytics** examines data to understand why something happened. It includes:
        - Correlation analysis
        - Anomaly detection
        - Root cause identification
        - Comparative analysis
        
        This type of analytics answers the question: **"Why did it happen?"**
        """)
    
    if st.session_state.data is not None:
        perform_diagnostic_analytics(st.session_state.data, st.session_state.validation_results)
    else:
        st.warning("Please upload a file first.")

def render_predictive_tab():
    st.header("🔮 Predictive Analytics")
    
    with st.expander("What is Predictive Analytics?"):
        st.markdown("""
        **Predictive Analytics** uses statistical algorithms and machine learning to predict future outcomes. It includes:
        - Forecasting future values
        - Trend projections
        - Risk assessment
        - Opportunity identification
        
        This type of analytics answers the question: **"What might happen in the future?"**
        """)
    
    if st.session_state.data is not None:
        perform_predictive_analytics(st.session_state.data, st.session_state.validation_results)
    else:
        st.warning("Please upload a file first.")

def render_prescriptive_tab():
    st.header("🧠 Prescriptive Analytics")
    
    with st.expander("What is Prescriptive Analytics?"):
        st.markdown("""
        **Prescriptive Analytics** suggests actions to take based on insights from the data. It includes:
        - Action recommendations
        - Decision support
        - Optimization strategies
        - Scenario analysis
        
        This type of analytics answers the question: **"What should be done?"**
        """)
    
    if st.session_state.data is not None:
        perform_prescriptive_analytics(st.session_state.data, st.session_state.validation_results)
    else:
        st.warning("Please upload a file first.")

def render_cognitive_tab():
    st.header("💡 Cognitive Analytics")
    
    with st.expander("What is Cognitive Analytics?"):
        st.markdown("""
        **Cognitive Analytics** leverages AI and natural language processing to generate human-like insights. It includes:
        - Natural language summaries
        - AI-generated insights
        - Pattern recognition
        - Context-aware analysis
        
        This type of analytics answers the question: **"What does this data tell us in human terms?"**
        """)
    
    if st.session_state.data is not None:
        perform_cognitive_analytics(st.session_state.data, st.session_state.validation_results)
    else:
        st.warning("Please upload a file first.")

def render_work_plan_tab():
    st.header("📋 Work Plan Generator")
    
    with st.expander("What is the Work Plan Generator?"):
        st.markdown("""
        The **Work Plan Generator** creates a time-based implementation roadmap based on the insights from all analytics. It includes:
        - Prioritized action items
        - Time estimates
        - Resource requirements
        - Implementation suggestions
        
        This helps you turn insights into actionable steps.
        """)
    
    if st.session_state.data is not None:
        generate_work_plan(st.session_state.data, st.session_state.validation_results)
    else:
        st.warning("Please upload a file first.")

if __name__ == "__main__":
    main()
