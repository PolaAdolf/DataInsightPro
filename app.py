import streamlit as st
import pandas as pd
import os
import numpy as np
import time
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
    initial_sidebar_state="expanded",
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
    # Add custom CSS for responsive design
    st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    .main .block-container {
        max-width: 100%;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    h1, h2, h3 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .stButton button {
        width: 100%;
    }
    
    @media (max-width: 768px) {
        .row-widget.stRadio > div {
            flex-direction: column;
        }
        
        .row-widget.stRadio > div > label {
            margin-bottom: 0.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Application header with responsive design
    st.title("Comprehensive Data Analytics Platform")
    st.markdown("<p style='margin-bottom: 1rem;'>Upload your data and get insights across five analytics dimensions</p>", unsafe_allow_html=True)
    
    # Sidebar for navigation with improved styling
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
        
        # Create buttons with improved styling
        for tab_id, tab_name in tabs.items():
            # Highlight the current tab
            button_style = "primary" if st.session_state.current_tab == tab_id else "secondary"
            if st.button(tab_name, key=f"btn_{tab_id}", type=button_style):
                st.session_state.current_tab = tab_id
        
        st.divider()
        
        # Settings and controls
        with st.expander("Settings"):
            st.caption("Reset the application to upload new data")
            if st.button("🔄 Reset Application", type="secondary"):
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
    
    # Use columns for a more responsive layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
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
        
        # Add a container with custom styling for the uploader
        with st.container():
            st.markdown("""
            <style>
            .uploadedFile {
                width: 100% !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
            
            # Show supported file types
            st.caption("Supported formats: CSV (.csv) and Excel (.xlsx)")
    
    with col2:
        # Information card about the platform
        with st.container():
            st.markdown("""
            <div style="padding: 1rem; border-radius: 0.5rem; background-color: #f8f9fa; margin-bottom: 1rem;">
            <h3 style="margin-top: 0;">📊 Data Analytics Platform</h3>
            <p>This platform enables you to perform five types of analytics on your data:</p>
            <ul>
                <li><strong>Descriptive Analytics:</strong> What happened?</li>
                <li><strong>Diagnostic Analytics:</strong> Why did it happen?</li>
                <li><strong>Predictive Analytics:</strong> What might happen?</li>
                <li><strong>Prescriptive Analytics:</strong> What should be done?</li>
                <li><strong>Cognitive Analytics:</strong> AI-powered insights</li>
            </ul>
            <p>Plus a Work Plan Generator to implement findings.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Process the uploaded file
    if uploaded_file is not None:
        try:
            # Show a loading spinner while processing
            with st.spinner('Processing your data...'):
                # Load the data
                df = load_data(uploaded_file)
                
                # Store in session state
                st.session_state.data = df
                st.session_state.file_uploaded = True
                
                # Validate the data
                st.session_state.validation_results = validate_data(df)
            
            # Display responsive data preview
            st.header("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Display validation results
            st.header("Data Validation Results")
            
            if st.session_state.validation_results['is_valid']:
                st.success("✅ Your data is valid and ready for analysis!")
                
                # Data summary with responsive layout
                st.markdown("### Data Summary")
                
                # Use more columns for better mobile layout
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    st.metric("Rows", df.shape[0])
                    st.metric("Columns", df.shape[1])
                
                with col2:
                    st.metric("Numeric Features", len(st.session_state.validation_results['numeric_columns']))
                    st.metric("Categorical Features", len(st.session_state.validation_results['categorical_columns']))
                
                with col3:
                    st.metric("Date Features", len(st.session_state.validation_results['date_columns']))
                    missing_pct = df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
                    st.metric("Missing Values", f"{missing_pct:.2f}%")
                
                # Information about data types
                with st.expander("Data Types Information"):
                    st.write(f"**Data Types:** {', '.join(df.dtypes.astype(str).unique())}")
                    
                    if st.session_state.validation_results['numeric_columns']:
                        st.write("**Numeric Columns:**")
                        st.write(", ".join(st.session_state.validation_results['numeric_columns']))
                    
                    if st.session_state.validation_results['categorical_columns']:
                        st.write("**Categorical Columns:**")
                        st.write(", ".join(st.session_state.validation_results['categorical_columns']))
                    
                    if st.session_state.validation_results['date_columns']:
                        st.write("**Date Columns:**")
                        st.write(", ".join(st.session_state.validation_results['date_columns']))
                    
                    if st.session_state.validation_results['text_columns']:
                        st.write("**Text Columns:**")
                        st.write(", ".join(st.session_state.validation_results['text_columns']))
                
                # Proceed to analytics button - make it stand out
                st.markdown("<div style='padding: 1rem 0;'></div>", unsafe_allow_html=True)
                if st.button("✨ Proceed to Analytics", key="proceed_btn", type="primary"):
                    st.session_state.current_tab = "descriptive"
                    st.rerun()
            else:
                st.error("❌ There are issues with your data:")
                for issue in st.session_state.validation_results['issues']:
                    st.warning(issue)
                
                with st.container():
                    st.markdown("""
                    ### Recommendations:
                    - Check for missing values and fill them appropriately
                    - Ensure data types are consistent
                    - Remove duplicates if present
                    - Fix any other highlighted issues and upload again
                    """)
                    
                    # Add a retry button
                    st.button("📤 Upload Another File", key="retry_btn")
                
        except Exception as e:
            st.error(f"Error processing your file: {str(e)}")
            st.session_state.file_uploaded = False
            
            # Display troubleshooting information
            with st.expander("Troubleshooting Tips"):
                st.markdown("""
                - Make sure your file is correctly formatted
                - Check if the file is not corrupted
                - Ensure the file has headers in the first row
                - Try simplifying complex data before uploading
                """)
                
                st.button("Try Again", key="try_again_btn")

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
        # Enhanced warning with guidance
        st.warning("Please upload a file first.")
        
        # Add helpful card with instructions
        st.markdown("""
        <div style="padding: 1.5rem; border-radius: 0.5rem; background-color: #f8f9fa; margin: 1rem 0;">
            <h3 style="margin-top: 0;">How to Get Started</h3>
            <ol>
                <li>Go to the <b>Upload Data</b> tab in the sidebar</li>
                <li>Upload a CSV or Excel file with your data</li>
                <li>Wait for validation to complete</li>
                <li>Once validated, return to this tab</li>
            </ol>
            <p>Need sample data? Use any CSV or Excel file with numeric and categorical columns.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick navigation button
        if st.button("Go to Upload Tab", type="primary"):
            st.session_state.current_tab = "upload"
            st.rerun()

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
        # Enhanced warning with guidance
        st.warning("Please upload a file first.")
        
        # Add helpful card with instructions
        st.markdown("""
        <div style="padding: 1.5rem; border-radius: 0.5rem; background-color: #f8f9fa; margin: 1rem 0;">
            <h3 style="margin-top: 0;">How to Get Started</h3>
            <ol>
                <li>Go to the <b>Upload Data</b> tab in the sidebar</li>
                <li>Upload a CSV or Excel file with your data</li>
                <li>Wait for validation to complete</li>
                <li>Once validated, return to this tab</li>
            </ol>
            <p>Need sample data? Use any CSV or Excel file with numeric and categorical columns.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick navigation button
        if st.button("Go to Upload Tab", type="primary", key="goto_upload_diag"):
            st.session_state.current_tab = "upload"
            st.rerun()

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
        # Enhanced warning with guidance
        st.warning("Please upload a file first.")
        
        # Add helpful card with instructions
        st.markdown("""
        <div style="padding: 1.5rem; border-radius: 0.5rem; background-color: #f8f9fa; margin: 1rem 0;">
            <h3 style="margin-top: 0;">How to Get Started</h3>
            <ol>
                <li>Go to the <b>Upload Data</b> tab in the sidebar</li>
                <li>Upload a CSV or Excel file with your data</li>
                <li>Wait for validation to complete</li>
                <li>Once validated, return to this tab</li>
            </ol>
            <p>Need sample data? Use any CSV or Excel file with numeric and categorical columns.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick navigation button
        if st.button("Go to Upload Tab", type="primary", key="goto_upload_pred"):
            st.session_state.current_tab = "upload"
            st.rerun()

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
        # Enhanced warning with guidance
        st.warning("Please upload a file first.")
        
        # Add helpful card with instructions
        st.markdown("""
        <div style="padding: 1.5rem; border-radius: 0.5rem; background-color: #f8f9fa; margin: 1rem 0;">
            <h3 style="margin-top: 0;">How to Get Started</h3>
            <ol>
                <li>Go to the <b>Upload Data</b> tab in the sidebar</li>
                <li>Upload a CSV or Excel file with your data</li>
                <li>Wait for validation to complete</li>
                <li>Once validated, return to this tab</li>
            </ol>
            <p>Need sample data? Use any CSV or Excel file with numeric and categorical columns.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick navigation button
        if st.button("Go to Upload Tab", type="primary", key="goto_upload_pres"):
            st.session_state.current_tab = "upload"
            st.rerun()

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
        # Enhanced warning with guidance
        st.warning("Please upload a file first.")
        
        # Add helpful card with instructions
        st.markdown("""
        <div style="padding: 1.5rem; border-radius: 0.5rem; background-color: #f8f9fa; margin: 1rem 0;">
            <h3 style="margin-top: 0;">How to Get Started</h3>
            <ol>
                <li>Go to the <b>Upload Data</b> tab in the sidebar</li>
                <li>Upload a CSV or Excel file with your data</li>
                <li>Wait for validation to complete</li>
                <li>Once validated, return to this tab</li>
            </ol>
            <p>Need sample data? Use any CSV or Excel file with numeric and categorical columns.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick navigation button
        if st.button("Go to Upload Tab", type="primary", key="goto_upload_cog"):
            st.session_state.current_tab = "upload"
            st.rerun()

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
        # Enhanced warning with guidance
        st.warning("Please upload a file first.")
        
        # Add helpful card with instructions
        st.markdown("""
        <div style="padding: 1.5rem; border-radius: 0.5rem; background-color: #f8f9fa; margin: 1rem 0;">
            <h3 style="margin-top: 0;">How to Get Started</h3>
            <ol>
                <li>Go to the <b>Upload Data</b> tab in the sidebar</li>
                <li>Upload a CSV or Excel file with your data</li>
                <li>Wait for validation to complete</li>
                <li>Once validated, return to this tab</li>
            </ol>
            <p>Need sample data? Use any CSV or Excel file with numeric and categorical columns.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick navigation button
        if st.button("Go to Upload Tab", type="primary", key="goto_upload_work"):
            st.session_state.current_tab = "upload"
            st.rerun()

if __name__ == "__main__":
    main()
