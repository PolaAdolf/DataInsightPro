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
    page_title="Smart Data Explorer",
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
    st.title("📊 Smart Data Explorer")
    st.markdown("<p style='margin-bottom: 1rem;'>Upload your data and discover valuable insights with easy-to-understand analysis</p>", unsafe_allow_html=True)
    
    # Sidebar for navigation with improved styling
    with st.sidebar:
        st.header("Navigation")
        
        # Only show analytics tabs if data is uploaded and validated
        if st.session_state.file_uploaded and st.session_state.validation_results['is_valid']:
            tabs = {
                "upload": "📂 Upload Data",
                "descriptive": "📊 Data Summary",
                "diagnostic": "🔍 Find Patterns",
                "predictive": "🔮 Future Predictions",
                "prescriptive": "💡 Smart Recommendations",
                "cognitive": "🤖 AI Insights",
                "work_plan": "📋 Action Plan"
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
            <h3 style="margin-top: 0;">📊 Smart Data Explorer</h3>
            <p>Discover what your data is telling you with five easy analysis types:</p>
            <ul>
                <li><strong>📊 Data Summary:</strong> See what's in your data</li>
                <li><strong>🔍 Find Patterns:</strong> Understand connections in your data</li>
                <li><strong>🔮 Future Predictions:</strong> See what might happen next</li>
                <li><strong>💡 Smart Recommendations:</strong> Get suggestions on what to do</li>
                <li><strong>🤖 AI Insights:</strong> Let AI explain your data in simple terms</li>
            </ul>
            <p>Plus an Action Plan to help you implement what you discover.</p>
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
                    st.metric("Number Columns", len(st.session_state.validation_results['numeric_columns']))
                    st.metric("Category Columns", len(st.session_state.validation_results['categorical_columns']))
                
                with col3:
                    st.metric("Date Columns", len(st.session_state.validation_results['date_columns']))
                    missing_pct = df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
                    st.metric("Empty Cells", f"{missing_pct:.2f}%")
                
                # Information about data types
                with st.expander("Data Types Information"):
                    st.write(f"**Data Types:** {', '.join(df.dtypes.astype(str).unique())}")
                    
                    if st.session_state.validation_results['numeric_columns']:
                        st.write("**Number Columns (for calculations):**")
                        st.write(", ".join(st.session_state.validation_results['numeric_columns']))
                    
                    if st.session_state.validation_results['categorical_columns']:
                        st.write("**Category Columns (groups/labels):**")
                        st.write(", ".join(st.session_state.validation_results['categorical_columns']))
                    
                    if st.session_state.validation_results['date_columns']:
                        st.write("**Date Columns:**")
                        st.write(", ".join(st.session_state.validation_results['date_columns']))
                    
                    if st.session_state.validation_results['text_columns']:
                        st.write("**Text Columns:**")
                        st.write(", ".join(st.session_state.validation_results['text_columns']))
                
                # Proceed to analytics button - make it stand out
                st.markdown("<div style='padding: 1rem 0;'></div>", unsafe_allow_html=True)
                if st.button("✨ Start Exploring Your Data", key="proceed_btn", type="primary"):
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
    st.header("📊 Data Summary")
    
    with st.expander("What is Data Summary?"):
        st.markdown("""
        **Data Summary** shows you what's in your data in an easy-to-understand way. It includes:
        - Basic calculations (average, middle value, highest, lowest)
        - Charts showing how your data is spread out
        - Visual patterns in your information
        - Trends over time if you have dates
        
        This section answers: **"What does my data look like?"**
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
    st.header("🔍 Find Patterns")
    
    with st.expander("What is Find Patterns?"):
        st.markdown("""
        **Find Patterns** helps you understand connections and relationships in your data. It includes:
        - How different pieces of data relate to each other
        - Spotting unusual values or outliers
        - Finding what causes changes in your data
        - Comparing different groups or categories
        
        This section answers: **"How are things connected in my data?"**
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
    st.header("🔮 Future Predictions")
    
    with st.expander("What is Future Predictions?"):
        st.markdown("""
        **Future Predictions** uses smart computer methods to guess what might happen next. It includes:
        - Predicting future numbers based on past data
        - Showing where trends are heading
        - Identifying potential risks and opportunities
        - Estimating future outcomes
        
        This section answers: **"What might happen in the future based on my data?"**
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
    st.header("💡 Smart Recommendations")
    
    with st.expander("What are Smart Recommendations?"):
        st.markdown("""
        **Smart Recommendations** suggests specific actions you can take based on what your data shows. It includes:
        - Clear steps you can take to improve things
        - Help with making better decisions
        - Ways to get better results
        - Testing different scenarios
        
        This section answers: **"What should I do based on what my data shows?"**
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
    st.header("🤖 AI Insights")
    
    with st.expander("What are AI Insights?"):
        st.markdown("""
        **AI Insights** uses artificial intelligence to explain your data in plain English. It includes:
        - Summaries written like a human would explain them
        - Smart observations about your data
        - Automatic pattern spotting
        - Easy-to-understand explanations
        
        This section answers: **"What does my data mean in simple, everyday language?"**
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
    st.header("📋 Action Plan")
    
    with st.expander("What is the Action Plan?"):
        st.markdown("""
        The **Action Plan** creates a step-by-step plan based on everything you've discovered about your data. It includes:
        - List of things to do in order of importance
        - How long each step might take
        - What resources you'll need
        - Practical steps you can actually take
        
        This helps you turn your data discoveries into real actions.
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
