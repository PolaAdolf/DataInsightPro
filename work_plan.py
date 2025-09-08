import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from utils.gemini import generate_work_plan as generate_gemini_work_plan, get_gemini_api_key
import json
import re
from datetime import datetime, timedelta

# Gemini API key function is now imported from utils.gemini

def generate_work_plan(df, validation_results):
    """
    Generate a work plan based on the data analysis
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset used for analysis
    validation_results : dict
        Results from data validation including column classifications
    """
    st.header("Implementation Work Plan")
    
    with st.expander("About the Work Plan Generator", expanded=True):
        st.markdown("""
        The **Work Plan Generator** creates a time-based implementation roadmap based on insights
        from all analytics performed on your data. It includes:
        
        - Prioritized action items derived from data analysis
        - Time estimates for implementation phases
        - Resource requirements and considerations
        - Success metrics and KPIs to track progress
        
        This tool helps you transform insights into structured, actionable steps for business improvement.
        """)
    
    # Create summary description of the dataset for AI
    data_summary = create_data_summary(df, validation_results)
    
    # Let user select work plan focus
    plan_focus = st.selectbox(
        "Select work plan focus:",
        [
            "Comprehensive Improvement Plan",
            "Data Quality Enhancement",
            "Business Process Optimization",
            "Performance Improvement",
            "Customer Experience Enhancement",
            "Risk Mitigation Strategy"
        ],
        key="plan_focus"
    )
    
    # Let user select timeline
    timeline = st.selectbox(
        "Select implementation timeline:",
        ["Short-term (1-3 months)", "Medium-term (3-6 months)", "Long-term (6-12 months)"],
        key="timeline"
    )
    
    # Let user specify additional context
    context = st.text_area(
        "Additional context or specific goals (optional):",
        height=100,
        key="context",
        placeholder="E.g., focus on cost reduction, team size constraints, budget limitations, specific business goals..."
    )
    
    # Generate the work plan
    if st.button("Generate Work Plan", key="gen_plan_btn"):
        # Verify API key
        api_key = get_gemini_api_key()
        if not api_key:
            st.error("Gemini API key not found. Please add it to your environment variables.")
            return
        
        with st.spinner("Generating comprehensive work plan..."):
            try:
                # Generate work plan with AI
                work_plan = generate_gemini_work_plan(data_summary, plan_focus, timeline, context)
                
                if work_plan:
                    display_work_plan(work_plan, timeline)
                else:
                    st.error("Failed to generate work plan. Please try again.")
            except Exception as e:
                st.error(f"Error generating work plan: {str(e)}")

def create_data_summary(df, validation_results):
    """
    Create a summary of the dataset for work plan generation
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to summarize
    validation_results : dict
        Results from data validation
        
    Returns:
    --------
    str : Summary of the dataset
    """
    # Basic dataset information
    summary = f"Dataset with {len(df)} rows and {len(df.columns)} columns.\n\n"
    
    # Column types
    summary += f"Column types: {len(validation_results['numeric_columns'])} numeric, "
    summary += f"{len(validation_results['categorical_columns'])} categorical, "
    summary += f"{len(validation_results['date_columns'])} date/time, "
    summary += f"{len(validation_results['text_columns'])} text.\n\n"
    
    # Data quality summary
    missing_data = df.isna().sum().sum()
    total_cells = df.size
    missing_percentage = (missing_data / total_cells) * 100
    
    summary += f"Data quality: {missing_percentage:.2f}% missing values"
    
    if 'issues' in validation_results and validation_results['issues']:
        summary += ", identified issues: " + ", ".join(validation_results['issues'])
    
    summary += ".\n\n"
    
    # Key numeric metrics if available
    if validation_results['numeric_columns']:
        summary += "Key metrics summary:\n"
        for col in validation_results['numeric_columns'][:3]:  # Limit to first 3 for brevity
            try:
                summary += f"- {col}: mean={df[col].mean():.2f}, median={df[col].median():.2f}, "
                summary += f"min={df[col].min():.2f}, max={df[col].max():.2f}\n"
            except:
                pass
    
    # Key categorical distributions if available
    if validation_results['categorical_columns']:
        summary += "\nKey categorical distributions:\n"
        for col in validation_results['categorical_columns'][:2]:  # Limit to first 2 for brevity
            try:
                top_val = df[col].value_counts().index[0]
                top_pct = df[col].value_counts(normalize=True).iloc[0] * 100
                summary += f"- {col}: most common value is '{top_val}' ({top_pct:.1f}%)\n"
            except:
                pass
    
    # Date range if available
    if validation_results['date_columns']:
        summary += "\nDate range:\n"
        for col in validation_results['date_columns'][:1]:  # Just use the first date column
            try:
                date_col = pd.to_datetime(df[col], errors='coerce')
                min_date = date_col.min()
                max_date = date_col.max()
                summary += f"- {col}: from {min_date} to {max_date}\n"
            except:
                pass
    
    return summary

# generate_work_plan function is now imported from utils.gemini as generate_gemini_work_plan
# Function is now using Gemini via imported generate_gemini_work_plan

def display_work_plan(work_plan, timeline):
    """
    Display the generated work plan in a structured format
    
    Parameters:
    -----------
    work_plan : dict
        The generated work plan in JSON format
    timeline : str
        The selected timeline
    """
    # Display executive summary
    st.subheader("Executive Summary")
    st.write(work_plan["executive_summary"])
    
    # Display implementation timeline
    st.subheader("Implementation Timeline")
    
    # Create a Gantt chart for phases
    phase_df = []
    for phase in work_plan["phases"]:
        phase_df.append({
            "Task": phase["name"],
            "Start": phase["start_month"],
            "End": phase["end_month"] + 0.99,  # Add 0.99 to make the bar extend to the end of the month
            "Type": "Phase"
        })
    
    # Add tasks to the Gantt chart
    task_counter = 0
    for phase_idx, phase in enumerate(work_plan["phases"]):
        phase_start = phase["start_month"]
        
        for task_idx, task in enumerate(phase["tasks"]):
            # Extract duration in weeks
            duration_text = task["duration"]
            weeks_match = re.search(r'(\d+\.?\d*)\s*weeks?', duration_text)
            days_match = re.search(r'(\d+\.?\d*)\s*days?', duration_text)
            
            if weeks_match:
                duration_months = float(weeks_match.group(1)) / 4  # Approximate weeks to months
            elif days_match:
                duration_months = float(days_match.group(1)) / 30  # Approximate days to months
            else:
                duration_months = 0.5  # Default to 2 weeks if parsing fails
            
            # Calculate task start based on dependencies
            task_start = phase_start
            
            # If not the first task and has dependencies, shift it a bit
            if task_idx > 0 and task["dependencies"] != "None" and task["dependencies"]:
                task_start += 0.2 * task_idx
            
            # Ensure task stays within phase
            task_end = min(task_start + duration_months, phase["end_month"])
            
            # Priority color
            priority_color = {
                "High": "red",
                "Medium": "orange",
                "Low": "blue"
            }.get(task["priority"], "gray")
            
            phase_df.append({
                "Task": f"  {task['name']}",  # Indent task names
                "Start": task_start,
                "End": task_end,
                "Type": "Task",
                "Priority": task["priority"],
                "Color": priority_color
            })
            
            task_counter += 1
    
    # Convert to DataFrame
    timeline_df = pd.DataFrame(phase_df)
    
    # Create the Gantt chart
    if not timeline_df.empty:
        fig = px.timeline(
            timeline_df, 
            x_start="Start", 
            x_end="End", 
            y="Task",
            color="Type",
            title="Implementation Timeline (Months)",
            color_discrete_map={"Phase": "rgb(102, 102, 255)", "Task": "rgb(60, 179, 113)"}
        )
        
        # Customize hover information
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>Start: Month %{x:.1f}<br>End: Month %{x_end:.1f}"
        )
        
        # Adjust layout
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="",
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=[f"Month {i}" for i in range(1, 13)]
            )
        )
        
        # Show the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Provide a legend for priority colors
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("🔴 High Priority")
        with col2:
            st.markdown("🟠 Medium Priority")
        with col3:
            st.markdown("🔵 Low Priority")
    
    # Display phases and tasks
    st.subheader("Detailed Implementation Plan")
    
    for phase in work_plan["phases"]:
        with st.expander(f"{phase['name']} (Months {phase['start_month']}-{phase['end_month']})", expanded=True):
            st.write(phase["description"])
            
            # Display tasks in a table
            task_df = []
            for task in phase["tasks"]:
                task_df.append({
                    "Task": task["name"],
                    "Description": task["description"],
                    "Duration": task["duration"],
                    "Resources": task["resources"],
                    "Dependencies": task["dependencies"],
                    "Priority": task["priority"]
                })
            
            if task_df:
                st.dataframe(pd.DataFrame(task_df), use_container_width=True)
            else:
                st.info("No tasks defined for this phase.")
    
    # Display KPIs
    st.subheader("Key Performance Indicators (KPIs)")
    kpi_df = []
    for kpi in work_plan["kpis"]:
        kpi_df.append({
            "KPI": kpi["name"],
            "Description": kpi["description"],
            "Target": kpi["target"]
        })
    
    if kpi_df:
        st.dataframe(pd.DataFrame(kpi_df), use_container_width=True)
    
    # Display risks in a table with color coding
    st.subheader("Risk Assessment")
    risk_df = []
    for risk in work_plan["risks"]:
        risk_df.append({
            "Risk": risk["risk"],
            "Impact": risk["impact"],
            "Mitigation Strategy": risk["mitigation"]
        })
    
    if risk_df:
        risk_df = pd.DataFrame(risk_df)
        
        # Display risks with conditional formatting
        def highlight_impact(val):
            color_map = {
                "High": "background-color: rgba(255, 0, 0, 0.2)",
                "Medium": "background-color: rgba(255, 165, 0, 0.2)",
                "Low": "background-color: rgba(0, 128, 0, 0.2)"
            }
            return color_map.get(val, "")
        
        # Apply styling
        styled_risk_df = risk_df.style.applymap(highlight_impact, subset=["Impact"])
        st.dataframe(styled_risk_df, use_container_width=True)
    
    # Display resource requirements
    st.subheader("Resource Requirements")
    for idx, resource in enumerate(work_plan["resources"]):
        with st.expander(f"{resource['role']}", expanded=idx == 0):
            st.write(f"**Responsibilities:** {resource['responsibilities']}")
            st.write(f"**Required Skills:** {resource['required_skills']}")
    
    # Download button for the work plan as JSON
    st.download_button(
        label="Download Work Plan (JSON)",
        data=json.dumps(work_plan, indent=2),
        file_name="work_plan.json",
        mime="application/json",
        key="download_workplan"
    )
