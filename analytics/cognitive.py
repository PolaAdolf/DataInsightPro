import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import os
from openai import OpenAI
import json
import time

def get_openai_api_key():
    """Get OpenAI API key from environment variable"""
    return os.getenv("OPENAI_API_KEY")

def perform_cognitive_analytics(df, validation_results):
    """
    Perform cognitive analytics on the dataset using AI and NLP
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    validation_results : dict
        Results from data validation including column classifications
    """
    st.subheader("AI-Powered Insights")
    
    # Check if API key exists
    api_key = get_openai_api_key()
    if not api_key:
        st.error("OpenAI API key not found. Please add it to your environment variables.")
        
        with st.expander("How to add OpenAI API key"):
            st.markdown("""
            To use cognitive analytics features, you need an OpenAI API key:
            
            1. Sign up for an account at [OpenAI](https://platform.openai.com)
            2. Create an API key in your account dashboard
            3. Add the API key to your environment variables as OPENAI_API_KEY
            
            This will enable AI-generated insights and analysis.
            """)
        return
    
    # Create dataset summary for the AI
    dataset_description = generate_dataset_description(df, validation_results)
    
    # Interactive analysis options
    analysis_type = st.radio(
        "Select analysis type:",
        ["Auto-generated Insights", "Custom Question Answering", "Natural Language Summary", "Executive Summary"],
        key="cognitive_type"
    )
    
    # Auto-generated Insights
    if analysis_type == "Auto-generated Insights":
        with st.expander("About Auto-generated Insights", expanded=True):
            st.markdown("""
            **Auto-generated Insights** uses artificial intelligence to automatically identify patterns,
            trends, anomalies, and recommendations based on your data. The AI analyzes:
            
            - Key statistical patterns and relationships
            - Potential business implications
            - Actionable insights and next steps
            - Opportunities for further investigation
            
            The system uses a combination of statistical analysis and natural language
            processing to translate complex data patterns into human-readable insights.
            """)
        
        if st.button("Generate Insights", key="gen_insights_btn"):
            with st.spinner("Analyzing data and generating insights..."):
                insights = generate_ai_insights(dataset_description, "general")
                
                if insights:
                    st.success("Analysis complete!")
                    st.markdown(insights)
                else:
                    st.error("Failed to generate insights. Please try again.")
    
    # Custom Question Answering
    elif analysis_type == "Custom Question Answering":
        with st.expander("About Custom Question Answering", expanded=True):
            st.markdown("""
            **Custom Question Answering** allows you to ask specific questions about your data
            and receive AI-generated answers. You can ask about:
            
            - Trends and patterns in the data
            - Relationships between variables
            - Potential causes of observed phenomena
            - Business implications of the data
            - Recommendations for action
            
            Ask your question in natural language, and the AI will analyze the data to provide
            a detailed answer.
            """)
        
        question = st.text_area(
            "What would you like to know about your data?",
            height=100,
            key="custom_question",
            placeholder="E.g., What factors most strongly influence sales? What patterns can be seen in customer behavior? What recommendations would you make based on this data?"
        )
        
        if st.button("Ask Question", key="ask_btn") and question:
            with st.spinner("Analyzing your question..."):
                answer = generate_ai_insights(dataset_description, question)
                
                if answer:
                    st.success("Analysis complete!")
                    st.markdown(answer)
                else:
                    st.error("Failed to generate answer. Please try again.")
    
    # Natural Language Summary
    elif analysis_type == "Natural Language Summary":
        with st.expander("About Natural Language Summary", expanded=True):
            st.markdown("""
            **Natural Language Summary** provides a comprehensive overview of your data in plain language.
            The AI generates a narrative that describes:
            
            - Key characteristics of the dataset
            - Main distributions and central tendencies
            - Notable patterns and relationships
            - Potential areas of interest
            
            This feature translates complex data into a readable narrative that anyone can understand,
            bridging the gap between data analysis and business interpretation.
            """)
            
        if st.button("Generate Summary", key="summary_btn"):
            with st.spinner("Generating natural language summary..."):
                summary = generate_natural_language_summary(df, validation_results, dataset_description)
                
                if summary:
                    st.success("Summary generated!")
                    st.markdown(summary)
                else:
                    st.error("Failed to generate summary. Please try again.")
    
    # Executive Summary
    elif analysis_type == "Executive Summary":
        with st.expander("About Executive Summary", expanded=True):
            st.markdown("""
            **Executive Summary** creates a concise, business-focused overview designed for decision-makers.
            The summary includes:
            
            - Key findings and their business implications
            - Critical metrics and their performance
            - Strategic recommendations
            - Suggested next steps
            
            Executive summaries are designed to be brief yet comprehensive, focusing on
            actionable insights rather than technical details.
            """)
            
        if st.button("Generate Executive Summary", key="exec_summary_btn"):
            with st.spinner("Generating executive summary..."):
                exec_summary = generate_executive_summary(df, validation_results, dataset_description)
                
                if exec_summary:
                    st.success("Executive summary generated!")
                    st.markdown(exec_summary)
                else:
                    st.error("Failed to generate executive summary. Please try again.")
    
    # Additional cognitive analytics for numeric datasets only
    if len(validation_results['numeric_columns']) >= 2:
        st.subheader("Key Driver Analysis")
        
        with st.expander("About Key Driver Analysis", expanded=True):
            st.markdown("""
            **Key Driver Analysis** identifies the most important factors that influence a target variable.
            The AI analyzes relationships between variables to determine:
            
            - Which factors have the strongest impact on your target
            - The nature and direction of these relationships
            - Potential causal connections
            - Actionable recommendations based on key drivers
            
            This analysis helps you focus on the variables that matter most for your business outcomes.
            """)
        
        # Let user select target variable
        target_variable = st.selectbox(
            "Select target variable to analyze key drivers:",
            validation_results['numeric_columns'],
            key="target_for_drivers"
        )
        
        if st.button("Analyze Key Drivers", key="drivers_btn"):
            with st.spinner("Analyzing key drivers..."):
                # Get potential driver variables (excluding the target)
                driver_vars = [col for col in validation_results['numeric_columns'] + validation_results['categorical_columns'] 
                              if col != target_variable]
                
                if len(driver_vars) < 1:
                    st.warning("Not enough potential driver variables for analysis.")
                else:
                    # Add target information to the dataset description
                    target_desc = f"\nTarget variable for key driver analysis: {target_variable}"
                    full_desc = dataset_description + target_desc
                    
                    # Generate key driver analysis
                    driver_analysis = generate_key_driver_analysis(df, target_variable, driver_vars, full_desc)
                    
                    if driver_analysis:
                        st.success("Key driver analysis complete!")
                        st.markdown(driver_analysis)
                    else:
                        st.error("Failed to generate key driver analysis. Please try again.")
                        
    # Sentiment and text analysis section (if text columns exist)
    if validation_results['text_columns']:
        st.subheader("Text & Sentiment Analysis")
        
        with st.expander("About Text & Sentiment Analysis", expanded=True):
            st.markdown("""
            **Text & Sentiment Analysis** examines textual data to extract themes, sentiments, and insights.
            The analysis includes:
            
            - Overall sentiment assessment (positive, negative, neutral)
            - Key topics and themes in the text
            - Frequent terms and their contexts
            - Sentiment trends and patterns
            
            This feature helps you understand the qualitative aspects of your data that might be
            missed in purely numerical analysis.
            """)
        
        # Let user select text column to analyze
        text_column = st.selectbox(
            "Select text column to analyze:",
            validation_results['text_columns'],
            key="text_column"
        )
        
        if st.button("Analyze Text", key="text_analysis_btn"):
            with st.spinner("Analyzing text data..."):
                # Check if there's enough text data
                text_data = df[text_column].dropna().astype(str)
                
                if len(text_data) < 3:
                    st.warning("Not enough text data for analysis. Need at least 3 text entries.")
                else:
                    # Sample text data (limit to 100 records for API efficiency)
                    if len(text_data) > 100:
                        sampled_text = text_data.sample(100, random_state=42)
                        st.info(f"Analyzing a random sample of 100 records out of {len(text_data)} total.")
                    else:
                        sampled_text = text_data
                    
                    # Join text data for analysis
                    combined_text = "\n\n---\n\n".join(sampled_text.values)
                    
                    # Create a text analysis description
                    text_desc = f"""
                    Text Analysis Request:
                    
                    Column being analyzed: {text_column}
                    Number of text entries: {len(sampled_text)}
                    
                    Text sample (combined entries):
                    {combined_text[:5000]}... [truncated if longer]
                    
                    Please provide a comprehensive text analysis including:
                    1. Overall sentiment analysis (positive, negative, neutral)
                    2. Key themes and topics identified
                    3. Notable patterns or trends in the text
                    4. Recommendations based on the text analysis
                    """
                    
                    # Generate text analysis
                    text_analysis = generate_text_analysis(text_desc)
                    
                    if text_analysis:
                        st.success("Text analysis complete!")
                        st.markdown(text_analysis)
                    else:
                        st.error("Failed to generate text analysis. Please try again.")

def generate_dataset_description(df, validation_results):
    """
    Generate a detailed description of the dataset for AI analysis
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to describe
    validation_results : dict
        Results from data validation including column classifications
    
    Returns:
    --------
    str : A detailed description of the dataset
    """
    # Basic dataset information
    row_count = len(df)
    col_count = len(df.columns)
    
    description = f"Dataset Information:\n"
    description += f"- {row_count} rows and {col_count} columns\n"
    description += f"- {len(validation_results['numeric_columns'])} numeric columns\n"
    description += f"- {len(validation_results['categorical_columns'])} categorical columns\n"
    description += f"- {len(validation_results['date_columns'])} date columns\n"
    description += f"- {len(validation_results['text_columns'])} text columns\n\n"
    
    # Missing data information
    missing_data = df.isna().sum().sum()
    missing_percentage = (missing_data / (row_count * col_count)) * 100
    description += f"Missing data: {missing_percentage:.2f}% of all values\n\n"
    
    # Column details
    description += "Column details:\n"
    
    # Numeric columns
    if validation_results['numeric_columns']:
        description += "\nNumeric columns:\n"
        for col in validation_results['numeric_columns']:
            try:
                mean_val = df[col].mean()
                median_val = df[col].median()
                min_val = df[col].min()
                max_val = df[col].max()
                std_val = df[col].std()
                missing = df[col].isna().sum()
                missing_pct = (missing / row_count) * 100
                
                description += f"- {col}: mean={mean_val:.2f}, median={median_val:.2f}, min={min_val:.2f}, "
                description += f"max={max_val:.2f}, std={std_val:.2f}, missing={missing_pct:.2f}%\n"
            except:
                description += f"- {col}: [Error calculating statistics]\n"
    
    # Categorical columns
    if validation_results['categorical_columns']:
        description += "\nCategorical columns:\n"
        for col in validation_results['categorical_columns']:
            try:
                unique_vals = df[col].nunique()
                top_vals = df[col].value_counts().head(3).to_dict()
                top_vals_str = ", ".join([f"'{k}': {v}" for k, v in top_vals.items()])
                missing = df[col].isna().sum()
                missing_pct = (missing / row_count) * 100
                
                description += f"- {col}: {unique_vals} unique values, top values: {top_vals_str}, "
                description += f"missing={missing_pct:.2f}%\n"
            except:
                description += f"- {col}: [Error calculating statistics]\n"
    
    # Date columns
    if validation_results['date_columns']:
        description += "\nDate columns:\n"
        for col in validation_results['date_columns']:
            try:
                # Convert to datetime
                date_col = pd.to_datetime(df[col], errors='coerce')
                min_date = date_col.min()
                max_date = date_col.max()
                missing = date_col.isna().sum()
                missing_pct = (missing / row_count) * 100
                
                description += f"- {col}: range from {min_date} to {max_date}, "
                description += f"missing={missing_pct:.2f}%\n"
            except:
                description += f"- {col}: [Error calculating statistics]\n"
    
    # Text columns
    if validation_results['text_columns']:
        description += "\nText columns:\n"
        for col in validation_results['text_columns']:
            try:
                # Convert to string and calculate average length
                text_col = df[col].astype(str)
                avg_length = text_col.str.len().mean()
                missing = df[col].isna().sum()
                missing_pct = (missing / row_count) * 100
                
                description += f"- {col}: average length={avg_length:.2f} chars, "
                description += f"missing={missing_pct:.2f}%\n"
            except:
                description += f"- {col}: [Error calculating statistics]\n"
    
    # Correlation information for numeric columns
    if len(validation_results['numeric_columns']) >= 2:
        description += "\nCorrelation information:\n"
        try:
            corr_matrix = df[validation_results['numeric_columns']].corr()
            
            # Get top 5 strongest correlations (excluding self-correlations)
            corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corrs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
            
            # Sort by absolute correlation value
            corrs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # Add top correlations to description
            for i, (col1, col2, corr_val) in enumerate(corrs[:5]):
                description += f"- {col1} and {col2}: {corr_val:.4f}\n"
        except:
            description += "- [Error calculating correlations]\n"
    
    return description

def generate_ai_insights(dataset_description, question_type):
    """
    Generate AI insights based on the dataset description
    
    Parameters:
    -----------
    dataset_description : str
        Detailed description of the dataset
    question_type : str
        Type of insights to generate (general or custom question)
    
    Returns:
    --------
    str : Generated insights or None if error
    """
    api_key = get_openai_api_key()
    if not api_key:
        return None
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Prepare prompt based on question type
        if question_type == "general":
            prompt = f"""
            You are a data analytics expert. Based on the following dataset description, 
            identify key insights, patterns, and recommendations.
            
            DATASET DESCRIPTION:
            {dataset_description}
            
            Please provide a comprehensive analysis that includes:
            1. Key patterns and trends identified in the data
            2. Notable relationships between variables
            3. Potential anomalies or areas of concern
            4. Business implications of these findings
            5. Specific, actionable recommendations based on the data
            
            Format your response in markdown with clear headings and bullet points.
            """
        else:
            # Custom question
            prompt = f"""
            You are a data analytics expert. Based on the following dataset description, 
            answer this specific question:
            
            QUESTION:
            {question_type}
            
            DATASET DESCRIPTION:
            {dataset_description}
            
            Provide a thorough, data-driven answer that includes:
            1. Direct response to the question based on the dataset
            2. Supporting evidence from the data
            3. Any relevant context or caveats
            4. Actionable recommendations where appropriate
            
            Format your response in markdown with clear headings and bullet points.
            """
        
        # Call OpenAI API
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data analytics expert providing insights from data analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800
        )
        
        # Return the generated insights
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"Error generating AI insights: {str(e)}")
        return None

def generate_natural_language_summary(df, validation_results, dataset_description):
    """
    Generate a natural language summary of the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to summarize
    validation_results : dict
        Results from data validation including column classifications
    dataset_description : str
        Pregenerated dataset description
    
    Returns:
    --------
    str : Natural language summary or None if error
    """
    api_key = get_openai_api_key()
    if not api_key:
        return None
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Enhance dataset description with specific request
        prompt = f"""
        You are a data analytics expert specializing in creating clear, natural language summaries.
        Please create a comprehensive narrative summary of this dataset.
        
        DATASET DESCRIPTION:
        {dataset_description}
        
        Your natural language summary should:
        1. Describe the dataset in plain language that a non-technical person could understand
        2. Explain key distributions, central tendencies, and patterns
        3. Highlight relationships between variables
        4. Note any interesting outliers or anomalies
        5. Explain what the data represents in a business or real-world context
        
        Format your response in markdown with clear paragraphs. Use a conversational, 
        narrative style rather than bullet points.
        """
        
        # Call OpenAI API
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data analytics expert creating clear, narrative summaries of data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        
        # Return the generated summary
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"Error generating natural language summary: {str(e)}")
        return None

def generate_executive_summary(df, validation_results, dataset_description):
    """
    Generate an executive summary of the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to summarize
    validation_results : dict
        Results from data validation including column classifications
    dataset_description : str
        Pregenerated dataset description
    
    Returns:
    --------
    str : Executive summary or None if error
    """
    api_key = get_openai_api_key()
    if not api_key:
        return None
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Prepare prompt for executive summary
        prompt = f"""
        You are a senior business analyst creating an executive summary for C-level executives.
        Based on the following dataset analysis, provide a concise, business-focused executive summary.
        
        DATASET DESCRIPTION:
        {dataset_description}
        
        Your executive summary should:
        1. Begin with a brief overview of what the data represents (1-2 sentences)
        2. Highlight 3-5 key findings with clear business implications
        3. Identify critical metrics and their performance
        4. Provide 3-4 specific, actionable recommendations
        5. Suggest next steps or areas for further investigation
        
        Format your response in markdown. Use clear, concise language appropriate for busy executives.
        Focus on business impact rather than technical details. Keep the entire summary under 500 words.
        """
        
        # Call OpenAI API
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a senior business analyst creating executive summaries for C-level executives."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=700
        )
        
        # Return the generated executive summary
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"Error generating executive summary: {str(e)}")
        return None

def generate_key_driver_analysis(df, target_variable, driver_variables, dataset_description):
    """
    Generate a key driver analysis for the target variable
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    target_variable : str
        The target variable to predict/explain
    driver_variables : list
        List of potential driver variables
    dataset_description : str
        Detailed description of the dataset
    
    Returns:
    --------
    str : Key driver analysis or None if error
    """
    api_key = get_openai_api_key()
    if not api_key:
        return None
    
    try:
        # Calculate correlations for numeric variables
        corr_data = []
        numeric_drivers = [var for var in driver_variables if var in df.select_dtypes(include=np.number).columns]
        
        if numeric_drivers:
            correlations = df[numeric_drivers + [target_variable]].corr()[target_variable].sort_values(ascending=False)
            
            # Exclude the target variable's self-correlation
            correlations = correlations[correlations.index != target_variable]
            
            for var, corr in correlations.items():
                corr_data.append(f"{var}: correlation = {corr:.4f}")
        
        # Prepare correlation information
        correlation_info = "\n".join(corr_data)
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Prepare prompt for key driver analysis
        prompt = f"""
        You are a data science expert specializing in key driver analysis.
        Please analyze what factors most influence the target variable.
        
        TARGET VARIABLE: {target_variable}
        
        POTENTIAL DRIVER VARIABLES:
        {', '.join(driver_variables)}
        
        CORRELATION WITH TARGET (for numeric variables):
        {correlation_info}
        
        DATASET DESCRIPTION:
        {dataset_description}
        
        Your key driver analysis should:
        1. Identify the top 3-5 variables that most strongly influence the target
        2. Explain the nature and direction of each relationship
        3. Provide business interpretation of why these drivers matter
        4. Suggest specific actions that could be taken based on these drivers
        5. Note any potential confounding factors or limitations
        
        Format your response in markdown with clear headings and sections.
        """
        
        # Call OpenAI API
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data science expert specializing in key driver analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800
        )
        
        # Return the generated key driver analysis
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"Error generating key driver analysis: {str(e)}")
        return None

def generate_text_analysis(text_description):
    """
    Generate an analysis of text data
    
    Parameters:
    -----------
    text_description : str
        Description of the text data to analyze
    
    Returns:
    --------
    str : Text analysis or None if error
    """
    api_key = get_openai_api_key()
    if not api_key:
        return None
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Prepare prompt for text analysis
        prompt = f"""
        You are a text analytics expert specializing in sentiment analysis and theme extraction.
        Please analyze the following text data.
        
        {text_description}
        
        Your text analysis should include:
        1. Overall sentiment assessment (positive, negative, neutral) with explanation
        2. Key themes and topics identified, with examples
        3. Frequently occurring terms and their context
        4. Any notable patterns or trends
        5. Business implications and recommendations based on this analysis
        
        Format your response in markdown with clear headings and sections.
        """
        
        # Call OpenAI API
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a text analytics expert specializing in sentiment analysis and theme extraction."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800
        )
        
        # Return the generated text analysis
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"Error generating text analysis: {str(e)}")
        return None
