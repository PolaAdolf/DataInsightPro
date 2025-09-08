import json
import logging
import os

from google import genai
from google.genai import types
from pydantic import BaseModel


# IMPORTANT: KEEP THIS COMMENT
# Follow these instructions when using this blueprint:
# - Note that the newest Gemini model series is "gemini-2.5-flash" or gemini-2.5-pro"
#   - do not change this unless explicitly requested by the user
# - Sometimes the google genai SDK has occasional type errors. You might need to run to validate, at time.  
# The SDK was recently renamed from google-generativeai to google-genai. This file reflects the new name and the new APIs.

# This API key is from Gemini Developer API Key, not vertex AI API Key
# Client will be initialized in individual functions when needed


def get_gemini_api_key():
    """Get Gemini API key from environment variable"""
    return os.getenv("GEMINI_API_KEY")


def generate_insights(dataset_description: str, question_type: str) -> str:
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
    api_key = get_gemini_api_key()
    if not api_key:
        return "Gemini API key not found. Please add it to your environment variables."
    
    try:
        # Initialize Gemini client
        client = genai.Client(api_key=api_key)
        
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
        
        # Call Gemini API
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        # Return the generated insights
        return response.text or "Unable to generate insights"
    
    except Exception as e:
        return f"Error generating AI insights: {str(e)}"


def generate_natural_language_summary(dataset_description: str) -> str:
    """
    Generate a natural language summary of the dataset
    
    Parameters:
    -----------
    dataset_description : str
        Detailed description of the dataset
    
    Returns:
    --------
    str : Generated summary or None if error
    """
    api_key = get_gemini_api_key()
    if not api_key:
        return "Gemini API key not found. Please add it to your environment variables."
    
    try:
        # Initialize Gemini client
        client = genai.Client(api_key=api_key)
        
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
        
        # Call Gemini API
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        # Return the generated summary
        return response.text or "Unable to generate summary"
    
    except Exception as e:
        return f"Error generating natural language summary: {str(e)}"


def generate_executive_summary(dataset_description: str) -> str:
    """
    Generate an executive summary of the dataset analysis
    
    Parameters:
    -----------
    dataset_description : str
        Detailed description of the dataset
    
    Returns:
    --------
    str : Generated executive summary or None if error
    """
    api_key = get_gemini_api_key()
    if not api_key:
        return "Gemini API key not found. Please add it to your environment variables."
    
    try:
        # Initialize Gemini client
        client = genai.Client(api_key=api_key)
        
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
        
        # Call Gemini API
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        # Return the generated executive summary
        return response.text or "Unable to generate executive summary"
    
    except Exception as e:
        return f"Error generating executive summary: {str(e)}"


def generate_key_driver_analysis(target_variable: str, driver_variables: list, correlation_info: str, dataset_description: str) -> str:
    """
    Generate a key driver analysis
    
    Parameters:
    -----------
    target_variable : str
        The target variable to analyze
    driver_variables : list
        List of potential driver variables
    correlation_info : str
        Correlation information with target
    dataset_description : str
        Detailed description of the dataset
    
    Returns:
    --------
    str : Generated key driver analysis or None if error
    """
    api_key = get_gemini_api_key()
    if not api_key:
        return "Gemini API key not found. Please add it to your environment variables."
    
    try:
        # Initialize Gemini client
        client = genai.Client(api_key=api_key)
        
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
        
        # Call Gemini API
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        # Return the generated key driver analysis
        return response.text or "Unable to generate key driver analysis"
    
    except Exception as e:
        return f"Error generating key driver analysis: {str(e)}"


def generate_text_analysis(text_description: str) -> str:
    """
    Generate text analysis insights
    
    Parameters:
    -----------
    text_description : str
        Description of the text data to analyze
    
    Returns:
    --------
    str : Generated text analysis or None if error
    """
    api_key = get_gemini_api_key()
    if not api_key:
        return "Gemini API key not found. Please add it to your environment variables."
    
    try:
        # Initialize Gemini client
        client = genai.Client(api_key=api_key)
        
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
        
        # Call Gemini API
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        # Return the generated text analysis
        return response.text or "Unable to generate text analysis"
    
    except Exception as e:
        return f"Error generating text analysis: {str(e)}"


def generate_prescriptive_insights(data_description: str, question: str) -> str:
    """
    Generate prescriptive insights using Gemini API
    
    Parameters:
    -----------
    data_description : str
        Description of the data
    question : str
        The specific question to ask
        
    Returns:
    --------
    str : The generated insights
    """
    # Check if API key is available
    api_key = get_gemini_api_key()
    if not api_key:
        return "Gemini API key not found. Please add it to your environment variables."
    
    try:
        # Initialize Gemini client
        client = genai.Client(api_key=api_key)
        
        # Create prompt
        prompt = f"""
        You are a data analytics expert. Based on the following data description, 
        provide specific, actionable recommendations.
        
        DATA DESCRIPTION:
        {data_description}
        
        QUESTION:
        {question}
        
        Your response should:
        1. Include 3-5 specific, data-driven recommendations
        2. For each recommendation, explain the rationale based on the data
        3. Suggest how to implement each recommendation
        4. Include potential impact or expected outcomes
        
        Format your response in bullet points and use business language.
        """
        
        # Call Gemini API
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        # Return the generated text
        return response.text or "Unable to generate insights"
    
    except Exception as e:
        return f"Error generating insights: {str(e)}"


def generate_work_plan(data_summary: str, plan_focus: str, timeline: str, context: str) -> str:
    """
    Generate a work plan using Gemini AI
    
    Parameters:
    -----------
    data_summary : str
        Summary of the dataset
    plan_focus : str
        Focus area for the work plan
    timeline : str
        Implementation timeline
    context : str
        Additional context or specific goals
    
    Returns:
    --------
    str or None : Generated work plan or None if error
    """
    api_key = get_gemini_api_key()
    if not api_key:
        return "Gemini API key not found. Please add it to your environment variables."
    
    try:
        # Initialize Gemini client
        client = genai.Client(api_key=api_key)
        
        # Determine timeline in months
        if "Short-term" in timeline:
            months = 3
        elif "Medium-term" in timeline:
            months = 6
        else:  # Long-term
            months = 12
        
        # Create prompt
        prompt = f"""
        You are a senior business consultant creating a detailed work implementation plan.
        
        DATA SUMMARY:
        {data_summary}
        
        PLAN FOCUS: {plan_focus}
        TIMELINE: {timeline} ({months} months)
        ADDITIONAL CONTEXT: {context if context else 'No additional context provided.'}
        
        Create a comprehensive work plan that includes:
        
        1. Executive Summary: A brief overview of the plan's goals and expected outcomes
        
        2. Phases of Implementation: Break down the {months}-month timeline into logical phases with specific goals for each phase
        
        3. Detailed Task List: For each phase, list specific tasks with:
           - Task description
           - Estimated duration (in days or weeks)
           - Required resources or skills
           - Dependencies on other tasks
           - Priority level (High, Medium, Low)
        
        4. Key Performance Indicators (KPIs): 3-5 specific metrics to track progress and success
        
        5. Risk Assessment: Identify potential risks and mitigation strategies
        
        6. Resource Requirements: Specify team roles, tools, and budget considerations
        
        Format your response in JSON with the following structure:
        {{
            "executive_summary": "summary text",
            "phases": [
                {{
                    "name": "Phase 1: [Phase Name]",
                    "description": "phase description",
                    "start_month": 1,
                    "end_month": 2,
                    "goals": ["goal 1", "goal 2"],
                    "tasks": [
                        {{
                            "task": "task description",
                            "duration": "X weeks",
                            "resources": ["resource 1", "resource 2"],
                            "dependencies": ["dependency 1"],
                            "priority": "High"
                        }}
                    ]
                }}
            ],
            "kpis": [
                {{
                    "name": "KPI name",
                    "description": "KPI description",
                    "target": "target value",
                    "measurement": "how to measure"
                }}
            ],
            "risks": [
                {{
                    "risk": "risk description",
                    "probability": "Low/Medium/High",
                    "impact": "Low/Medium/High",
                    "mitigation": "mitigation strategy"
                }}
            ],
            "resources": {{
                "team_roles": ["role 1", "role 2"],
                "tools_required": ["tool 1", "tool 2"],
                "budget_considerations": ["consideration 1", "consideration 2"]
            }}
        }}
        
        Ensure the JSON is valid and complete. Base all recommendations on the data insights provided.
        """
        
        # Call Gemini API
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt
        )
        
        # Return the generated work plan
        return response.text or "Unable to generate work plan"
    
    except Exception as e:
        return f"Error generating work plan: {str(e)}"