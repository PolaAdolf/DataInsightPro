# Overview

This is a comprehensive data analytics platform built with Streamlit that provides end-to-end data analysis capabilities. The platform offers five types of analytics: descriptive, diagnostic, predictive, prescriptive, and cognitive analytics, along with an AI-powered work plan generator. Users can upload CSV or Excel files and receive interactive visualizations, statistical insights, machine learning predictions, and actionable recommendations for business improvement.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit-based web application with responsive design
- **Layout**: Wide layout with expandable sidebar navigation
- **State Management**: Session state variables for data persistence across interactions
- **User Interface**: Tab-based navigation with file upload, validation, and multiple analytics views
- **Visualization**: Plotly for interactive charts and graphs with responsive design

## Backend Architecture
- **Modular Design**: Separated analytics modules (descriptive, diagnostic, predictive, prescriptive, cognitive)
- **Utility Layer**: Dedicated modules for data loading, validation, visualization, and work plan generation
- **Data Processing Pipeline**: Sequential flow from upload → validation → analytics → insights
- **Error Handling**: Comprehensive validation and error checking throughout the pipeline

## Data Processing
- **File Support**: CSV and Excel file upload capabilities
- **Data Validation**: Multi-level validation including schema checks, missing value analysis, and column classification
- **Column Classification**: Automatic detection of numeric, categorical, date, and text columns
- **Data Preprocessing**: Automated cleaning including whitespace removal and empty column dropping

## Machine Learning Integration
- **Scikit-learn**: Core ML library for predictive analytics including regression, classification, and clustering
- **XGBoost**: Advanced gradient boosting for enhanced predictions
- **Time Series**: ARIMA modeling for forecasting
- **Feature Engineering**: Automated preprocessing pipelines with scaling and encoding

## Analytics Modules
- **Descriptive Analytics**: Statistical summaries, distributions, and basic visualizations
- **Diagnostic Analytics**: Correlation analysis, hypothesis testing, and relationship exploration
- **Predictive Analytics**: Machine learning models for regression, classification, and time series forecasting
- **Prescriptive Analytics**: Optimization algorithms and clustering for actionable recommendations
- **Cognitive Analytics**: AI-powered natural language insights and pattern recognition

# External Dependencies

## AI Services
- **OpenAI API**: Powers cognitive analytics and work plan generation with GPT models
- **API Key Management**: Environment variable-based configuration for secure API access

## Python Libraries
- **Streamlit**: Web application framework for the user interface
- **Pandas/Numpy**: Core data manipulation and numerical computing
- **Plotly**: Interactive visualization library for charts and graphs
- **Scikit-learn**: Machine learning algorithms and preprocessing utilities
- **XGBoost**: Gradient boosting framework for advanced predictions
- **Statsmodels**: Statistical modeling, particularly for time series analysis
- **SciPy**: Scientific computing for statistical tests and optimization

## Data Sources
- **File Upload**: Local CSV and Excel file processing
- **No Database**: Application operates on uploaded files without persistent storage
- **Session Storage**: Temporary data persistence during user sessions

## Deployment
- **Port Configuration**: Configured to run on port 5000
- **Environment Variables**: Requires OPENAI_API_KEY for AI-powered features
- **Requirements**: Dependencies managed through pyproject.toml