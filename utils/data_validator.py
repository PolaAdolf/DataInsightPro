import pandas as pd
import numpy as np
import re
from datetime import datetime

def validate_data(df):
    """
    Validate the uploaded data for analysis readiness
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The data to validate
    
    Returns:
    --------
    dict
        Validation results including validity flag, issues, and column classifications
    """
    validation = {
        'is_valid': True,
        'issues': [],
        'numeric_columns': [],
        'categorical_columns': [],
        'date_columns': [],
        'text_columns': []
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation['is_valid'] = False
        validation['issues'].append("The uploaded file is empty")
        return validation
    
    # Check for too few rows
    if df.shape[0] < 5:
        validation['is_valid'] = False
        validation['issues'].append("The dataset has too few rows (minimum 5 required)")
    
    # Check for too few columns
    if df.shape[1] < 2:
        validation['is_valid'] = False
        validation['issues'].append("The dataset has too few columns (minimum 2 required)")
    
    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        validation['is_valid'] = False
        validation['issues'].append("The dataset contains duplicate column names")
    
    # Check for high proportion of missing values
    missing_proportion = df.isna().mean()
    high_missing_cols = missing_proportion[missing_proportion > 0.5].index.tolist()
    
    if high_missing_cols:
        validation['is_valid'] = False
        validation['issues'].append(f"The following columns have over 50% missing values: {', '.join(high_missing_cols)}")
    
    # Check for overall missing values
    total_missing = df.isna().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_percentage = (total_missing / total_cells) * 100
    
    if missing_percentage > 30:
        validation['is_valid'] = False
        validation['issues'].append(f"The dataset has {missing_percentage:.2f}% missing values overall")
    
    # Identify data types for columns
    for column in df.columns:
        # Try to convert to numeric
        numeric_values = pd.to_numeric(df[column], errors='coerce')
        
        # Count non-NaN values after conversion
        valid_numeric_count = (~numeric_values.isna()).sum()
        
        # If most values convert to numeric successfully, classify as numeric
        if valid_numeric_count / len(df) > 0.7:
            validation['numeric_columns'].append(column)
            continue
        
        # Try to detect dates
        is_date = False
        if df[column].dtype == 'object':
            # Sample some values for date detection
            sample_values = df[column].dropna().sample(min(10, len(df[column].dropna()))).astype(str)
            date_count = 0
            
            for val in sample_values:
                try:
                    # Try various date formats
                    pd.to_datetime(val)
                    date_count += 1
                except:
                    pass
            
            # If most sampled values are dates, classify as date
            if date_count / len(sample_values) > 0.7:
                validation['date_columns'].append(column)
                is_date = True
        
        if not is_date:
            # Check if it's a categorical variable
            unique_ratio = df[column].nunique() / len(df)
            
            if unique_ratio < 0.2:  # Less than 20% unique values is categorical
                validation['categorical_columns'].append(column)
            else:
                # Likely text data
                validation['text_columns'].append(column)
    
    # Ensure there's at least one numeric column for analysis
    if len(validation['numeric_columns']) == 0:
        validation['is_valid'] = False
        validation['issues'].append("The dataset doesn't contain any numeric columns for analysis")
    
    # Check if there's at least one potential feature column and one target column
    if len(validation['numeric_columns'] + validation['categorical_columns']) < 2:
        validation['issues'].append("The dataset should have at least two columns that can be used as features/target")
    
    # Check for 100% duplicate rows
    if df.duplicated().sum() / len(df) > 0.9:
        validation['issues'].append("The dataset contains mostly duplicate rows")
    
    return validation
