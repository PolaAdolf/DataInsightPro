import pandas as pd
import streamlit as st

def load_data(uploaded_file):
    """
    Load data from an uploaded file (CSV or Excel)
    
    Parameters:
    -----------
    uploaded_file : UploadedFile
        The file uploaded by the user
    
    Returns:
    --------
    pandas.DataFrame
        The loaded data as a pandas DataFrame
    """
    try:
        # Get file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Load based on file type
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension == 'xlsx':
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Basic preprocessing
        # Convert column names to string and strip whitespace
        df.columns = [str(col).strip() for col in df.columns]
        
        # Drop columns that are entirely empty
        df = df.dropna(axis=1, how='all')
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        raise e
