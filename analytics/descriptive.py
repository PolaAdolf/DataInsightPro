import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.visualization import create_countplot, create_histogram, create_boxplot, create_piechart, create_heatmap

def perform_descriptive_analytics(df, validation_results):
    """
    Perform descriptive analytics on the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    validation_results : dict
        Results from data validation including column classifications
    """
    st.subheader("Statistical Summary")
    
    # Summary statistics for numeric columns
    if validation_results['numeric_columns']:
        with st.expander("Numeric Data Summary", expanded=True):
            st.dataframe(df[validation_results['numeric_columns']].describe())
            
            # Distribution of a selected numeric column
            selected_num_col = st.selectbox(
                "Select a numeric column to visualize its distribution:",
                validation_results['numeric_columns']
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Histogram")
                fig = create_histogram(df, selected_num_col)
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.write("Box Plot")
                fig = create_boxplot(df, selected_num_col)
                st.plotly_chart(fig, use_container_width=True)
                
            # Descriptive statistics details
            st.write("**Detailed Statistics:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                mean_val = df[selected_num_col].mean()
                median_val = df[selected_num_col].median()
                st.metric("Mean", f"{mean_val:.2f}")
                st.metric("Median", f"{median_val:.2f}")
            
            with col2:
                min_val = df[selected_num_col].min()
                max_val = df[selected_num_col].max()
                st.metric("Minimum", f"{min_val:.2f}")
                st.metric("Maximum", f"{max_val:.2f}")
            
            with col3:
                std_val = df[selected_num_col].std()
                range_val = max_val - min_val
                st.metric("Standard Deviation", f"{std_val:.2f}")
                st.metric("Range", f"{range_val:.2f}")
    
    # Summary for categorical columns
    if validation_results['categorical_columns']:
        with st.expander("Categorical Data Summary", expanded=True):
            selected_cat_col = st.selectbox(
                "Select a categorical column to visualize:",
                validation_results['categorical_columns']
            )
            
            # Calculate value counts
            value_counts = df[selected_cat_col].value_counts()
            
            # Display counts as a table
            st.write("**Value Counts:**")
            st.dataframe(value_counts.reset_index().rename(columns={"index": selected_cat_col, selected_cat_col: "Count"}))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Bar Chart")
                fig = create_countplot(df, selected_cat_col)
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.write("Pie Chart")
                fig = create_piechart(df, selected_cat_col)
                st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis if date columns exist
    if validation_results['date_columns']:
        with st.expander("Time Series Analysis", expanded=True):
            date_col = st.selectbox(
                "Select a date column:",
                validation_results['date_columns']
            )
            
            # Try to convert to datetime
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                
                # Select a numeric column for time series
                if validation_results['numeric_columns']:
                    time_series_col = st.selectbox(
                        "Select a numeric column to analyze over time:",
                        validation_results['numeric_columns']
                    )
                    
                    # Group by date and calculate statistics
                    time_data = df.set_index(date_col)[time_series_col].resample('D').mean()
                    
                    # Plot time series
                    fig = px.line(
                        time_data.reset_index(), 
                        x=date_col, 
                        y=time_series_col,
                        title=f"{time_series_col} Over Time"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate rolling average
                    window_size = min(30, len(time_data) // 2) if len(time_data) > 5 else 1
                    time_data_rolling = time_data.rolling(window=window_size).mean()
                    
                    # Plot with trend
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=time_data.index, 
                        y=time_data.values,
                        mode='lines',
                        name='Original'
                    ))
                    
                    if len(time_data) > 5:  # Only add trend line if enough data
                        fig.add_trace(go.Scatter(
                            x=time_data_rolling.index,
                            y=time_data_rolling.values,
                            mode='lines',
                            name=f'{window_size}-Day Rolling Average',
                            line=dict(width=3, color='red')
                        ))
                    
                    fig.update_layout(
                        title=f"{time_series_col} with Trend Analysis",
                        xaxis_title=date_col,
                        yaxis_title=time_series_col
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No numeric columns available for time series analysis")
            except Exception as e:
                st.error(f"Error processing date column: {str(e)}")
    
    # Missing value analysis
    with st.expander("Missing Value Analysis", expanded=True):
        # Calculate missing values
        missing = df.isna().sum().reset_index()
        missing.columns = ['Column', 'Missing Count']
        missing['Missing Percentage'] = (missing['Missing Count'] / len(df)) * 100
        missing = missing.sort_values('Missing Percentage', ascending=False)
        
        # Only show columns with missing values
        missing = missing[missing['Missing Count'] > 0]
        
        if len(missing) > 0:
            st.write("**Columns with Missing Values:**")
            st.dataframe(missing)
            
            # Visualize missing values
            fig = px.bar(
                missing, 
                x='Column', 
                y='Missing Percentage',
                title="Missing Values by Column (%)",
                color='Missing Percentage',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found in the dataset!")
    
    # Correlation matrix for numeric columns
    if len(validation_results['numeric_columns']) > 1:
        with st.expander("Correlation Analysis", expanded=True):
            st.write("**Correlation Matrix:**")
            
            # Calculate correlation
            corr_matrix = df[validation_results['numeric_columns']].corr()
            
            # Create heatmap
            fig = create_heatmap(corr_matrix)
            st.plotly_chart(fig, use_container_width=True)
            
            # Highlight strong correlations
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        strong_correlations.append({
                            'Variables': f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]}",
                            'Correlation': corr_matrix.iloc[i, j]
                        })
            
            if strong_correlations:
                st.write("**Strong Correlations (|r| > 0.7):**")
                st.dataframe(pd.DataFrame(strong_correlations))
            else:
                st.info("No strong correlations found in the dataset.")
    
    # Data distribution overview
    with st.expander("Data Distribution Overview", expanded=True):
        st.write("**Data Type Distribution:**")
        
        # Create a summary of column types
        type_counts = {
            'Numeric Columns': len(validation_results['numeric_columns']),
            'Categorical Columns': len(validation_results['categorical_columns']),
            'Date Columns': len(validation_results['date_columns']),
            'Text Columns': len(validation_results['text_columns'])
        }
        
        # Visualize column type distribution
        fig = px.pie(
            values=list(type_counts.values()),
            names=list(type_counts.keys()),
            title="Column Type Distribution",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display column lists by type
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numeric Columns:**")
            st.write(", ".join(validation_results['numeric_columns']) if validation_results['numeric_columns'] else "None")
            
            st.write("**Date Columns:**")
            st.write(", ".join(validation_results['date_columns']) if validation_results['date_columns'] else "None")
        
        with col2:
            st.write("**Categorical Columns:**")
            st.write(", ".join(validation_results['categorical_columns']) if validation_results['categorical_columns'] else "None")
            
            st.write("**Text Columns:**")
            st.write(", ".join(validation_results['text_columns']) if validation_results['text_columns'] else "None")
