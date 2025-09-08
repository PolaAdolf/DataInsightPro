import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.visualization import create_countplot, create_histogram, create_boxplot, create_piechart, create_heatmap
from utils.smart_preprocessor import SmartDataPreprocessor

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
    st.subheader("📈 Data Overview")
    
    # Smart preprocessing first
    with st.expander("🔧 Smart Data Preparation", expanded=False):
        st.markdown("**Want to make your data analysis even better?** Let me check for improvements!")
        
        if st.button("🔍 Run Smart Data Check", key="smart_check_desc"):
            preprocessor = SmartDataPreprocessor(df)
            health_report = preprocessor.analyze_data_health()
            preprocessor.show_issues_and_solutions()
            preprocessor.auto_scale_features()
            preprocessor.show_applied_fixes()
            
            # Update the dataframe if fixes were applied
            processed_df = preprocessor.get_processed_data()
            if len(processed_df) != len(df) or not processed_df.equals(df):
                st.session_state.data = processed_df
                st.success("🎉 Your data has been enhanced! The analysis below uses the improved data.")
                st.info("💡 The improvements are automatically applied to all other analysis sections too!")
                st.rerun()
    
    # Summary statistics for numeric columns
    if validation_results['numeric_columns']:
        with st.expander("🔢 Number Columns Analysis", expanded=True):
            st.write("**Quick Statistics for All Number Columns:**")
            st.dataframe(df[validation_results['numeric_columns']].describe())
            
            # Add customization options
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_num_col = st.selectbox(
                    "Choose a number column to explore:",
                    validation_results['numeric_columns']
                )
            with col2:
                chart_color_theme = st.selectbox(
                    "Pick a color theme:",
                    ["Blues", "Reds", "Greens", "Purples", "Oranges"],
                    index=0
                )
                
                # Map theme to actual colors
                color_mapping = {
                    "Blues": "steelblue",
                    "Reds": "crimson", 
                    "Greens": "forestgreen",
                    "Purples": "mediumpurple",
                    "Oranges": "darkorange"
                }
                chart_color = color_mapping[chart_color_theme]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("📊 **Distribution Chart** (shows how values are spread)")
                try:
                    fig = create_histogram(df, selected_num_col)
                    # Apply color theme safely
                    fig.update_traces(marker_color=chart_color)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add interpretation
                    st.markdown(f"""
                    💡 **What this chart tells you:**
                    - Shows how often different values appear in {selected_num_col}
                    - Tall bars = more common values
                    - Wide spread = lots of variety in your data
                    - Narrow peaks = most values are similar
                    """)
                except Exception as e:
                    st.error("😅 Oops! There was an issue creating this chart. This might happen if your data has some unusual values.")
                    st.info("💡 Try selecting a different column or check your data for any empty cells.")
                
            with col2:
                st.write("📦 **Box Chart** (shows outliers and ranges)")
                try:
                    fig = create_boxplot(df, selected_num_col)
                    # Apply color theme safely
                    fig.update_traces(marker_color=chart_color)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add interpretation
                    st.markdown(f"""
                    💡 **What this chart tells you:**
                    - The box shows where most of your values fall
                    - The line in the middle is the median (middle value)
                    - Dots outside the box are unusual values (outliers)
                    - Longer box = more spread in your data
                    """)
                except Exception as e:
                    st.error("😅 Oops! There was an issue creating this chart. This might happen if your data has some unusual values.")
                    st.info("💡 Try selecting a different column or check your data for any empty cells.")
                
            # Descriptive statistics details with error handling
            st.write("📊 **Key Numbers Explained:**")
            
            try:
                # Convert to numeric if needed and handle errors
                numeric_data = pd.to_numeric(df[selected_num_col], errors='coerce')
                
                if numeric_data.isna().all():
                    st.error("😅 This column contains text or mixed data types. Please select a column with only numbers.")
                    st.info("💡 Look for columns like 'price', 'quantity', 'age', 'score', etc.")
                else:
                    # Remove any NaN values for calculations
                    clean_data = numeric_data.dropna()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        mean_val = clean_data.mean()
                        median_val = clean_data.median()
                        st.metric("Average (Mean)", f"{mean_val:.2f}")
                        st.metric("Middle Value (Median)", f"{median_val:.2f}")
                    
                    with col2:
                        min_val = clean_data.min()
                        max_val = clean_data.max()
                        st.metric("Lowest Value", f"{min_val:.2f}")
                        st.metric("Highest Value", f"{max_val:.2f}")
                    
                    with col3:
                        std_val = clean_data.std()
                        range_val = max_val - min_val
                        st.metric("Variation (Std Dev)", f"{std_val:.2f}")
                        st.metric("Total Range", f"{range_val:.2f}")
                    
                    # Add simple explanations
                    st.markdown("### 📝 What these numbers mean:")
                    st.markdown(f"""
                    - **Average:** The typical value is {mean_val:.2f}
                    - **Middle Value:** Half the values are above {median_val:.2f} and half are below
                    - **Range:** Values go from {min_val:.2f} to {max_val:.2f}
                    - **Variation:** Shows how spread out the values are (lower = more similar values)
                    """)
                    
                    if len(clean_data) < len(df[selected_num_col]):
                        missing_count = len(df[selected_num_col]) - len(clean_data)
                        st.info(f"ℹ️ Note: {missing_count} non-numeric values were excluded from calculations.")
                        
            except Exception as e:
                st.error("😅 There was an issue analyzing this column. It might contain mixed data types.")
                st.info("💡 Try selecting a different column with numeric values only.")
    
    # Summary for categorical columns
    if validation_results['categorical_columns']:
        with st.expander("🏷️ Category Columns Analysis", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_cat_col = st.selectbox(
                    "Choose a category column to explore:",
                    validation_results['categorical_columns']
                )
            with col2:
                chart_style = st.selectbox(
                    "Pick chart colors:",
                    ["Colorful", "Blues", "Greens", "Warm", "Cool"],
                    index=0
                )
            
            # Calculate value counts
            value_counts = df[selected_cat_col].value_counts()
            
            # Display counts as a table
            st.write("📊 **How many of each category:**")
            counts_table = value_counts.reset_index().rename(columns={"index": selected_cat_col, selected_cat_col: "Count"})
            counts_table['Percentage'] = (counts_table['Count'] / counts_table['Count'].sum() * 100).round(1)
            st.dataframe(counts_table)
            
            # Get color scheme
            color_schemes = {
                "Colorful": "Set3",
                "Blues": "Blues",
                "Greens": "Greens", 
                "Warm": "Reds",
                "Cool": "Blues"
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("📊 **Bar Chart** (compare amounts)")
                try:
                    fig = create_countplot(df, selected_cat_col)
                    # Apply color scheme
                    if chart_style != "Colorful":
                        fig.update_traces(marker_color=color_schemes[chart_style])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add interpretation for bar chart
                    most_common = value_counts.index[0]
                    most_common_count = value_counts.iloc[0]
                    total_count = value_counts.sum()
                    percentage = (most_common_count / total_count) * 100
                    
                    st.markdown(f"""
                    💡 **What this bar chart tells you:**
                    - **Most common category:** {most_common} ({most_common_count} occurrences, {percentage:.1f}% of your data)
                    - **Tallest bar** = most frequent category
                    - **Height differences** show how balanced or unbalanced your categories are
                    - **Business insight:** {'Your data is fairly balanced across categories' if percentage < 40 else f'Your data is heavily concentrated in "{most_common}"'}
                    """)
                except Exception as e:
                    st.error("😅 Could not create bar chart. This might happen with certain data types.")
                    st.info("💡 Try selecting a different category column.")
                
            with col2:
                st.write("🍰 **Pie Chart** (see proportions)")
                try:
                    fig = create_piechart(df, selected_cat_col)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add interpretation for pie chart
                    num_categories = len(value_counts)
                    largest_slice = (value_counts.iloc[0] / value_counts.sum()) * 100
                    
                    st.markdown(f"""
                    💡 **What this pie chart tells you:**
                    - **{num_categories} different categories** in your data
                    - **Largest slice:** {largest_slice:.1f}% of the total
                    - **Slice sizes** show relative importance of each category
                    - **Business insight:** {'Good diversity in your categories' if num_categories > 3 and largest_slice < 50 else 'One category dominates your data'}
                    """)
                except Exception as e:
                    st.error("😅 Could not create pie chart. This might happen with certain data types.")
                    st.info("💡 Try selecting a different category column.")
    
    # Time series analysis if date columns exist
    if validation_results['date_columns']:
        with st.expander("📅 Time Trends Analysis", expanded=True):
            date_col = st.selectbox(
                "Select a date column:",
                validation_results['date_columns']
            )
            
            # Try to convert to datetime
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                
                # Select a numeric column for time series
                if validation_results['numeric_columns']:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        time_series_col = st.selectbox(
                            "Choose what to track over time:",
                            validation_results['numeric_columns']
                        )
                    with col2:
                        show_trend = st.checkbox("📈 Show trend line", value=True)
                    
                    # Group by date and calculate statistics
                    time_data = df.set_index(date_col)[time_series_col].resample('D').mean()
                    
                    # Plot time series
                    fig = px.line(
                        time_data.reset_index(), 
                        x=date_col, 
                        y=time_series_col,
                        title=f"📈 How {time_series_col} Changes Over Time"
                    )
                    fig.update_traces(line_color='blue', line_width=2)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add time series interpretation
                    first_value = time_data.iloc[0] if len(time_data) > 0 else 0
                    last_value = time_data.iloc[-1] if len(time_data) > 0 else 0
                    change = last_value - first_value
                    change_pct = (change / first_value * 100) if first_value != 0 else 0
                    
                    st.markdown(f"""
                    📈 **Time Trend Analysis:**
                    - **Overall change:** {change:.2f} ({change_pct:+.1f}% from start to end)
                    - **Trend direction:** {'📈 Upward' if change > 0 else '📉 Downward' if change < 0 else '➡️ Stable'}
                    - **Business insight:** {'Growing trend - positive momentum' if change_pct > 10 else 'Declining trend - needs attention' if change_pct < -10 else 'Relatively stable over time'}
                    """)
                    
                    # Add seasonal pattern detection
                    if len(time_data) > 12:
                        seasonal_std = time_data.groupby(time_data.index.month).mean().std()
                        overall_std = time_data.std()
                        if seasonal_std > overall_std * 0.3:
                            st.info("🔄 **Seasonal Pattern Detected:** Your data shows regular patterns that repeat over months - this could be seasonal business cycles!")
                    
                    
                    # Calculate rolling average
                    if show_trend and len(time_data) > 5:
                        window_size = min(30, len(time_data) // 2)
                        time_data_rolling = time_data.rolling(window=window_size).mean()
                        
                        # Plot with trend
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=time_data.index, 
                            y=time_data.values,
                            mode='lines',
                            name='Actual Values',
                            line=dict(color='lightblue', width=1)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=time_data_rolling.index,
                            y=time_data_rolling.values,
                            mode='lines',
                            name='Trend Line (Smoothed)',
                            line=dict(width=3, color='red')
                        ))
                        
                        fig.update_layout(
                            title=f"📈 {time_series_col} Trend Over Time",
                            xaxis_title="Date",
                            yaxis_title=time_series_col
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add explanation
                        st.info(f"💡 **Tip:** The red line shows the general trend by smoothing out daily ups and downs over {window_size} days.")
                else:
                    st.info("📊 No number columns found to track over time. Upload data with both dates and numbers to see trends!")
            except Exception as e:
                st.error(f"Error processing date column: {str(e)}")
    
    # Missing value analysis
    with st.expander("🕳️ Empty Data Analysis", expanded=True):
        # Calculate missing values
        missing = df.isna().sum().reset_index()
        missing.columns = ['Column', 'Missing Count']
        missing['Missing Percentage'] = (missing['Missing Count'] / len(df)) * 100
        missing = missing.sort_values('Missing Percentage', ascending=False)
        
        # Only show columns with missing values
        missing = missing[missing['Missing Count'] > 0]
        
        if len(missing) > 0:
            st.write("⚠️ **Columns with empty cells:**")
            st.dataframe(missing)
            
            # Add color customization
            col1, col2 = st.columns([3, 1])
            with col2:
                missing_color = st.selectbox(
                    "Chart color:",
                    ["Reds", "Oranges", "Blues", "Purples"]
                )
            
            with col1:
                # Visualize missing values
                fig = px.bar(
                    missing, 
                    x='Column', 
                    y='Missing Percentage',
                    title="🕳️ Percentage of Empty Cells by Column",
                    color='Missing Percentage',
                    color_continuous_scale=missing_color
                )
                fig.update_layout(xaxis_title="Column Name", yaxis_title="% Empty")
                st.plotly_chart(fig, use_container_width=True)
            
            # Add helpful explanation
            st.markdown("📝 **What this means:** Empty cells can affect your analysis. Consider filling them with appropriate values or removing rows with too many empty cells.")
        else:
            st.success("✅ Great! No empty cells found in your data!")
    
    # Correlation matrix for numeric columns
    if len(validation_results['numeric_columns']) > 1:
        with st.expander("🔗 How Number Columns Relate to Each Other", expanded=True):
            st.write("🔗 **Relationship Strength Between Number Columns:**")
            st.markdown("Values close to 1 = strong positive relationship, close to -1 = strong negative relationship, close to 0 = no relationship")
            
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
                st.write("💪 **Strong Relationships Found (connection strength > 0.7):**")
                corr_df = pd.DataFrame(strong_correlations)
                corr_df['Strength'] = corr_df['Correlation'].apply(lambda x: 'Strong Positive' if x > 0.7 else 'Strong Negative')
                st.dataframe(corr_df)
                st.markdown("💡 **Tip:** These columns move together! When one changes, the other tends to change too.")
            else:
                st.info("🔍 No strong relationships found between your number columns.")
    
    # Data distribution overview
    with st.expander("📈 Your Data at a Glance", expanded=True):
        st.write("📊 **What types of data you have:**")
        
        # Create a summary of column types
        type_counts = {
            'Number Columns (for calculations)': len(validation_results['numeric_columns']),
            'Category Columns (groups/labels)': len(validation_results['categorical_columns']),
            'Date Columns (time-based)': len(validation_results['date_columns']),
            'Text Columns (words/descriptions)': len(validation_results['text_columns'])
        }
        
        # Add smart recommendations based on data types
        st.markdown("📝 **Smart Suggestions for Your Data:**")
        
        recommendations = []
        if len(validation_results['numeric_columns']) >= 2:
            recommendations.append("✅ You have multiple number columns - perfect for finding relationships and patterns!")
        if len(validation_results['categorical_columns']) >= 1:
            recommendations.append("✅ Your category columns are great for grouping and comparing different segments.")
        if len(validation_results['date_columns']) >= 1:
            recommendations.append("✅ With date columns, you can track trends and changes over time.")
        if len(validation_results['text_columns']) >= 1:
            recommendations.append("✅ Text columns can provide rich insights with AI analysis.")
        
        for rec in recommendations:
            st.markdown(rec)
        
        # Visualize column type distribution with customization
        col1, col2 = st.columns([3, 1])
        with col2:
            pie_colors = st.selectbox(
                "Pie chart style:",
                ["Bright", "Pastel", "Earth", "Ocean"]
            )
        
        color_schemes = {
            "Bright": "Set1",
            "Pastel": "Pastel1", 
            "Earth": "Set2",
            "Ocean": "Set3"
        }
        
        with col1:
            # Visualize column type distribution
            fig = px.pie(
                values=list(type_counts.values()),
                names=list(type_counts.keys()),
                title="📊 Your Data Types Overview",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Display column lists by type in a more user-friendly way
        st.markdown("### 📋 Your Columns by Type")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔢 Number Columns (for calculations):**")
            if validation_results['numeric_columns']:
                for col in validation_results['numeric_columns']:
                    st.markdown(f"• {col}")
            else:
                st.markdown("• None found")
            
            st.markdown("**📅 Date Columns (time-based):**")
            if validation_results['date_columns']:
                for col in validation_results['date_columns']:
                    st.markdown(f"• {col}")
            else:
                st.markdown("• None found")
        
        with col2:
            st.markdown("**🏷️ Category Columns (groups/labels):**")
            if validation_results['categorical_columns']:
                for col in validation_results['categorical_columns']:
                    st.markdown(f"• {col}")
            else:
                st.markdown("• None found")
            
            st.markdown("**📝 Text Columns (words/descriptions):**")
            if validation_results['text_columns']:
                for col in validation_results['text_columns']:
                    st.markdown(f"• {col}")
            else:
                st.markdown("• None found")
