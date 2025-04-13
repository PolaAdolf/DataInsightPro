import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.visualization import create_scatter, create_heatmap, create_correlation_chart, create_box_comparison
from scipy import stats

def perform_diagnostic_analytics(df, validation_results):
    """
    Perform diagnostic analytics on the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    validation_results : dict
        Results from data validation including column classifications
    """
    st.subheader("Correlation Analysis")
    
    # Correlation analysis for numeric variables
    if len(validation_results['numeric_columns']) > 1:
        with st.expander("Correlation Between Variables", expanded=True):
            # Select variables for correlation
            x_var = st.selectbox(
                "Select first variable (X):",
                validation_results['numeric_columns'],
                key="diag_x_var"
            )
            
            y_var = st.selectbox(
                "Select second variable (Y):",
                [col for col in validation_results['numeric_columns'] if col != x_var],
                key="diag_y_var"
            )
            
            # Calculate correlation
            correlation = df[x_var].corr(df[y_var])
            
            # Scatter plot with trendline
            st.write(f"**Correlation between {x_var} and {y_var}:** {correlation:.4f}")
            
            if abs(correlation) < 0.3:
                strength = "weak"
            elif abs(correlation) < 0.7:
                strength = "moderate"
            else:
                strength = "strong"
                
            direction = "positive" if correlation > 0 else "negative"
            
            st.write(f"This indicates a {strength} {direction} relationship.")
            
            # Scatter plot
            fig = create_scatter(df, x_var, y_var, add_trendline=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical test
            try:
                # Calculate statistical significance
                slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_var].dropna(), df[y_var].dropna())
                
                st.write("**Statistical Test Results:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("R-squared", f"{r_value**2:.4f}")
                
                with col2:
                    st.metric("p-value", f"{p_value:.4f}")
                    sig_status = "Significant" if p_value < 0.05 else "Not Significant"
                
                with col3:
                    st.metric("Significance", sig_status, delta=None)
                
                st.markdown(f"""
                - **Interpretation**: 
                  - The R-squared value of {r_value**2:.4f} means that {r_value**2*100:.1f}% of the variation in {y_var} 
                    can be explained by {x_var}.
                  - The relationship is statistically {"significant" if p_value < 0.05 else "not significant"} (p-value: {p_value:.4f}).
                  - For each unit increase in {x_var}, {y_var} changes by {slope:.4f} units on average.
                """)
            except:
                st.warning("Could not perform statistical test due to insufficient or invalid data.")
    else:
        st.info("At least 2 numeric columns are needed for correlation analysis.")
    
    # Outlier detection and analysis
    st.subheader("Outlier Analysis")
    
    with st.expander("Identify and Analyze Outliers", expanded=True):
        if validation_results['numeric_columns']:
            # Let user select a column for outlier analysis
            outlier_col = st.selectbox(
                "Select a column to analyze for outliers:",
                validation_results['numeric_columns'],
                key="outlier_column"
            )
            
            # Calculate outlier boundaries using IQR method
            Q1 = df[outlier_col].quantile(0.25)
            Q3 = df[outlier_col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)]
            outlier_percentage = (len(outliers) / len(df)) * 100
            
            # Show statistics
            st.write(f"**Outlier Analysis for {outlier_col}:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Outliers", f"{len(outliers)}")
            
            with col2:
                st.metric("Outlier Percentage", f"{outlier_percentage:.2f}%")
            
            with col3:
                st.metric("Normal Range", f"{lower_bound:.2f} to {upper_bound:.2f}")
            
            # Visualize outliers with box plot
            fig = px.box(
                df, 
                y=outlier_col,
                title=f"Box Plot with Outliers for {outlier_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show outlier data if exists
            if len(outliers) > 0:
                with st.expander("View Outlier Data Points"):
                    st.dataframe(outliers)
                    
                # Analyze impact of outliers
                st.write("**Impact of Outliers:**")
                
                # Statistics with and without outliers
                normal_data = df[(df[outlier_col] >= lower_bound) & (df[outlier_col] <= upper_bound)]
                
                stats_with_outliers = {
                    'Mean': df[outlier_col].mean(),
                    'Median': df[outlier_col].median(),
                    'Std Dev': df[outlier_col].std()
                }
                
                stats_without_outliers = {
                    'Mean': normal_data[outlier_col].mean(),
                    'Median': normal_data[outlier_col].median(),
                    'Std Dev': normal_data[outlier_col].std()
                }
                
                # Create comparison table
                comparison_df = pd.DataFrame({
                    'With Outliers': stats_with_outliers,
                    'Without Outliers': stats_without_outliers,
                    'Percentage Change': {
                        k: (stats_without_outliers[k] - v) / v * 100 
                        for k, v in stats_with_outliers.items()
                    }
                })
                
                st.table(comparison_df)
                
                # Visualize distribution with and without outliers
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(
                    x=df[outlier_col],
                    name='With Outliers',
                    opacity=0.7,
                    histnorm='probability density'
                ))
                
                fig.add_trace(go.Histogram(
                    x=normal_data[outlier_col],
                    name='Without Outliers',
                    opacity=0.7,
                    histnorm='probability density'
                ))
                
                fig.update_layout(
                    title=f"Distribution of {outlier_col} With and Without Outliers",
                    barmode='overlay',
                    xaxis_title=outlier_col,
                    yaxis_title='Probability Density'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success(f"No outliers detected in {outlier_col} based on the IQR method.")
        else:
            st.info("No numeric columns available for outlier analysis.")
    
    # Group comparison
    st.subheader("Group Comparison Analysis")
    
    with st.expander("Compare Data Across Groups", expanded=True):
        if validation_results['categorical_columns'] and validation_results['numeric_columns']:
            # Select variables for comparison
            group_var = st.selectbox(
                "Select categorical variable for grouping:",
                validation_results['categorical_columns'],
                key="group_var"
            )
            
            # Limit to top categories if too many
            value_counts = df[group_var].value_counts()
            if len(value_counts) > 10:
                top_categories = value_counts.nlargest(10).index.tolist()
                st.info(f"Showing only top 10 categories out of {len(value_counts)} due to large number of categories.")
                comparison_df = df[df[group_var].isin(top_categories)]
            else:
                comparison_df = df
            
            # Select metric to compare
            metric_var = st.selectbox(
                "Select numeric variable to compare across groups:",
                validation_results['numeric_columns'],
                key="metric_var"
            )
            
            # Create comparison visualization
            fig = create_box_comparison(comparison_df, group_var, metric_var)
            st.plotly_chart(fig, use_container_width=True)
            
            # ANOVA Test if more than 2 groups
            unique_groups = comparison_df[group_var].nunique()
            
            if unique_groups >= 2:
                st.write("**Statistical Comparison:**")
                
                try:
                    # Prepare data for ANOVA
                    groups = []
                    labels = []
                    
                    for name, group in comparison_df.groupby(group_var):
                        if len(group) > 0:  # Only include groups with data
                            groups.append(group[metric_var].dropna())
                            labels.append(str(name))
                    
                    if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                        # Perform ANOVA
                        f_statistic, p_value = stats.f_oneway(*groups)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("F-statistic", f"{f_statistic:.4f}")
                        
                        with col2:
                            st.metric("p-value", f"{p_value:.4f}")
                        
                        # Interpret results
                        if p_value < 0.05:
                            st.success(f"There is a statistically significant difference in {metric_var} between the different {group_var} groups (p-value: {p_value:.4f}).")
                        else:
                            st.info(f"There is no statistically significant difference in {metric_var} between the different {group_var} groups (p-value: {p_value:.4f}).")
                        
                        # Group statistics
                        group_stats = comparison_df.groupby(group_var)[metric_var].agg(['mean', 'median', 'std', 'count'])
                        group_stats = group_stats.sort_values('mean', ascending=False)
                        group_stats = group_stats.reset_index()
                        
                        # Format the output
                        group_stats['mean'] = group_stats['mean'].round(2)
                        group_stats['median'] = group_stats['median'].round(2)
                        group_stats['std'] = group_stats['std'].round(2)
                        
                        st.write("**Group Statistics:**")
                        st.dataframe(group_stats)
                    else:
                        st.warning("Some groups don't have enough data for statistical comparison.")
                except Exception as e:
                    st.error(f"Error performing statistical test: {str(e)}")
            else:
                st.info("Need at least 2 groups for statistical comparison.")
        elif not validation_results['categorical_columns']:
            st.info("No categorical columns available for grouping.")
        else:
            st.info("No numeric columns available for comparison.")
    
    # Trend analysis
    st.subheader("Trend Analysis")
    
    with st.expander("Identify Trends and Patterns", expanded=True):
        if validation_results['date_columns'] and validation_results['numeric_columns']:
            # Select date column for trend analysis
            date_col = st.selectbox(
                "Select date column:",
                validation_results['date_columns'],
                key="trend_date_col"
            )
            
            # Convert to datetime
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                
                # Select metric for trend
                trend_metric = st.selectbox(
                    "Select metric to analyze trend:",
                    validation_results['numeric_columns'],
                    key="trend_metric"
                )
                
                # Group by time periods
                time_period = st.selectbox(
                    "Select time period for aggregation:",
                    ['Day', 'Week', 'Month', 'Quarter', 'Year'],
                    key="time_period"
                )
                
                # Map selected period to pandas resample rule
                resample_map = {
                    'Day': 'D',
                    'Week': 'W',
                    'Month': 'M',
                    'Quarter': 'Q',
                    'Year': 'Y'
                }
                
                # Aggregate data by time period
                try:
                    df_trend = df.set_index(date_col)
                    trend_data = df_trend[trend_metric].resample(resample_map[time_period]).mean()
                    trend_data = trend_data.reset_index()
                    
                    # Create trend visualization
                    fig = px.line(
                        trend_data,
                        x=date_col,
                        y=trend_metric,
                        title=f"Trend of {trend_metric} by {time_period}",
                        markers=True
                    )
                    
                    # Add trendline
                    fig.update_layout(
                        xaxis_title=date_col,
                        yaxis_title=trend_metric
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate trend statistics
                    if len(trend_data) > 1:
                        # Calculate period-over-period change
                        trend_data['previous'] = trend_data[trend_metric].shift(1)
                        trend_data['change'] = trend_data[trend_metric] - trend_data['previous']
                        trend_data['percent_change'] = (trend_data['change'] / trend_data['previous']) * 100
                        
                        # Calculate overall trend
                        first_value = trend_data[trend_metric].iloc[0]
                        last_value = trend_data[trend_metric].iloc[-1]
                        total_change = last_value - first_value
                        total_percent_change = (total_change / first_value) * 100 if first_value != 0 else 0
                        
                        # Display trend statistics
                        st.write("**Trend Statistics:**")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Total Change", 
                                f"{total_change:.2f}",
                                delta=f"{total_percent_change:.2f}%" if total_percent_change else None
                            )
                        
                        with col2:
                            avg_change = trend_data['change'].mean()
                            st.metric("Average Change per Period", f"{avg_change:.2f}")
                        
                        with col3:
                            # Linear regression for trend slope
                            x = np.arange(len(trend_data))
                            y = trend_data[trend_metric].values
                            
                            try:
                                slope, _, _, p_value, _ = stats.linregress(x, y)
                                trend_direction = "Increasing" if slope > 0 else "Decreasing" if slope < 0 else "Stable"
                                st.metric("Trend Direction", trend_direction)
                            except:
                                st.metric("Trend Direction", "Insufficient Data")
                        
                        # Show recent trend data
                        st.write(f"**Recent {time_period}-by-{time_period} Changes:**")
                        recent_data = trend_data.dropna().tail(5)[
                            [date_col, trend_metric, 'change', 'percent_change']
                        ]
                        recent_data['percent_change'] = recent_data['percent_change'].round(2)
                        recent_data.columns = [date_col, trend_metric, 'Change', 'Percent Change (%)']
                        st.dataframe(recent_data)
                        
                        # Seasonality analysis if enough data
                        if len(trend_data) >= 12 and time_period in ['Day', 'Week', 'Month']:
                            st.write("**Seasonality Analysis:**")
                            
                            # Extract seasonal components by time period
                            if time_period == 'Month':
                                trend_data['period'] = trend_data[date_col].dt.month
                                period_name = 'Month'
                            elif time_period == 'Week':
                                trend_data['period'] = trend_data[date_col].dt.isocalendar().week
                                period_name = 'Week of Year'
                            else:  # Day
                                trend_data['period'] = trend_data[date_col].dt.dayofweek
                                period_name = 'Day of Week'
                                # Map numbers to day names
                                if period_name == 'Day of Week':
                                    day_map = {
                                        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
                                        3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
                                    }
                                    trend_data['period_name'] = trend_data['period'].map(day_map)
                                
                            # Group by period
                            seasonal_data = trend_data.groupby('period')[trend_metric].mean().reset_index()
                            
                            # Sort properly
                            seasonal_data = seasonal_data.sort_values('period')
                            
                            # Add period names if available
                            if 'period_name' in trend_data.columns:
                                period_mapping = trend_data[['period', 'period_name']].drop_duplicates()
                                seasonal_data = seasonal_data.merge(period_mapping, on='period')
                                x_col = 'period_name'
                            else:
                                x_col = 'period'
                            
                            # Create seasonality visualization
                            fig = px.bar(
                                seasonal_data,
                                x=x_col,
                                y=trend_metric,
                                title=f"Average {trend_metric} by {period_name}",
                                color=trend_metric,
                                color_continuous_scale='Viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough data points for trend analysis.")
                        
                except Exception as e:
                    st.error(f"Error in trend analysis: {str(e)}")
            except Exception as e:
                st.error(f"Error processing date column: {str(e)}")
        elif not validation_results['date_columns']:
            st.info("No date columns available for trend analysis.")
        else:
            st.info("No numeric columns available for trend analysis.")
