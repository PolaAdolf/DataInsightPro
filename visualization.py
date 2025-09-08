import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_countplot(df, column, title=None, color=None, orientation='v'):
    """
    Create a count plot (bar chart of value counts) for a categorical column
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    column : str
        The categorical column to plot
    title : str, optional
        The title for the plot
    color : str, optional
        Column to use for coloring the bars
    orientation : str, optional
        'v' for vertical or 'h' for horizontal
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The plotly figure object
    """
    # Compute value counts
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = [column, 'count']
    
    # Sort by count in descending order
    value_counts = value_counts.sort_values('count', ascending=False)
    
    # Limit to top 20 categories if too many
    if len(value_counts) > 20:
        value_counts = value_counts.head(20)
        title_suffix = " (Top 20 Categories)"
    else:
        title_suffix = ""
    
    # Set default title
    if title is None:
        title = f"Count of {column}{title_suffix}"
    
    # Create figure
    if orientation == 'h':
        fig = px.bar(
            value_counts, 
            y=column, 
            x='count', 
            color=color if color else None,
            title=title,
            labels={'count': 'Count', column: column},
            orientation='h'
        )
    else:
        fig = px.bar(
            value_counts, 
            x=column, 
            y='count', 
            color=color if color else None,
            title=title,
            labels={'count': 'Count', column: column}
        )
    
    # Update layout
    fig.update_layout(
        xaxis_title=column if orientation == 'v' else 'Count',
        yaxis_title='Count' if orientation == 'v' else column
    )
    
    return fig

def create_histogram(df, column, bins=None, color=None, title=None):
    """
    Create a histogram for a numeric column
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    column : str
        The numeric column to plot
    bins : int, optional
        Number of bins for the histogram
    color : str, optional
        Column to use for coloring the bars
    title : str, optional
        The title for the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The plotly figure object
    """
    # Set default number of bins based on data size
    if bins is None:
        bins = min(30, max(10, int(np.sqrt(len(df)))))
    
    # Set default title
    if title is None:
        title = f"Distribution of {column}"
    
    # Create histogram
    fig = px.histogram(
        df, 
        x=column, 
        color=color if color else None,
        nbins=bins,
        title=title,
        marginal='box'  # Add box plot on top
    )
    
    # Add a vertical line for the mean
    fig.add_vline(
        x=df[column].mean(),
        line_dash="dash", 
        line_color="red",
        annotation_text="Mean",
        annotation_position="top"
    )
    
    # Add a vertical line for the median
    fig.add_vline(
        x=df[column].median(),
        line_dash="dash", 
        line_color="green",
        annotation_text="Median",
        annotation_position="bottom"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=column,
        yaxis_title='Count'
    )
    
    return fig

def create_boxplot(df, column, by=None, title=None):
    """
    Create a box plot for a numeric column, optionally grouped by a categorical column
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    column : str
        The numeric column to plot
    by : str, optional
        Categorical column to group by
    title : str, optional
        The title for the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The plotly figure object
    """
    # Set default title
    if title is None:
        if by:
            title = f"Distribution of {column} by {by}"
        else:
            title = f"Distribution of {column}"
    
    # Create box plot
    if by:
        fig = px.box(
            df, 
            x=by, 
            y=column,
            title=title,
            points="all"  # Show all points
        )
    else:
        fig = px.box(
            df, 
            y=column,
            title=title,
            points="all"  # Show all points
        )
    
    # Update layout
    if by:
        fig.update_layout(
            xaxis_title=by,
            yaxis_title=column
        )
    else:
        fig.update_layout(
            yaxis_title=column
        )
    
    return fig

def create_piechart(df, column, title=None, hole=0.4):
    """
    Create a pie chart for a categorical column
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    column : str
        The categorical column to plot
    title : str, optional
        The title for the plot
    hole : float, optional
        Size of the hole in the middle (0 for pie, > 0 for donut)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The plotly figure object
    """
    # Compute value counts
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = [column, 'count']
    
    # Limit to top 10 categories and group the rest as "Other"
    if len(value_counts) > 10:
        top_10 = value_counts.head(10)
        other_count = value_counts['count'][10:].sum()
        
        if other_count > 0:
            other_row = pd.DataFrame({column: ['Other'], 'count': [other_count]})
            value_counts = pd.concat([top_10, other_row], ignore_index=True)
            title_suffix = " (Top 10 + Other)"
        else:
            value_counts = top_10
            title_suffix = " (Top 10 Categories)"
    else:
        title_suffix = ""
    
    # Set default title
    if title is None:
        title = f"Distribution of {column}{title_suffix}"
    
    # Create pie chart
    fig = px.pie(
        value_counts, 
        names=column, 
        values='count',
        title=title,
        hole=hole,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    # Update layout
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig

def create_heatmap(corr_matrix, title=None):
    """
    Create a heatmap from a correlation matrix
    
    Parameters:
    -----------
    corr_matrix : pandas.DataFrame
        The correlation matrix
    title : str, optional
        The title for the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The plotly figure object
    """
    # Set default title
    if title is None:
        title = "Correlation Matrix"
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title=title
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        width=800
    )
    
    return fig

def create_scatter(df, x, y, color=None, size=None, add_trendline=False, title=None):
    """
    Create a scatter plot for two numeric columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    x : str
        The column to use for x-axis
    y : str
        The column to use for y-axis
    color : str, optional
        Column to use for coloring the points
    size : str, optional
        Column to use for sizing the points
    add_trendline : bool, optional
        Whether to add a trendline
    title : str, optional
        The title for the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The plotly figure object
    """
    # Set default title
    if title is None:
        title = f"{y} vs {x}"
    
    # Create scatter plot
    fig = px.scatter(
        df, 
        x=x, 
        y=y,
        color=color if color else None,
        size=size if size else None,
        title=title,
        opacity=0.7
    )
    
    # Add trendline if requested
    if add_trendline:
        fig.update_layout(
            shapes=[
                dict(
                    type='line',
                    xref='x', yref='y',
                    x0=df[x].min(), y0=df[y].min(),
                    x1=df[x].max(), y1=df[y].max(),
                    line=dict(color='red', dash='dash')
                )
            ]
        )
        
        # Add trendline with Plotly Express
        fig = px.scatter(
            df, 
            x=x, 
            y=y,
            color=color if color else None,
            size=size if size else None,
            trendline="ols",
            title=title,
            opacity=0.7
        )
    
    # Update layout
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y
    )
    
    return fig

def create_correlation_chart(df, target, features, title=None):
    """
    Create a bar chart showing correlation between features and a target variable
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    target : str
        The target variable
    features : list
        List of feature columns to check correlation with
    title : str, optional
        The title for the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The plotly figure object
    """
    # Calculate correlations
    correlations = []
    
    for feature in features:
        if feature != target:
            correlation = df[[feature, target]].corr().iloc[0, 1]
            correlations.append({'Feature': feature, 'Correlation': correlation})
    
    # Convert to DataFrame and sort
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('Correlation', ascending=False)
    
    # Set default title
    if title is None:
        title = f"Correlation with {target}"
    
    # Create bar chart
    fig = px.bar(
        corr_df,
        x='Feature',
        y='Correlation',
        title=title,
        color='Correlation',
        color_continuous_scale='RdBu_r',
        text='Correlation'
    )
    
    # Update layout
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        xaxis_title='Feature',
        yaxis_title='Correlation Coefficient',
        yaxis=dict(range=[-1, 1])
    )
    
    # Add horizontal line at y=0
    fig.add_shape(
        type='line',
        x0=-0.5,
        y0=0,
        x1=len(features) - 0.5,
        y1=0,
        line=dict(color='black', dash='dash')
    )
    
    return fig

def create_box_comparison(df, group_col, metric_col, title=None):
    """
    Create a box plot comparison of a metric across groups
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    group_col : str
        The categorical column defining groups
    metric_col : str
        The numeric column to compare across groups
    title : str, optional
        The title for the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The plotly figure object
    """
    # Set default title
    if title is None:
        title = f"Comparison of {metric_col} across {group_col} groups"
    
    # Create violin plot with box plot inside
    fig = px.violin(
        df,
        x=group_col,
        y=metric_col,
        title=title,
        color=group_col,
        box=True,
        points="all"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=group_col,
        yaxis_title=metric_col,
        showlegend=False
    )
    
    return fig

def create_line_chart(df, x, y, color=None, title=None):
    """
    Create a line chart for trend visualization
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    x : str
        The column to use for x-axis (typically time)
    y : str
        The column to use for y-axis
    color : str, optional
        Column to use for creating multiple lines
    title : str, optional
        The title for the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The plotly figure object
    """
    # Set default title
    if title is None:
        title = f"Trend of {y} over {x}"
    
    # Create line chart
    fig = px.line(
        df,
        x=x,
        y=y,
        color=color if color else None,
        title=title,
        markers=True
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y
    )
    
    return fig

def create_radar_chart(df, categories, values, title=None):
    """
    Create a radar chart (spider/web chart)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing one row of data
    categories : list
        List of category names (column names in df)
    values : list
        List of values for each category
    title : str, optional
        The title for the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The plotly figure object
    """
    # Set default title
    if title is None:
        title = "Radar Chart"
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=title
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.1]
            )
        ),
        title=title
    )
    
    return fig

def create_funnel_chart(values, categories, title=None):
    """
    Create a funnel chart
    
    Parameters:
    -----------
    values : list
        List of numeric values
    categories : list
        List of category labels
    title : str, optional
        The title for the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The plotly figure object
    """
    # Set default title
    if title is None:
        title = "Funnel Chart"
    
    # Create funnel chart
    fig = go.Figure(go.Funnel(
        y=categories,
        x=values,
        textinfo="value+percent initial"
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        funnelmode="stack"
    )
    
    return fig

def create_bubble_chart(df, x, y, size, color=None, hover_name=None, title=None):
    """
    Create a bubble chart
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    x : str
        The column for x-axis
    y : str
        The column for y-axis
    size : str
        The column for bubble size
    color : str, optional
        The column for bubble color
    hover_name : str, optional
        The column to use for hover labels
    title : str, optional
        The title for the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The plotly figure object
    """
    # Set default title
    if title is None:
        title = f"{y} vs {x} (sized by {size})"
    
    # Create bubble chart
    fig = px.scatter(
        df,
        x=x,
        y=y,
        size=size,
        color=color if color else None,
        hover_name=hover_name if hover_name else None,
        title=title,
        size_max=60,
        opacity=0.7
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y
    )
    
    return fig

def create_parallel_coordinates(df, dimensions, color, title=None):
    """
    Create a parallel coordinates plot
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    dimensions : list
        List of columns to use as dimensions
    color : str
        Column to use for coloring the lines
    title : str, optional
        The title for the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The plotly figure object
    """
    # Set default title
    if title is None:
        title = "Parallel Coordinates Plot"
    
    # Create parallel coordinates plot
    fig = px.parallel_coordinates(
        df,
        dimensions=dimensions,
        color=color,
        title=title,
        color_continuous_scale=px.colors.diverging.Tealrose
    )
    
    # Update layout
    fig.update_layout(
        coloraxis_colorbar=dict(title=color)
    )
    
    return fig

def create_density_heatmap(df, x, y, title=None):
    """
    Create a 2D density heatmap
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    x : str
        The column for x-axis
    y : str
        The column for y-axis
    title : str, optional
        The title for the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The plotly figure object
    """
    # Set default title
    if title is None:
        title = f"Density of {y} vs {x}"
    
    # Create density heatmap
    fig = px.density_heatmap(
        df,
        x=x,
        y=y,
        title=title,
        marginal_x="histogram",
        marginal_y="histogram"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y
    )
    
    return fig
