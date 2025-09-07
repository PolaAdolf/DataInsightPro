import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linprog
import itertools
import os
from utils.gemini import generate_prescriptive_insights, get_gemini_api_key
import time
import re

# Gemini API key function is now imported from utils.gemini

# generate_prescriptive_insights function is now imported from utils.gemini

def perform_prescriptive_analytics(df, validation_results):
    """
    Perform prescriptive analytics on the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    validation_results : dict
        Results from data validation including column classifications
    """
    st.subheader("Prescriptive Analytics")
    
    with st.expander("Data Segmentation & Targeting", expanded=True):
        if len(validation_results['numeric_columns']) < 2:
            st.warning("At least 2 numeric columns are needed for clustering.")
        else:
            # Select features for clustering
            cluster_features = st.multiselect(
                "Select numeric features for segmentation:",
                validation_results['numeric_columns'],
                default=validation_results['numeric_columns'][:min(len(validation_results['numeric_columns']), 3)],
                key="cluster_features"
            )
            
            if cluster_features:
                # Number of clusters
                n_clusters = st.slider(
                    "Number of segments/clusters:",
                    min_value=2,
                    max_value=10,
                    value=3,
                    key="n_clusters"
                )
                
                # Perform clustering
                if st.button("Generate Segments", key="cluster_btn"):
                    with st.spinner("Segmenting data..."):
                        try:
                            # Prepare data
                            cluster_data = df[cluster_features].copy()
                            
                            # Handle missing values
                            cluster_data = cluster_data.fillna(cluster_data.mean())
                            
                            # Scale data
                            scaler = StandardScaler()
                            scaled_data = scaler.fit_transform(cluster_data)
                            
                            # Perform KMeans clustering
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
                            clusters = kmeans.fit_predict(scaled_data)
                            
                            # Add cluster labels to original data
                            df_with_clusters = df.copy()
                            df_with_clusters['Segment'] = clusters + 1  # 1-based indexing for user-friendliness
                            
                            # Display cluster distribution
                            cluster_counts = df_with_clusters['Segment'].value_counts().sort_index()
                            
                            st.write("**Segment Distribution:**")
                            
                            fig = px.pie(
                                values=cluster_counts.values,
                                names=cluster_counts.index.map(lambda x: f"Segment {x}"),
                                title="Data Segments Distribution",
                                color_discrete_sequence=px.colors.qualitative.Bold
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show segment profiles
                            st.write("**Segment Profiles:**")
                            
                            # Calculate segment profiles
                            segment_profiles = df_with_clusters.groupby('Segment')[cluster_features].mean()
                            
                            # Scale profiles for radar chart
                            scaled_profiles = segment_profiles.copy()
                            for col in scaled_profiles.columns:
                                min_val = df_with_clusters[col].min()
                                max_val = df_with_clusters[col].max()
                                scaled_profiles[col] = (scaled_profiles[col] - min_val) / (max_val - min_val) if max_val > min_val else 0
                            
                            # Create radar chart for each segment
                            for segment in sorted(df_with_clusters['Segment'].unique()):
                                st.write(f"**Segment {segment} Profile:**")
                                
                                # Radar chart for segment
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatterpolar(
                                    r=scaled_profiles.loc[segment].values,
                                    theta=scaled_profiles.columns,
                                    fill='toself',
                                    name=f'Segment {segment}'
                                ))
                                
                                fig.update_layout(
                                    polar=dict(
                                        radialaxis=dict(
                                            visible=True,
                                            range=[0, 1]
                                        )
                                    ),
                                    title=f"Segment {segment} Profile"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Segment characteristics table
                                segment_chars = segment_profiles.loc[segment].reset_index()
                                segment_chars.columns = ['Feature', 'Average Value']
                                segment_chars['Feature'] = segment_chars['Feature'].astype(str)
                                segment_chars['Average Value'] = segment_chars['Average Value'].round(2)
                                
                                # Get overall averages for comparison
                                overall_avgs = df[cluster_features].mean().reset_index()
                                overall_avgs.columns = ['Feature', 'Overall Average']
                                overall_avgs['Overall Average'] = overall_avgs['Overall Average'].round(2)
                                
                                # Merge segment and overall stats
                                comparison = segment_chars.merge(overall_avgs, on='Feature')
                                
                                # Calculate difference and % difference
                                comparison['Difference'] = (comparison['Average Value'] - comparison['Overall Average']).round(2)
                                comparison['% Difference'] = (comparison['Difference'] / comparison['Overall Average'] * 100).round(2)
                                
                                st.dataframe(comparison.sort_values('% Difference', ascending=False))
                            
                            # 2D visualization of clusters if more than 2 features
                            if len(cluster_features) > 2:
                                st.write("**2D Cluster Visualization (PCA):**")
                                
                                # Perform PCA for visualization
                                pca = PCA(n_components=2)
                                pca_result = pca.fit_transform(scaled_data)
                                
                                # Create visualization dataframe
                                pca_df = pd.DataFrame({
                                    'PC1': pca_result[:, 0],
                                    'PC2': pca_result[:, 1],
                                    'Segment': df_with_clusters['Segment'].astype(str)
                                })
                                
                                # Scatter plot of clusters
                                fig = px.scatter(
                                    pca_df,
                                    x='PC1',
                                    y='PC2',
                                    color='Segment',
                                    title="Cluster Visualization (PCA)",
                                    color_discrete_sequence=px.colors.qualitative.Bold
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Explained variance
                                explained_var = pca.explained_variance_ratio_ * 100
                                st.write(f"Explained variance: PC1 ({explained_var[0]:.2f}%), PC2 ({explained_var[1]:.2f}%)")
                            
                            # Segment-based recommendations
                            st.write("**Segment-Based Recommendations:**")
                            
                            # Create segment description for OpenAI
                            segment_description = "Dataset segmentation analysis results:\n\n"
                            
                            # Overall dataset description
                            segment_description += f"Overall dataset: {len(df)} records with features: {', '.join(cluster_features)}\n\n"
                            
                            # Add segment profiles
                            for segment in sorted(df_with_clusters['Segment'].unique()):
                                segment_size = len(df_with_clusters[df_with_clusters['Segment'] == segment])
                                segment_pct = segment_size / len(df_with_clusters) * 100
                                
                                segment_description += f"Segment {segment} ({segment_size} records, {segment_pct:.1f}% of total):\n"
                                
                                # Add key characteristics (top 3 differentiating features)
                                segment_data = df_with_clusters[df_with_clusters['Segment'] == segment]
                                
                                # Get comparison stats
                                for feature in cluster_features:
                                    segment_avg = segment_data[feature].mean()
                                    overall_avg = df[feature].mean()
                                    pct_diff = (segment_avg - overall_avg) / overall_avg * 100 if overall_avg != 0 else 0
                                    
                                    segment_description += f"- {feature}: {segment_avg:.2f} "
                                    segment_description += f"({'higher' if pct_diff > 0 else 'lower'} than overall by {abs(pct_diff):.1f}%)\n"
                                
                                segment_description += "\n"
                            
                            # Generate AI recommendations
                            question = "Based on the segment profiles described above, what specific actions should be taken for each segment? Provide segment-specific recommendations and strategies."
                            
                            with st.spinner("Generating AI recommendations..."):
                                ai_insights = generate_prescriptive_insights(segment_description, question)
                                st.markdown(ai_insights)
                            
                            # Download segment data
                            segment_data_csv = df_with_clusters.to_csv(index=False)
                            st.download_button(
                                label="Download Segmented Data",
                                data=segment_data_csv,
                                file_name="segmented_data.csv",
                                mime="text/csv",
                                key="download_segments"
                            )
                                
                        except Exception as e:
                            st.error(f"Error performing segmentation: {str(e)}")
            else:
                st.info("Please select at least one feature for segmentation.")
    
    with st.expander("Optimization Analysis", expanded=True):
        st.write("**Resource Allocation Optimizer**")
        
        # Check if we have numeric columns for optimization
        if not validation_results['numeric_columns']:
            st.warning("Optimization requires numeric columns to work with.")
        else:
            # Simple optimization scenario
            st.write("Define an optimization scenario by allocating resources to maximize an outcome:")
            
            # Select target (objective) to maximize
            objective_col = st.selectbox(
                "Select target variable to maximize:",
                validation_results['numeric_columns'],
                key="objective_col"
            )
            
            # Select constraint variables
            constraint_options = [col for col in validation_results['numeric_columns'] if col != objective_col]
            
            if not constraint_options:
                st.warning("Need at least one additional numeric column as constraint.")
            else:
                constraint_cols = st.multiselect(
                    "Select resource variables (constraints):",
                    constraint_options,
                    default=constraint_options[:min(len(constraint_options), 2)],
                    key="constraint_cols"
                )
                
                if constraint_cols:
                    # Set up simple linear optimization problem
                    st.write("**Configure Optimization Parameters:**")
                    
                    # Create input fields for constraints
                    constraints = {}
                    
                    col1, col2 = st.columns(2)
                    
                    for i, col in enumerate(constraint_cols):
                        with col1 if i % 2 == 0 else col2:
                            # Get column statistics
                            col_min = float(df[col].min())
                            col_max = float(df[col].max())
                            col_mean = float(df[col].mean())
                            
                            # Default to the mean value as constraint limit
                            constraints[col] = st.number_input(
                                f"Maximum {col}:",
                                min_value=col_min,
                                max_value=col_max * 2,  # Allow some flexibility
                                value=col_mean * 1.5,  # Slightly higher than mean
                                key=f"constraint_{col}"
                            )
                    
                    # Run optimization
                    if st.button("Run Optimization", key="optimize_btn"):
                        with st.spinner("Calculating optimal allocation..."):
                            try:
                                # Get clean data for optimization
                                opt_data = df[[objective_col] + constraint_cols].dropna()
                                
                                if len(opt_data) < 10:
                                    st.error("Not enough clean data for optimization. Need at least 10 rows.")
                                else:
                                    # Build a simple linear model to estimate coefficients
                                    X = opt_data[constraint_cols]
                                    y = opt_data[objective_col]
                                    
                                    # Add constant term
                                    X_with_const = pd.DataFrame({'const': np.ones(len(X))})
                                    X_with_const = pd.concat([X_with_const, X], axis=1)
                                    
                                    # Calculate coefficients using simple linear regression
                                    # (X'X)^-1 X'y
                                    try:
                                        coeffs = np.linalg.pinv(X_with_const.T.dot(X_with_const)).dot(X_with_const.T).dot(y)
                                        
                                        # Extract coefficients (skip the constant term)
                                        objective_coeffs = coeffs[1:]
                                        
                                        # Set up linear programming problem
                                        # Negate coefficients for maximization (linprog does minimization)
                                        c = -1 * objective_coeffs
                                        
                                        # Simple bounds for each variable
                                        bounds = [(0, None) for _ in range(len(constraint_cols))]
                                        
                                        # Constraint matrix (identity matrix)
                                        A = np.eye(len(constraint_cols))
                                        
                                        # Constraint limits
                                        b = [constraints[col] for col in constraint_cols]
                                        
                                        # Solve optimization problem
                                        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
                                        
                                        if result.success:
                                            st.success("Optimization completed successfully!")
                                            
                                            # Prepare results
                                            optimal_allocation = pd.DataFrame({
                                                'Resource': constraint_cols,
                                                'Optimal Allocation': result.x.round(2),
                                                'Constraint Limit': b,
                                                'Utilization (%)': (result.x / b * 100).round(2)
                                            })
                                            
                                            # Calculate estimated objective value
                                            estimated_objective = -result.fun
                                            
                                            # Display results
                                            st.write("**Optimal Resource Allocation:**")
                                            st.dataframe(optimal_allocation)
                                            
                                            st.write(f"**Estimated {objective_col}:** {estimated_objective:.2f}")
                                            
                                            # Visualize allocation
                                            fig = px.bar(
                                                optimal_allocation,
                                                x='Resource',
                                                y='Optimal Allocation',
                                                text='Optimal Allocation',
                                                title="Optimal Resource Allocation",
                                                color='Utilization (%)',
                                                color_continuous_scale='RdYlGn'
                                            )
                                            
                                            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Constraint utilization
                                            fig = px.bar(
                                                optimal_allocation,
                                                x='Resource',
                                                y='Utilization (%)',
                                                text='Utilization (%)',
                                                title="Resource Utilization (%)",
                                                color='Utilization (%)',
                                                color_continuous_scale='RdYlGn'
                                            )
                                            
                                            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                                            fig.update_layout(yaxis_range=[0, 100])
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Generate recommendations
                                            st.write("**Optimization Recommendations:**")
                                            
                                            # Create description for OpenAI
                                            optimization_description = f"Resource allocation optimization for maximizing {objective_col}:\n\n"
                                            
                                            # Add resources and allocations
                                            for i, row in optimal_allocation.iterrows():
                                                optimization_description += f"- {row['Resource']}: Optimal allocation of {row['Optimal Allocation']:.2f}"
                                                optimization_description += f" ({row['Utilization (%)']:.1f}% of maximum {row['Constraint Limit']:.2f})\n"
                                            
                                            # Add estimated outcome
                                            optimization_description += f"\nEstimated {objective_col} with this allocation: {estimated_objective:.2f}\n"
                                            
                                            # Add model information
                                            optimization_description += "\nCoefficients (impact of each resource unit on the outcome):\n"
                                            for col, coef in zip(constraint_cols, objective_coeffs):
                                                optimization_description += f"- {col}: {abs(coef):.4f} ({'positive' if coef > 0 else 'negative'} impact)\n"
                                            
                                            # Generate AI recommendations
                                            question = f"Based on the optimization results above, what specific actions should be taken to implement this resource allocation? Provide practical implementation steps and expected benefits."
                                            
                                            with st.spinner("Generating implementation recommendations..."):
                                                ai_insights = generate_prescriptive_insights(optimization_description, question)
                                                st.markdown(ai_insights)
                                        else:
                                            st.error(f"Optimization failed: {result.message}")
                                    except Exception as e:
                                        st.error(f"Error during optimization calculation: {str(e)}")
                            except Exception as e:
                                st.error(f"Error setting up optimization: {str(e)}")
                else:
                    st.info("Please select at least one resource variable.")
    
    with st.expander("What-If Analysis", expanded=True):
        st.write("**Scenario Planning & Impact Analysis**")
        
        # Select a target variable to analyze
        if validation_results['numeric_columns']:
            target_var = st.selectbox(
                "Select target variable to analyze:",
                validation_results['numeric_columns'],
                key="whatif_target"
            )
            
            # Select influencing variables
            influencers = [col for col in validation_results['numeric_columns'] if col != target_var]
            
            if not influencers:
                st.warning("Need at least one additional numeric column as influencer.")
            else:
                selected_influencers = st.multiselect(
                    "Select variables that influence the target:",
                    influencers,
                    default=influencers[:min(len(influencers), 2)],
                    key="whatif_influencers"
                )
                
                if selected_influencers:
                    st.write("**Configure Scenarios:**")
                    
                    # Create baseline and scenarios
                    baseline = {}
                    scenario_1 = {}
                    scenario_2 = {}
                    
                    for col in selected_influencers:
                        # Get column statistics
                        col_min = float(df[col].min())
                        col_max = float(df[col].max())
                        col_mean = float(df[col].mean())
                        col_std = float(df[col].std())
                        
                        st.write(f"**{col}** - Current Mean: {col_mean:.2f}, Min: {col_min:.2f}, Max: {col_max:.2f}")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            baseline[col] = st.number_input(
                                f"Baseline {col}:",
                                min_value=col_min - col_std,
                                max_value=col_max + col_std,
                                value=col_mean,
                                key=f"baseline_{col}"
                            )
                        
                        with col2:
                            scenario_1[col] = st.number_input(
                                f"Scenario 1 {col}:",
                                min_value=col_min - col_std,
                                max_value=col_max + col_std,
                                value=min(col_max, col_mean * 1.1),  # 10% increase as default
                                key=f"scenario1_{col}"
                            )
                        
                        with col3:
                            scenario_2[col] = st.number_input(
                                f"Scenario 2 {col}:",
                                min_value=col_min - col_std,
                                max_value=col_max + col_std,
                                value=max(col_min, col_mean * 0.9),  # 10% decrease as default
                                key=f"scenario2_{col}"
                            )
                    
                    # Run what-if analysis
                    if st.button("Analyze Scenarios", key="whatif_btn"):
                        with st.spinner("Calculating scenario impacts..."):
                            try:
                                # Get clean data for analysis
                                whatif_data = df[[target_var] + selected_influencers].dropna()
                                
                                if len(whatif_data) < 10:
                                    st.error("Not enough clean data for what-if analysis. Need at least 10 rows.")
                                else:
                                    # Build a simple linear model
                                    X = whatif_data[selected_influencers]
                                    y = whatif_data[target_var]
                                    
                                    # Add constant term
                                    X_with_const = pd.DataFrame({'const': np.ones(len(X))})
                                    X_with_const = pd.concat([X_with_const, X], axis=1)
                                    
                                    # Calculate coefficients
                                    try:
                                        coeffs = np.linalg.pinv(X_with_const.T.dot(X_with_const)).dot(X_with_const.T).dot(y)
                                        
                                        # Extract coefficients
                                        const_term = coeffs[0]
                                        feature_coeffs = coeffs[1:]
                                        
                                        # Create equation representation
                                        equation = f"{target_var} = {const_term:.4f}"
                                        for col, coef in zip(selected_influencers, feature_coeffs):
                                            sign = "+" if coef >= 0 else "-"
                                            equation += f" {sign} {abs(coef):.4f} × {col}"
                                        
                                        st.write("**Prediction Model:**")
                                        st.code(equation)
                                        
                                        # Create prediction function
                                        def predict(scenario):
                                            result = const_term
                                            for col, coef in zip(selected_influencers, feature_coeffs):
                                                result += coef * scenario[col]
                                            return result
                                        
                                        # Make predictions
                                        baseline_pred = predict(baseline)
                                        scenario1_pred = predict(scenario_1)
                                        scenario2_pred = predict(scenario_2)
                                        
                                        # Calculate impacts
                                        impact1 = scenario1_pred - baseline_pred
                                        impact1_pct = (impact1 / baseline_pred) * 100 if baseline_pred != 0 else 0
                                        
                                        impact2 = scenario2_pred - baseline_pred
                                        impact2_pct = (impact2 / baseline_pred) * 100 if baseline_pred != 0 else 0
                                        
                                        # Display predictions
                                        st.write("**Scenario Impact Analysis:**")
                                        
                                        impact_df = pd.DataFrame({
                                            'Scenario': ['Baseline', 'Scenario 1', 'Scenario 2'],
                                            f'Predicted {target_var}': [baseline_pred, scenario1_pred, scenario2_pred],
                                            'Change from Baseline': [0, impact1, impact2],
                                            'Change (%)': [0, impact1_pct, impact2_pct]
                                        })
                                        
                                        # Format numbers
                                        impact_df[f'Predicted {target_var}'] = impact_df[f'Predicted {target_var}'].round(2)
                                        impact_df['Change from Baseline'] = impact_df['Change from Baseline'].round(2)
                                        impact_df['Change (%)'] = impact_df['Change (%)'].round(2)
                                        
                                        st.dataframe(impact_df)
                                        
                                        # Visualize scenario comparison
                                        fig = go.Figure()
                                        
                                        fig.add_trace(go.Bar(
                                            x=impact_df['Scenario'],
                                            y=impact_df[f'Predicted {target_var}'],
                                            text=impact_df[f'Predicted {target_var}'],
                                            textposition='auto',
                                            name=f'Predicted {target_var}'
                                        ))
                                        
                                        fig.update_layout(
                                            title=f"Scenario Comparison for {target_var}",
                                            xaxis_title="Scenario",
                                            yaxis_title=target_var
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Feature impact analysis
                                        st.write("**Feature Impact Analysis:**")
                                        
                                        # Calculate individual feature impacts
                                        feature_impacts = []
                                        
                                        for col, coef in zip(selected_influencers, feature_coeffs):
                                            # Scenario 1 impact
                                            s1_change = scenario_1[col] - baseline[col]
                                            s1_impact = coef * s1_change
                                            s1_pct = (s1_impact / baseline_pred) * 100 if baseline_pred != 0 else 0
                                            
                                            # Scenario 2 impact
                                            s2_change = scenario_2[col] - baseline[col]
                                            s2_impact = coef * s2_change
                                            s2_pct = (s2_impact / baseline_pred) * 100 if baseline_pred != 0 else 0
                                            
                                            feature_impacts.append({
                                                'Feature': col,
                                                'Coefficient': coef,
                                                'Scenario 1 Change': s1_change,
                                                'Scenario 1 Impact': s1_impact,
                                                'Scenario 1 Impact (%)': s1_pct,
                                                'Scenario 2 Change': s2_change,
                                                'Scenario 2 Impact': s2_impact,
                                                'Scenario 2 Impact (%)': s2_pct
                                            })
                                        
                                        # Convert to DataFrame
                                        impact_table = pd.DataFrame(feature_impacts)
                                        
                                        # Round values
                                        for col in impact_table.columns:
                                            if col != 'Feature':
                                                impact_table[col] = impact_table[col].round(2)
                                        
                                        st.dataframe(impact_table)
                                        
                                        # Visualize relative impacts for Scenario 1
                                        fig = px.bar(
                                            impact_table,
                                            x='Feature',
                                            y='Scenario 1 Impact',
                                            title="Feature Impacts in Scenario 1",
                                            color='Scenario 1 Impact',
                                            color_continuous_scale='RdBu',
                                            color_continuous_midpoint=0
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Generate recommendations based on the scenarios
                                        st.write("**Scenario-Based Recommendations:**")
                                        
                                        # Create description for OpenAI
                                        scenario_description = f"What-If analysis for {target_var}:\n\n"
                                        
                                        # Add baseline and scenarios
                                        scenario_description += "Baseline scenario:\n"
                                        for col in selected_influencers:
                                            scenario_description += f"- {col}: {baseline[col]:.2f}\n"
                                        scenario_description += f"Predicted {target_var}: {baseline_pred:.2f}\n\n"
                                        
                                        scenario_description += "Scenario 1:\n"
                                        for col in selected_influencers:
                                            change = scenario_1[col] - baseline[col]
                                            change_pct = (change / baseline[col]) * 100 if baseline[col] != 0 else 0
                                            direction = "increase" if change > 0 else "decrease"
                                            scenario_description += f"- {col}: {scenario_1[col]:.2f} ({abs(change_pct):.1f}% {direction})\n"
                                        scenario_description += f"Predicted {target_var}: {scenario1_pred:.2f} ({impact1_pct:.1f}% change)\n\n"
                                        
                                        scenario_description += "Scenario 2:\n"
                                        for col in selected_influencers:
                                            change = scenario_2[col] - baseline[col]
                                            change_pct = (change / baseline[col]) * 100 if baseline[col] != 0 else 0
                                            direction = "increase" if change > 0 else "decrease"
                                            scenario_description += f"- {col}: {scenario_2[col]:.2f} ({abs(change_pct):.1f}% {direction})\n"
                                        scenario_description += f"Predicted {target_var}: {scenario2_pred:.2f} ({impact2_pct:.1f}% change)\n\n"
                                        
                                        # Add feature impacts
                                        scenario_description += "Feature impacts:\n"
                                        for impact in feature_impacts:
                                            scenario_description += f"- {impact['Feature']} (coefficient {impact['Coefficient']:.4f}):\n"
                                            scenario_description += f"  - In Scenario 1: Impact of {impact['Scenario 1 Impact']:.2f} ({impact['Scenario 1 Impact (%)']:.1f}% of baseline)\n"
                                            scenario_description += f"  - In Scenario 2: Impact of {impact['Scenario 2 Impact']:.2f} ({impact['Scenario 2 Impact (%)']:.1f}% of baseline)\n"
                                        
                                        # Generate AI recommendations
                                        question = f"Based on the what-if analysis results above, what specific actions should be taken? Evaluate both scenarios and recommend which one is better and why. Also suggest how to implement the recommended scenario."
                                        
                                        with st.spinner("Generating scenario recommendations..."):
                                            ai_insights = generate_prescriptive_insights(scenario_description, question)
                                            st.markdown(ai_insights)
                                    except Exception as e:
                                        st.error(f"Error in scenario calculation: {str(e)}")
                            except Exception as e:
                                st.error(f"Error setting up what-if analysis: {str(e)}")
                else:
                    st.info("Please select at least one influencer variable.")
        else:
            st.warning("No numeric columns available for what-if analysis.")
