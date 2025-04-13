import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def perform_predictive_analytics(df, validation_results):
    """
    Perform predictive analytics on the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    validation_results : dict
        Results from data validation including column classifications
    """
    st.subheader("Predictive Modeling")
    
    # Check if there's enough data for prediction
    if len(df) < 10:
        st.error("Not enough data for predictive modeling. At least 10 rows required.")
        return
    
    # Make a copy to avoid modifying original data
    pred_df = df.copy()
    
    # Option for time series vs. regular prediction
    pred_type = st.radio(
        "Select prediction type:",
        ["Regular Prediction", "Time Series Forecasting"],
        key="pred_type"
    )
    
    if pred_type == "Regular Prediction":
        if len(validation_results['numeric_columns']) < 2 and len(validation_results['categorical_columns']) == 0:
            st.error("Not enough feature columns for prediction. Need at least one feature and one target.")
            return
            
        with st.expander("Build Prediction Model", expanded=True):
            # Step 1: Select target variable
            available_targets = validation_results['numeric_columns'] + validation_results['categorical_columns']
            
            target_var = st.selectbox(
                "Select target variable to predict:",
                available_targets,
                key="target_var"
            )
            
            # Determine if classification or regression
            is_classification = target_var in validation_results['categorical_columns']
            
            # Step 2: Select features
            potential_features = [col for col in pred_df.columns if col != target_var]
            
            selected_features = st.multiselect(
                "Select features for prediction (leave empty to use all):",
                potential_features,
                default=[],
                key="selected_features"
            )
            
            if not selected_features:  # If no features selected, use all except target
                selected_features = potential_features
                st.info(f"Using all {len(selected_features)} available features for prediction.")
            
            # Step 3: Display data shapes
            st.write(f"**Dataset Information:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"Total samples: {len(pred_df)}")
            
            with col2:
                st.write(f"Features: {len(selected_features)}")
            
            with col3:
                st.write(f"Target: {target_var}")
            
            # Step 4: Set up model parameters
            st.write("**Model Configuration:**")
            
            test_size = st.slider(
                "Test set size (%):",
                min_value=10,
                max_value=50,
                value=20,
                step=5,
                key="test_size"
            ) / 100
            
            model_type = st.selectbox(
                "Select model type:",
                ["Simple", "Advanced"],
                key="model_type"
            )
            
            # Step 5: Train the model
            if st.button("Train Model", key="train_btn"):
                try:
                    with st.spinner("Training model..."):
                        # Prepare X and y
                        X = pred_df[selected_features]
                        y = pred_df[target_var]
                        
                        # Identify numeric and categorical features
                        numeric_features = [col for col in selected_features if col in validation_results['numeric_columns']]
                        categorical_features = [col for col in selected_features if col in validation_results['categorical_columns']]
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )
                        
                        # Set up preprocessing pipeline
                        preprocessing = []
                        
                        if numeric_features:
                            numeric_transformer = Pipeline(steps=[
                                ('imputer', SimpleImputer(strategy='mean')),
                                ('scaler', StandardScaler())
                            ])
                            preprocessing.append(('num', numeric_transformer, numeric_features))
                        
                        if categorical_features:
                            categorical_transformer = Pipeline(steps=[
                                ('imputer', SimpleImputer(strategy='most_frequent')),
                                ('onehot', OneHotEncoder(handle_unknown='ignore'))
                            ])
                            preprocessing.append(('cat', categorical_transformer, categorical_features))
                        
                        preprocessor = ColumnTransformer(transformers=preprocessing)
                        
                        # Create and train model pipeline
                        if is_classification:
                            if model_type == "Simple":
                                model = LogisticRegression(max_iter=1000, random_state=42)
                                model_name = "Logistic Regression"
                            else:
                                model = RandomForestClassifier(n_estimators=100, random_state=42)
                                model_name = "Random Forest Classifier"
                        else:
                            if model_type == "Simple":
                                model = LinearRegression()
                                model_name = "Linear Regression"
                            else:
                                model = RandomForestRegressor(n_estimators=100, random_state=42)
                                model_name = "Random Forest Regressor"
                        
                        # Create pipeline
                        pipeline = Pipeline(steps=[
                            ('preprocessor', preprocessor),
                            ('model', model)
                        ])
                        
                        # Train model
                        pipeline.fit(X_train, y_train)
                        
                        # Make predictions
                        y_pred = pipeline.predict(X_test)
                        
                        # Evaluate model
                        st.write(f"**Model Performance ({model_name}):**")
                        
                        if is_classification:
                            # Classification metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            
                            st.write(f"Accuracy: {accuracy:.4f}")
                            
                            # Classification report
                            report = classification_report(y_test, y_pred, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df)
                            
                            # Confusion matrix
                            cm = confusion_matrix(y_test, y_pred)
                            cm_df = pd.DataFrame(
                                cm, 
                                index=[f'Actual {i}' for i in sorted(y.unique())], 
                                columns=[f'Predicted {i}' for i in sorted(y.unique())]
                            )
                            
                            st.write("**Confusion Matrix:**")
                            st.dataframe(cm_df)
                            
                            # Feature importance
                            if model_type == "Advanced":
                                importances = pipeline.named_steps['model'].feature_importances_
                                
                                # Get feature names after preprocessing
                                if hasattr(pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
                                    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
                                else:
                                    feature_names = [f"Feature {i}" for i in range(len(importances))]
                                
                                # Create feature importance df
                                importance_df = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Importance': importances
                                }).sort_values('Importance', ascending=False)
                                
                                st.write("**Feature Importance:**")
                                
                                # Visualize feature importance
                                fig = px.bar(
                                    importance_df.head(10),
                                    x='Importance',
                                    y='Feature',
                                    orientation='h',
                                    title="Top 10 Feature Importance",
                                    color='Importance',
                                    color_continuous_scale='Viridis'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Regression metrics
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            r2 = r2_score(y_test, y_pred)
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("R² Score", f"{r2:.4f}")
                            
                            with col2:
                                st.metric("RMSE", f"{rmse:.4f}")
                            
                            with col3:
                                st.metric("MSE", f"{mse:.4f}")
                            
                            # Plot actual vs predicted
                            pred_df = pd.DataFrame({
                                'Actual': y_test,
                                'Predicted': y_pred
                            })
                            
                            fig = px.scatter(
                                pred_df,
                                x='Actual',
                                y='Predicted',
                                title=f"Actual vs Predicted {target_var}",
                                opacity=0.7
                            )
                            
                            # Add 45-degree line
                            min_val = min(pred_df['Actual'].min(), pred_df['Predicted'].min())
                            max_val = max(pred_df['Actual'].max(), pred_df['Predicted'].max())
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=[min_val, max_val],
                                    y=[min_val, max_val],
                                    mode='lines',
                                    name='Perfect Prediction',
                                    line=dict(color='red', dash='dash')
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Feature importance for Random Forest
                            if model_type == "Advanced":
                                try:
                                    importances = pipeline.named_steps['model'].feature_importances_
                                    
                                    # Get feature names after preprocessing
                                    if hasattr(pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
                                        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
                                    else:
                                        feature_names = [f"Feature {i}" for i in range(len(importances))]
                                    
                                    # Create feature importance df
                                    importance_df = pd.DataFrame({
                                        'Feature': feature_names,
                                        'Importance': importances
                                    }).sort_values('Importance', ascending=False)
                                    
                                    st.write("**Feature Importance:**")
                                    
                                    # Visualize feature importance
                                    fig = px.bar(
                                        importance_df.head(10),
                                        x='Importance',
                                        y='Feature',
                                        orientation='h',
                                        title="Top 10 Feature Importance",
                                        color='Importance',
                                        color_continuous_scale='Viridis'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Could not compute feature importance: {str(e)}")
                        
                        # Model insights
                        st.write("**Model Insights:**")
                        
                        if is_classification:
                            st.markdown(f"""
                            - This classification model can predict **{target_var}** with an accuracy of **{accuracy:.2f}**.
                            - {len(X_train)} samples were used for training and {len(X_test)} samples for testing.
                            - The model performs best for the following classes: {", ".join(report_df.index[report_df['f1-score'] > report_df['f1-score'].mean()].tolist())}
                            """)
                        else:
                            st.markdown(f"""
                            - This regression model can explain **{r2*100:.2f}%** of the variance in **{target_var}**.
                            - The model has a Root Mean Square Error (RMSE) of **{rmse:.4f}**.
                            - {len(X_train)} samples were used for training and {len(X_test)} samples for testing.
                            """)
                            
                        # What-if analysis
                        st.write("**What-If Analysis:**")
                        st.write("Change feature values below to see predicted outcomes:")
                        
                        # Create feature sliders for numeric, dropdowns for categorical
                        what_if_values = {}
                        
                        # Use columns for better layout
                        cols = st.columns(2)
                        col_idx = 0
                        
                        for feature in selected_features:
                            with cols[col_idx % 2]:
                                if feature in validation_results['numeric_columns']:
                                    # Numeric slider
                                    min_val = float(pred_df[feature].min())
                                    max_val = float(pred_df[feature].max())
                                    step = (max_val - min_val) / 100
                                    
                                    default_val = float(pred_df[feature].mean())
                                    
                                    what_if_values[feature] = st.slider(
                                        f"{feature}:",
                                        min_value=min_val,
                                        max_value=max_val,
                                        value=default_val,
                                        step=step,
                                        key=f"whatif_{feature}"
                                    )
                                elif feature in validation_results['categorical_columns']:
                                    # Categorical dropdown
                                    options = pred_df[feature].dropna().unique()
                                    default_val = pred_df[feature].mode()[0]
                                    
                                    what_if_values[feature] = st.selectbox(
                                        f"{feature}:",
                                        options=options,
                                        index=np.where(options == default_val)[0][0] if default_val in options else 0,
                                        key=f"whatif_{feature}"
                                    )
                            col_idx += 1
                        
                        # Create prediction input
                        what_if_input = pd.DataFrame([what_if_values])
                        
                        # Make prediction
                        prediction = pipeline.predict(what_if_input)[0]
                        
                        # Display prediction
                        st.write("**Prediction Result:**")
                        st.info(f"The predicted {target_var} value is: **{prediction}**")
                
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")
    else:  # Time Series Forecasting
        if not validation_results['date_columns']:
            st.error("No date columns found. Time series forecasting requires a date column.")
            return
        
        if not validation_results['numeric_columns']:
            st.error("No numeric columns found. Time series forecasting requires a numeric target.")
            return
        
        with st.expander("Time Series Forecasting", expanded=True):
            # Step 1: Select date column
            date_col = st.selectbox(
                "Select date column:",
                validation_results['date_columns'],
                key="ts_date_col"
            )
            
            # Step 2: Select target column
            target_col = st.selectbox(
                "Select target column to forecast:",
                validation_results['numeric_columns'],
                key="ts_target_col"
            )
            
            # Step 3: Set forecast horizon
            forecast_periods = st.slider(
                "Number of periods to forecast:",
                min_value=1,
                max_value=30,
                value=5,
                key="forecast_periods"
            )
            
            # Step 4: Select model
            model_type = st.selectbox(
                "Select forecasting model:",
                ["ARIMA", "XGBoost"],
                key="ts_model_type"
            )
            
            # Step 5: Generate forecast
            if st.button("Generate Forecast", key="forecast_btn"):
                try:
                    with st.spinner("Generating forecast..."):
                        # Prepare time series data
                        ts_df = pred_df[[date_col, target_col]].copy()
                        
                        # Ensure date column is datetime
                        ts_df[date_col] = pd.to_datetime(ts_df[date_col])
                        
                        # Sort by date
                        ts_df = ts_df.sort_values(date_col)
                        
                        # Set date as index
                        ts_df = ts_df.set_index(date_col)
                        
                        # Drop missing values
                        ts_df = ts_df.dropna()
                        
                        # Resample to ensure regular time series
                        # Detect frequency
                        if len(ts_df) > 1:
                            # Try to infer frequency
                            inferred_freq = pd.infer_freq(ts_df.index)
                            
                            if inferred_freq is None:
                                # Try common frequencies
                                for freq in ['D', 'W', 'M', 'Q', 'Y']:
                                    try:
                                        ts_df = ts_df.resample(freq).mean()
                                        break
                                    except:
                                        continue
                                
                                # If still no valid frequency, use daily
                                if inferred_freq is None:
                                    st.warning("Could not detect time series frequency. Using daily frequency.")
                                    ts_df = ts_df.resample('D').mean()
                            else:
                                ts_df = ts_df.resample(inferred_freq).mean()
                        
                        # Fill missing values in resampled data
                        ts_df = ts_df.fillna(method='ffill').fillna(method='bfill')
                        
                        # Ensure we have enough data
                        if len(ts_df) < max(5, forecast_periods):
                            st.error(f"Not enough data points for forecasting. Need at least {max(5, forecast_periods)}.")
                            return
                        
                        # Generate forecast
                        if model_type == "ARIMA":
                            # Train ARIMA model
                            arima_order = (1, 1, 1)  # Default order
                            
                            # Create model
                            model = ARIMA(ts_df[target_col], order=arima_order)
                            model_fit = model.fit()
                            
                            # Make forecast
                            forecast = model_fit.forecast(steps=forecast_periods)
                            
                            # Create forecast dataframe
                            last_date = ts_df.index[-1]
                            forecast_dates = pd.date_range(start=last_date, periods=forecast_periods + 1, freq=ts_df.index.freq)[1:]
                            
                            forecast_df = pd.DataFrame({
                                date_col: forecast_dates,
                                target_col: forecast,
                                'type': 'Forecast'
                            })
                            
                            historical_df = pd.DataFrame({
                                date_col: ts_df.index,
                                target_col: ts_df[target_col],
                                'type': 'Historical'
                            })
                            
                            # Combine historical and forecast
                            result_df = pd.concat([historical_df, forecast_df])
                            
                        else:  # XGBoost
                            # Create lagged features
                            max_lag = min(5, len(ts_df) // 5)
                            
                            # Create features with lag values
                            for lag in range(1, max_lag + 1):
                                ts_df[f'lag_{lag}'] = ts_df[target_col].shift(lag)
                            
                            # Drop rows with NaN from lagging
                            lag_df = ts_df.dropna()
                            
                            # Prepare train data
                            X = lag_df.drop(columns=[target_col])
                            y = lag_df[target_col]
                            
                            # Train XGBoost model
                            model = xgb.XGBRegressor(objective='reg:squarederror')
                            model.fit(X, y)
                            
                            # Generate future values
                            future_df = pd.DataFrame()
                            last_values = ts_df.tail(max_lag)[target_col].values
                            
                            # Last known date
                            last_date = ts_df.index[-1]
                            forecast_dates = pd.date_range(start=last_date, periods=forecast_periods + 1, freq=ts_df.index.freq)[1:]
                            
                            # Make forecasts one step at a time
                            forecasts = []
                            
                            for _ in range(forecast_periods):
                                # Create feature row
                                features = {}
                                for i in range(1, max_lag + 1):
                                    features[f'lag_{i}'] = last_values[-i]
                                
                                # Make prediction
                                prediction = model.predict(pd.DataFrame([features]))[0]
                                forecasts.append(prediction)
                                
                                # Update last values
                                last_values = np.append(last_values, prediction)
                                last_values = last_values[1:]
                            
                            # Create forecast dataframe
                            forecast_df = pd.DataFrame({
                                date_col: forecast_dates,
                                target_col: forecasts,
                                'type': 'Forecast'
                            })
                            
                            historical_df = pd.DataFrame({
                                date_col: ts_df.index,
                                target_col: ts_df[target_col],
                                'type': 'Historical'
                            })
                            
                            # Combine historical and forecast
                            result_df = pd.concat([historical_df, forecast_df])
                        
                        # Visualize forecast
                        st.write(f"**Time Series Forecast for {target_col}**")
                        
                        fig = px.line(
                            result_df,
                            x=date_col,
                            y=target_col,
                            color='type',
                            title=f"Time Series Forecast for {target_col}",
                            color_discrete_map={'Historical': 'blue', 'Forecast': 'red'}
                        )
                        
                        # Add range to forecast
                        if model_type == "ARIMA":
                            # Add confidence intervals
                            conf_int = model_fit.get_forecast(steps=forecast_periods).conf_int()
                            lower_bound = conf_int.iloc[:, 0]
                            upper_bound = conf_int.iloc[:, 1]
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=forecast_dates,
                                    y=upper_bound,
                                    fill=None,
                                    mode='lines',
                                    line_color='rgba(255,0,0,0.2)',
                                    name='Upper Bound'
                                )
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=forecast_dates,
                                    y=lower_bound,
                                    fill='tonexty',
                                    mode='lines',
                                    line_color='rgba(255,0,0,0.2)',
                                    name='Lower Bound'
                                )
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display forecast values
                        st.write("**Forecast Values:**")
                        forecast_table = forecast_df[[date_col, target_col]].copy()
                        forecast_table[target_col] = forecast_table[target_col].round(2)
                        st.dataframe(forecast_table)
                        
                        # Calculate forecast statistics
                        last_historical = historical_df[target_col].iloc[-1]
                        first_forecast = forecast_df[target_col].iloc[0]
                        last_forecast = forecast_df[target_col].iloc[-1]
                        
                        # Calculate changes
                        immediate_change = first_forecast - last_historical
                        immediate_pct_change = (immediate_change / last_historical) * 100 if last_historical != 0 else 0
                        
                        total_change = last_forecast - last_historical
                        total_pct_change = (total_change / last_historical) * 100 if last_historical != 0 else 0
                        
                        # Display forecast insights
                        st.write("**Forecast Insights:**")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "Immediate Forecast Change",
                                f"{immediate_change:.2f}",
                                delta=f"{immediate_pct_change:.2f}%"
                            )
                            
                            st.write(f"Latest actual value: {last_historical:.2f}")
                            st.write(f"First forecast value: {first_forecast:.2f}")
                        
                        with col2:
                            st.metric(
                                f"Total Change after {forecast_periods} periods",
                                f"{total_change:.2f}",
                                delta=f"{total_pct_change:.2f}%"
                            )
                            
                            avg_forecast = forecast_df[target_col].mean()
                            st.write(f"Average forecast value: {avg_forecast:.2f}")
                            st.write(f"Last forecast value: {last_forecast:.2f}")
                        
                        # Trend direction
                        forecast_direction = "increasing" if total_change > 0 else "decreasing" if total_change < 0 else "stable"
                        
                        # Forecast summary
                        st.info(f"""
                        **Forecast Summary:**
                        The {model_type} model predicts that {target_col} will be {forecast_direction} over the next {forecast_periods} periods.
                        The model forecasts a change from {last_historical:.2f} to {last_forecast:.2f}, 
                        representing a {abs(total_pct_change):.2f}% {"increase" if total_pct_change > 0 else "decrease"}.
                        """)
                        
                except Exception as e:
                    st.error(f"Error during forecasting: {str(e)}")
