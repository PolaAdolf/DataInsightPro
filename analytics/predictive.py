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
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from utils.smart_preprocessor import SmartDataPreprocessor
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow for advanced neural networks
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

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
    st.subheader("🔮 Future Predictions")

    # Smart preprocessing first
    with st.expander("🔧 Smart Data Preparation for AI Models", expanded=False):
        st.markdown("**Let me optimize your data for the best AI predictions!** This will fix any issues and enhance your dataset.")

        if st.button("🚀 Prepare Data for AI", key="smart_prep_pred"):
            with st.spinner("Analyzing and optimizing your data..."):
                preprocessor = SmartDataPreprocessor(df)
                health_report = preprocessor.analyze_data_health()
                preprocessor.show_issues_and_solutions()
                preprocessor.auto_scale_features()

                # Auto-apply critical fixes
                for issue in preprocessor.issues:
                    if issue['type'] == 'text_in_numeric':
                        preprocessor.fix_text_in_numeric(issue['column'])
                        st.success(f"✅ Auto-fixed: Converted '{issue['column']}' to numbers")

                preprocessor.show_applied_fixes()

                # Update the dataframe if fixes were applied
                processed_df = preprocessor.get_processed_data()
                if len(processed_df) != len(df) or not processed_df.equals(df):
                    st.session_state.data = processed_df
                    st.success("🎉 Your data is now AI-ready! All prediction models will use this enhanced data.")
                    st.rerun()

    # Check if there's enough data for prediction
    if len(df) < 10:
        st.error("😅 Need at least 10 rows of data for AI predictions. Try uploading a larger dataset or use the data expansion feature above!")
        st.info("💡 **Tip:** Use the Smart Data Preparation section above to generate more examples from your existing data.")
        return

    # Make a copy and ensure all data is properly formatted
    pred_df = df.copy()

    # Auto-fix common issues that cause errors
    try:
        # Convert object columns that are actually numeric
        for col in pred_df.columns:
            if pred_df[col].dtype == 'object':
                # Try to convert to numeric
                numeric_converted = pd.to_numeric(pred_df[col], errors='coerce')
                if not numeric_converted.isna().all() and numeric_converted.notna().sum() > len(pred_df) * 0.7:
                    pred_df[col] = numeric_converted
                    pred_df[col].fillna(pred_df[col].median(), inplace=True)
                    st.info(f"🔧 Auto-fixed: Converted '{col}' from text to numbers for predictions")

        # Update validation results based on fixed data
        validation_results = {
            'numeric_columns': pred_df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': pred_df.select_dtypes(include=['object']).columns.tolist(),
            'date_columns': [col for col in pred_df.columns if 'date' in col.lower() or pred_df[col].dtype == 'datetime64[ns]'],
            'text_columns': []
        }

    except Exception as e:
        st.warning("⚠️ Found some data formatting issues. Using Smart Data Preparation above is recommended for best results.")

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
                "Choose your AI model approach:",
                ["🚀 Beginner-Friendly", "🎯 Advanced Traditional", "🧠 Deep Learning (Neural Networks)", "🌳 Decision Trees", "📊 All Models Comparison"],
                key="model_type"
            )

            # Add model recommendations based on data characteristics
            st.markdown("### 💡 Smart Model Recommendations")
            data_size = len(pred_df)
            num_features = len(selected_features)

            if data_size < 100:
                st.info("🔍 **For your small dataset:** Decision Trees or Simple models work best and are easy to understand.")
            elif data_size < 1000:
                st.info("📈 **For your medium dataset:** Traditional Advanced models (Random Forest, XGBoost) give good balance of accuracy and speed.")
            else:
                st.info("🚀 **For your large dataset:** Deep Learning Neural Networks can discover complex patterns others might miss.")

            if num_features > 10:
                st.warning("📊 **Many features detected:** Consider feature selection or use models that handle high dimensions well (Random Forest, Neural Networks).")

            if is_classification:
                unique_classes = len(pred_df[target_var].unique())
                if unique_classes > 10:
                    st.info("🏷️ **Many categories detected:** Neural Networks or Random Forest handle multi-class problems best.")

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

                        # Helper function to create TensorFlow model
                        def create_tensorflow_model(input_shape, is_classification, num_classes=None):
                            """Create a TensorFlow/Keras model"""
                            model = keras.Sequential([
                                keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
                                keras.layers.Dropout(0.3),
                                keras.layers.Dense(64, activation='relu'),
                                keras.layers.Dropout(0.2),
                                keras.layers.Dense(32, activation='relu'),
                            ])

                            if is_classification:
                                if num_classes == 2:
                                    model.add(keras.layers.Dense(1, activation='sigmoid'))
                                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                                else:
                                    model.add(keras.layers.Dense(num_classes, activation='softmax'))
                                    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                            else:
                                model.add(keras.layers.Dense(1))
                                model.compile(optimizer='adam', loss='mse', metrics=['mae'])

                            return model

                        # Get models to train based on selection
                        models_to_train = {}

                        if model_type == "🚀 Beginner-Friendly":
                            if is_classification:
                                models_to_train["Simple Logistic Model"] = LogisticRegression(max_iter=1000, random_state=42)
                            else:
                                models_to_train["Simple Linear Model"] = LinearRegression()

                        elif model_type == "🎯 Advanced Traditional":
                            if is_classification:
                                models_to_train.update({
                                    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                                    "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
                                    "Support Vector Machine": SVC(random_state=42, probability=True)
                                })
                            else:
                                models_to_train.update({
                                    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                                    "XGBoost": xgb.XGBRegressor(random_state=42, verbosity=0),
                                    "Support Vector Machine": SVR()
                                })

                        elif model_type == "🧠 Deep Learning (Neural Networks)":
                            if is_classification:
                                models_to_train["Neural Network"] = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
                            else:
                                models_to_train["Neural Network"] = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

                        elif model_type == "🌳 Decision Trees":
                            if is_classification:
                                models_to_train.update({
                                    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
                                    "Random Forest (Tree Ensemble)": RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
                                })
                            else:
                                models_to_train.update({
                                    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10),
                                    "Random Forest (Tree Ensemble)": RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
                                })

                        elif model_type == "📊 All Models Comparison":
                            if is_classification:
                                models_to_train.update({
                                    "Simple Logistic": LogisticRegression(max_iter=1000, random_state=42),
                                    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=8),
                                    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
                                    "Neural Network": MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=42)
                                })
                            else:
                                models_to_train.update({
                                    "Simple Linear": LinearRegression(),
                                    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=8),
                                    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
                                    "Neural Network": MLPRegressor(hidden_layer_sizes=(50,), max_iter=300, random_state=42)
                                })

                        # Train models and store results
                        model_results = {}
                        best_model_name = None
                        best_score = -float('inf')

                        for model_name, model in models_to_train.items():
                            st.write(f"🔄 Training {model_name}...")
                            progress_bar = st.progress(0)

                            try:
                                # Create pipeline
                                pipeline = Pipeline(steps=[
                                    ('preprocessor', preprocessor),
                                    ('model', model)
                                ])

                                # Train model
                                pipeline.fit(X_train, y_train)
                                progress_bar.progress(0.7)

                                # Make predictions
                                y_pred = pipeline.predict(X_test)
                                progress_bar.progress(1.0)

                                # Calculate metrics
                                if is_classification:
                                    accuracy = accuracy_score(y_test, y_pred)
                                    score = accuracy
                                    model_results[model_name] = {
                                        'pipeline': pipeline,
                                        'predictions': y_pred,
                                        'accuracy': accuracy,
                                        'score': score
                                    }
                                else:
                                    r2 = r2_score(y_test, y_pred)
                                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                                    score = r2
                                    model_results[model_name] = {
                                        'pipeline': pipeline,
                                        'predictions': y_pred,
                                        'r2': r2,
                                        'rmse': rmse,
                                        'score': score
                                    }

                                # Track best model
                                if score > best_score:
                                    best_score = score
                                    best_model_name = model_name

                                st.success(f"✅ {model_name} trained successfully!")
                                progress_bar.empty()

                            except Exception as e:
                                progress_bar.empty()
                                st.warning(f"⚠️ {model_name} couldn't be trained with this data. Trying simpler approach...")
                                continue

                        if not model_results:
                            st.error("😅 No models could be trained successfully. Try a different approach or check your data.")
                            return

                        # Use the best model for detailed analysis
                        if model_type == "📊 All Models Comparison":
                            st.markdown("### 🏆 Model Comparison Results")

                            # Create comparison table
                            comparison_data = []
                            for name, result in model_results.items():
                                if is_classification:
                                    comparison_data.append({
                                        'Model': name,
                                        'Accuracy': f"{result['accuracy']:.1%}",
                                        'Performance': "🌟 Excellent" if result['accuracy'] > 0.9 else 
                                                             "👍 Good" if result['accuracy'] > 0.7 else 
                                                             "👌 Fair" if result['accuracy'] > 0.5 else "📈 Needs Work"
                                    })
                                else:
                                    comparison_data.append({
                                        'Model': name,
                                        'R² Score': f"{result['r2']:.3f}",
                                        'RMSE': f"{result['rmse']:.3f}",
                                        'Performance': "🌟 Excellent" if result['r2'] > 0.8 else 
                                                             "👍 Good" if result['r2'] > 0.6 else 
                                                             "👌 Fair" if result['r2'] > 0.4 else "📈 Needs Work"
                                    })

                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df)

                            st.markdown(f"### 🏅 Best Model: {best_model_name}")

                        # Use best model for detailed analysis
                        pipeline = model_results[best_model_name]['pipeline']
                        y_pred = model_results[best_model_name]['predictions']
                        model_name = best_model_name

                        # Evaluate model
                        st.write(f"**Model Performance ({model_name}):**")

                        if is_classification:
                            try:
                                # Classification metrics
                                accuracy = accuracy_score(y_test, y_pred)

                                st.write(f"🎯 **Model Accuracy: {accuracy:.1%}**")
                                st.markdown("*This means the model correctly predicts the outcome this percentage of the time.*")

                                # Classification report
                                report = classification_report(y_test, y_pred, output_dict=True)
                                report_df = pd.DataFrame(report).transpose()

                                # Only show main classes, not averages
                                main_classes = [col for col in report_df.index if col not in ['accuracy', 'macro avg', 'weighted avg']]
                                if main_classes:
                                    st.write("📊 **Performance by Category:**")
                                    display_df = report_df.loc[main_classes, ['precision', 'recall', 'f1-score']].round(3)
                                    st.dataframe(display_df)

                                    st.markdown("""
                                    💡 **What these metrics mean:**
                                    - **Precision**: When the model says it's this category, how often is it right?
                                    - **Recall**: Of all actual cases in this category, how many did the model find?
                                    - **F1-score**: Overall performance score (higher is better)
                                    """)
                            except Exception as e:
                                st.error("😅 There was an issue calculating accuracy metrics.")
                                accuracy = 0
                                report_df = pd.DataFrame()

                            # Confusion matrix with error handling
                            try:
                                cm = confusion_matrix(y_test, y_pred)
                                unique_labels = sorted(y.unique()) if hasattr(y, 'unique') else sorted(set(y))

                                if len(unique_labels) == len(cm):
                                    cm_df = pd.DataFrame(
                                        cm,
                                        index=[f'Actual {str(i)}' for i in unique_labels],
                                        columns=[f'Predicted {str(i)}' for i in unique_labels]
                                    )

                                    st.write("🎯 **Prediction Accuracy Matrix:**")
                                    st.dataframe(cm_df)

                                    st.markdown("""
                                    💡 **How to read this:**
                                    - **Diagonal numbers** = correct predictions (higher is better)
                                    - **Off-diagonal numbers** = mistakes the AI made
                                    - **Perfect prediction** would have all numbers on the diagonal
                                    """)
                            except Exception as e:
                                st.info("📊 Confusion matrix couldn't be generated, but accuracy metrics above show model performance.")

                            # Feature importance
                            if model_type in ["🎯 Advanced Traditional", "🌳 Decision Trees", "📊 All Models Comparison"]:
                                try:
                                    # Get feature importances if the model has them
                                    if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
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
                                    st.warning(f"Could not display feature importance for this model.")


                        else: # Regression
                            try:
                                # Regression metrics
                                mse = mean_squared_error(y_test, y_pred)
                                rmse = np.sqrt(mse)
                                r2 = r2_score(y_test, y_pred)

                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric("📈 R² Score", f"{r2:.3f}")
                                    st.caption("How well the model explains your data")

                                with col2:
                                    st.metric("📏 RMSE", f"{rmse:.3f}")
                                    st.caption("Average prediction error")

                                with col3:
                                    accuracy_pct = max(0, min(100, r2 * 100))
                                    st.metric("🎯 Model Quality", f"{accuracy_pct:.1f}%")
                                    st.caption("Overall performance")

                                # Add interpretation
                                if r2 > 0.8:
                                    quality = "Excellent! 🌟"
                                elif r2 > 0.6:
                                    quality = "Good 👍"
                                elif r2 > 0.4:
                                    quality = "Fair 👌"
                                else:
                                    quality = "Needs improvement 📈"

                                st.markdown(f"""
                                💡 **What this means:**
                                - **R² Score of {r2:.3f}** means the model explains {r2*100:.1f}% of the patterns in your data
                                - **RMSE of {rmse:.3f}** is the typical prediction error
                                - **Model Quality: {quality}**
                                """)
                            except Exception as e:
                                st.error("😅 There was an issue calculating performance metrics.")
                                mse = rmse = r2 = 0

                            # Plot actual vs predicted
                            pred_df_plot = pd.DataFrame({
                                'Actual': y_test,
                                'Predicted': y_pred
                            })

                            fig = px.scatter(
                                pred_df_plot,
                                x='Actual',
                                y='Predicted',
                                title=f"Actual vs Predicted {target_var}",
                                opacity=0.7
                            )

                            # Add 45-degree line
                            min_val = min(pred_df_plot['Actual'].min(), pred_df_plot['Predicted'].min())
                            max_val = max(pred_df_plot['Actual'].max(), pred_df_plot['Predicted'].max())

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
                            if model_type in ["🎯 Advanced Traditional", "🌳 Decision Trees", "📊 All Models Comparison"]:
                                try:
                                    # Get feature importances if the model has them
                                    if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
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
                                    st.warning(f"Could not display feature importance for this model.")
