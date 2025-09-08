import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import SMOTE for advanced data augmentation
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

class SmartDataPreprocessor:
    """Smart data preprocessing for non-technical users"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
        self.issues = []
        self.recommendations = []
        self.applied_fixes = []
        
    def analyze_data_health(self):
        """Analyze data and identify issues"""
        st.markdown("### 🔍 Smart Data Health Check")
        
        health_report = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_data': self.df.isnull().sum().sum(),
            'text_in_numeric': 0,
            'mixed_types': 0,
            'small_dataset': len(self.df) < 100,
            'unbalanced_categories': False
        }
        
        # Check for text in numeric columns
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Try to convert to numeric
                numeric_test = pd.to_numeric(self.df[col], errors='coerce')
                if not numeric_test.isna().all():
                    # Some values can be converted to numeric
                    if numeric_test.notna().sum() > len(self.df) * 0.5:
                        health_report['text_in_numeric'] += 1
                        self.issues.append({
                            'type': 'text_in_numeric',
                            'column': col,
                            'description': f"'{col}' has numbers stored as text",
                            'impact': "This prevents mathematical calculations and predictions",
                            'solution': "Convert text numbers to actual numbers"
                        })
        
        # Check for mixed data types
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Check if column has mixed content
                sample_values = self.df[col].dropna().astype(str)
                if len(sample_values) > 0:
                    # Check for dates
                    try:
                        pd.to_datetime(sample_values.iloc[0])
                        date_count = 0
                        for val in sample_values.head(10):
                            try:
                                pd.to_datetime(val)
                                date_count += 1
                            except:
                                pass
                        if date_count > 5:
                            self.issues.append({
                                'type': 'date_formatting',
                                'column': col,
                                'description': f"'{col}' contains dates but isn't recognized as dates",
                                'impact': "Time-based analysis won't work properly",
                                'solution': "Convert to proper date format"
                            })
                    except:
                        pass
        
        # Check for small dataset
        if health_report['small_dataset']:
            self.issues.append({
                'type': 'small_dataset',
                'column': 'all',
                'description': f"Only {len(self.df)} rows of data",
                'impact': "AI models need more examples for accurate predictions",
                'solution': "Generate more similar examples based on your existing data"
            })
        
        # Display health report
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Rows", health_report['total_rows'])
            color = "🟢" if health_report['total_rows'] >= 100 else "🟡" if health_report['total_rows'] >= 50 else "🔴"
            st.caption(f"{color} {'Good size' if health_report['total_rows'] >= 100 else 'Could use more data'}")
        
        with col2:
            st.metric("📋 Columns", health_report['total_columns'])
            st.caption("🟢 Ready for analysis")
        
        with col3:
            st.metric("❌ Missing Values", health_report['missing_data'])
            color = "🟢" if health_report['missing_data'] == 0 else "🟡"
            st.caption(f"{color} {'Perfect' if health_report['missing_data'] == 0 else 'Can be fixed'}")
        
        with col4:
            issues_count = len(self.issues)
            st.metric("⚠️ Issues Found", issues_count)
            color = "🟢" if issues_count == 0 else "🟡" if issues_count <= 2 else "🔴"
            st.caption(f"{color} {'All good' if issues_count == 0 else 'Fixable issues'}")
        
        return health_report
    
    def show_issues_and_solutions(self):
        """Display issues with user-friendly solutions"""
        if not self.issues:
            st.success("🎉 Great news! Your data looks perfect and ready for analysis!")
            return
        
        st.markdown("### 🛠️ Smart Fixes Available")
        st.markdown("I found some issues that I can fix automatically to make your analysis more powerful:")
        
        for i, issue in enumerate(self.issues):
            with st.expander(f"🔧 Fix #{i+1}: {issue['description']}", expanded=True):
                st.markdown(f"**Problem:** {issue['impact']}")
                st.markdown(f"**Solution:** {issue['solution']}")
                
                if issue['type'] == 'text_in_numeric':
                    if st.button(f"✅ Convert '{issue['column']}' to numbers", key=f"fix_numeric_{i}"):
                        self.fix_text_in_numeric(issue['column'])
                        st.success(f"✅ Fixed! '{issue['column']}' is now ready for calculations.")
                        st.rerun()
                
                elif issue['type'] == 'date_formatting':
                    if st.button(f"📅 Fix date format in '{issue['column']}'", key=f"fix_date_{i}"):
                        self.fix_date_formatting(issue['column'])
                        st.success(f"✅ Fixed! '{issue['column']}' is now recognized as dates.")
                        st.rerun()
                
                elif issue['type'] == 'small_dataset':
                    st.markdown("**Options for expanding your dataset:**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🎯 Generate Smart Examples", key=f"augment_smart_{i}"):
                            self.smart_data_augmentation()
                            st.success("✅ Added intelligent examples based on your data patterns!")
                            st.rerun()
                    
                    with col2:
                        if st.button("📈 Create Variations", key=f"augment_simple_{i}"):
                            self.simple_data_augmentation()
                            st.success("✅ Created realistic variations of your existing data!")
                            st.rerun()
    
    def fix_text_in_numeric(self, column):
        """Convert text numbers to actual numbers"""
        try:
            # Convert to numeric, replacing errors with NaN
            self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
            
            # Handle any remaining NaN values
            if self.df[column].isna().any():
                # Fill NaN with median for numeric data
                median_val = self.df[column].median()
                self.df[column].fillna(median_val, inplace=True)
            
            self.applied_fixes.append(f"Converted '{column}' from text to numbers")
            
        except Exception as e:
            st.error(f"😅 Couldn't fix '{column}' automatically. It might need manual cleaning.")
    
    def fix_date_formatting(self, column):
        """Fix date formatting issues"""
        try:
            # Try to convert to datetime
            self.df[column] = pd.to_datetime(self.df[column], errors='coerce')
            self.applied_fixes.append(f"Fixed date format in '{column}'")
            
        except Exception as e:
            st.error(f"😅 Couldn't fix dates in '{column}' automatically. Try a different format.")
    
    def smart_data_augmentation(self):
        """Generate intelligent additional data using advanced techniques"""
        try:
            original_size = len(self.df)
            target_size = max(200, original_size * 3)  # At least 200 rows or 3x current size
            
            # Separate numeric and categorical columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
            
            if len(numeric_cols) >= 2 and SMOTE_AVAILABLE:
                # Use SMOTE for generating synthetic examples
                X = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
                
                # Create a temporary target variable for SMOTE
                y = np.random.randint(0, 2, len(X))
                
                # Calculate how many samples to generate
                samples_needed = target_size - original_size
                
                if samples_needed > 0:
                    # Use SMOTE to generate new samples
                    smote = SMOTE(sampling_strategy='auto', k_neighbors=min(5, len(X)-1), random_state=42)
                    
                    # Create balanced dataset
                    unique_classes, class_counts = np.unique(y, return_counts=True)
                    max_count = max(class_counts)
                    target_count = max_count + (samples_needed // len(unique_classes))
                    
                    # Manually create sampling strategy
                    sampling_strategy = {cls: target_count for cls in unique_classes}
                    smote.sampling_strategy = sampling_strategy
                    
                    X_resampled, y_resampled = smote.fit_resample(X, y)
                    
                    # Create new dataframe with synthetic data
                    new_data = pd.DataFrame(X_resampled, columns=numeric_cols)
                    
                    # For categorical columns, sample from existing values
                    for cat_col in categorical_cols:
                        if cat_col in self.df.columns:
                            # Sample from existing categorical values
                            existing_values = self.df[cat_col].dropna().values
                            if len(existing_values) > 0:
                                new_data[cat_col] = np.random.choice(existing_values, len(new_data))
                    
                    # Keep only the new synthetic samples
                    synthetic_data = new_data.iloc[len(self.df):].copy()
                    
                    # Append synthetic data to original
                    self.df = pd.concat([self.df, synthetic_data], ignore_index=True)
                    
                    self.applied_fixes.append(f"Generated {len(synthetic_data)} intelligent examples using AI patterns")
            
            else:
                # Fallback to simple augmentation if not enough numeric columns
                self.simple_data_augmentation()
                
        except Exception as e:
            st.warning("🔄 Using simpler data generation method...")
            self.simple_data_augmentation()
    
    def simple_data_augmentation(self):
        """Simple data augmentation using bootstrapping and noise injection"""
        try:
            original_size = len(self.df)
            target_size = min(300, original_size * 2)  # Double the data, max 300 rows
            samples_needed = target_size - original_size
            
            if samples_needed > 0:
                # Bootstrap sampling with small modifications
                bootstrap_indices = np.random.choice(len(self.df), size=samples_needed, replace=True)
                bootstrap_data = self.df.iloc[bootstrap_indices].copy()
                
                # Add small noise to numeric columns
                numeric_cols = bootstrap_data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if bootstrap_data[col].std() > 0:
                        noise = np.random.normal(0, bootstrap_data[col].std() * 0.05, len(bootstrap_data))
                        bootstrap_data[col] = bootstrap_data[col] + noise
                
                # Reset index for the bootstrap data
                bootstrap_data.reset_index(drop=True, inplace=True)
                
                # Append to original data
                self.df = pd.concat([self.df, bootstrap_data], ignore_index=True)
                
                self.applied_fixes.append(f"Created {samples_needed} realistic variations of your existing data")
                
        except Exception as e:
            st.error(f"😅 Couldn't generate additional data automatically: {str(e)}")
    
    def auto_scale_features(self):
        """Automatically scale numeric features for better AI performance"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) <= 1:
            return
        
        # Check if scaling is needed
        ranges = {}
        for col in numeric_cols:
            col_range = self.df[col].max() - self.df[col].min()
            ranges[col] = col_range
        
        max_range = max(ranges.values())
        min_range = min([r for r in ranges.values() if r > 0])  # Avoid division by zero
        
        # If ranges differ by more than 100x, suggest scaling
        if max_range / min_range > 100:
            st.markdown("### ⚖️ Smart Scaling Suggestion")
            st.markdown(f"""
            I noticed your number columns use very different scales:
            - Largest range: {max_range:,.0f}
            - Smallest range: {min_range:,.0f}
            
            **Why this matters:** AI models might think the bigger numbers are more important than smaller ones.
            
            **Solution:** Make all columns use similar scales (0-1) so the AI treats them fairly.
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🎯 Yes, make scales fair", key="apply_scaling"):
                    scaler = MinMaxScaler()
                    self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
                    self.applied_fixes.append("Applied smart scaling to make all number columns use similar ranges")
                    st.success("✅ All number columns now use fair scales (0-1)!")
                    st.rerun()
            
            with col2:
                if st.button("⏭️ Skip scaling", key="skip_scaling"):
                    st.info("Keeping original scales. You can always apply this later.")
    
    def get_processed_data(self):
        """Return the processed dataframe"""
        return self.df
    
    def show_applied_fixes(self):
        """Show summary of all applied fixes"""
        if self.applied_fixes:
            st.markdown("### ✅ Applied Improvements")
            for fix in self.applied_fixes:
                st.success(f"✅ {fix}")
            
            # Show before/after comparison
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Data:**")
                st.write(f"• {len(self.original_df)} rows")
                st.write(f"• {len(self.original_df.columns)} columns")
            
            with col2:
                st.markdown("**Enhanced Data:**")
                st.write(f"• {len(self.df)} rows (+{len(self.df) - len(self.original_df)})")
                st.write(f"• {len(self.df.columns)} columns")
                
            if len(self.df) > len(self.original_df):
                improvement = ((len(self.df) - len(self.original_df)) / len(self.original_df)) * 100
                st.info(f"📈 Dataset expanded by {improvement:.0f}% for better AI predictions!")