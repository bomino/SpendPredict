import streamlit as st
from config.styles import get_css, apply_style

def show():
    st.set_page_config(
        page_title="About - ML Prediction Platform",
        page_icon="ℹ️",
        layout="wide"
    )
    
    st.markdown(get_css(), unsafe_allow_html=True)

    st.markdown("""
    <div class="app-header">
        <h1>ℹ️ About ML Prediction Platform</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('''
    ## Overview
    The ML Prediction Platform is a powerful tool designed to help businesses leverage machine learning for accurate spending predictions. This platform streamlines the entire machine learning workflow, from data preprocessing to model deployment and management.

    ## Key Features

    ### Multiple Model Types
    Choose from various advanced machine learning algorithms:
    - Random Forest
    - Gradient Boosting
    - XGBoost
    - LightGBM
    - CatBoost
    - Stacking Ensemble

    ### Automated Feature Engineering
    - Temporal feature extraction
    - Category-based aggregations
    - Rolling statistics
    - Lag features

    ### Intelligent Model Management
    - Version control for all trained models
    - Performance metric tracking
    - Model comparison visualization
    - Easy model selection for predictions

    ### Advanced Visualization
    - Interactive performance metrics
    - Error analysis
    - Feature importance plots
    - Prediction vs. actual comparisons
    ''')

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## How to Use

        ### Training a New Model
        1. Upload your historical data (CSV format)
        2. Select relevant columns:
           - Date column
           - Categorical features
           - Target variable (spending)
        3. Choose a model type
        4. Click "Train Model"
        """)

        st.markdown("""
        ### Making Predictions
        1. Select a trained model
        2. Upload new data
        3. Get predictions and download results
        """)

    with col2:
        st.markdown("""
        ## Best Practices
        - Ensure data quality and completeness
        - Use consistent date formats
        - Include sufficient historical data
        - Regularly retrain models with new data
        - Monitor model performance over time
        """)

        st.markdown("""
        ## Data Requirements
        - CSV format
        - Must include date column
        - At least one categorical column
        - Target variable (numerical)
        """)

    st.markdown("""
    ## Technical Specifications
    - Automated cross-validation
    - Grid search optimization
    - Robust error handling
    - Comprehensive performance metrics
    - Interactive visualization tools
    """)

if __name__ == "__main__":
    show()