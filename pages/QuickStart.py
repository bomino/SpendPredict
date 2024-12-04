import streamlit as st
from config.styles import get_css, apply_style

def show():
    st.set_page_config(
        page_title="Quick Start - ML Prediction Platform",
        page_icon="🚀",
        layout="wide"
    )
    
    st.markdown(get_css(), unsafe_allow_html=True)

    st.markdown("""
    <div class="app-header">
        <h1>🚀 Quick Start Guide</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## Step 1: Prepare Your Data
    """)
    
    with st.expander("Data Requirements", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### Required Columns
            - **Date Column**: Transaction or event dates
            - **Category Column**: Grouping variables (e.g., department, product category)
            - **Target Column**: Numerical value to predict (e.g., spending amount)
            """)
        
        with col2:
            st.markdown("""
            ### Sample CSV Format
            ```csv
            Date,Category,Amount
            2023-01-01,Electronics,1500
            2023-01-01,Clothing,800
            2023-01-02,Electronics,2000
            ```
            """)

    st.markdown("""
    ## Step 2: Train Your First Model
    """)
    
    with st.expander("Training Steps", expanded=True):
        st.markdown("""
        1. Go to "Train New Model" tab
        2. Upload your CSV file
        3. Select your columns:
           - Choose the date column
           - Pick categorical columns
           - Select target column
        4. Choose a model type (start with 'random_forest')
        5. Click "Train Model"
        """)

    st.markdown("""
    ## Step 3: Make Predictions
    """)
    
    with st.expander("Prediction Process", expanded=True):
        st.markdown("""
        1. Switch to "Model Management" tab
        2. Select your trained model
        3. Upload new data for predictions
        4. Download prediction results
        """)

    st.markdown("""
    ## Tips for Best Results
    """)
    
    with st.expander("Quick Tips", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### Data Quality
            - Clean, consistent date format
            - No missing values
            - Consistent category names
            - Remove outliers
            """)
        
        with col2:
            st.markdown("""
            ### Model Selection
            - Start with Random Forest
            - Try XGBoost for better accuracy
            - Use Stacking for complex patterns
            - Monitor model metrics
            """)

    st.markdown("""
    ## Need Help?
    Check the About page for detailed documentation or contact support for assistance.
    """)

if __name__ == "__main__":
    show()