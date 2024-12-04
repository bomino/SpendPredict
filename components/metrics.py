import streamlit as st
from config.styles import apply_style, ColorPalette

def display_detailed_metrics(model, metrics, y_test, y_pred, target_col):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(apply_style("metric", 
            value=f"{metrics['rmse']:.2f}",
            label="RMSE"
        ), unsafe_allow_html=True)
        
    with col2:
        st.markdown(apply_style("metric",
            value=f"{metrics['mae']:.2f}",
            label="MAE"
        ), unsafe_allow_html=True)
        
    with col3:
        st.markdown(apply_style("metric",
            value=f"{metrics['r2']:.3f}",
            label="R² Score"
        ), unsafe_allow_html=True)
        
    with col4:
        cv_mean = metrics['cv_scores'].mean() if 'cv_scores' in metrics else metrics['cv_score']
        cv_std = metrics['cv_scores'].std() if 'cv_scores' in metrics else 0
        st.markdown(apply_style("metric",
            value=f"{cv_mean:.3f} ± {cv_std:.3f}",
            label="CV Score"
        ), unsafe_allow_html=True)