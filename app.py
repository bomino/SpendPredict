import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64
from preprocessing import display_df_info, preprocess_data
from model import train_model, add_features
from model_management import ModelManager
from config.styles import get_css, apply_style,ColorPalette


def display_detailed_metrics(model, metrics, y_test, y_pred, target_col):
    col1, col2, col3, col4 = st.columns(4)
    colors = ColorPalette()
    
    metrics_data = [
        (col1, 'RMSE', metrics['rmse'], colors.primary),
        (col2, 'MAE', metrics['mae'], colors.success),
        (col3, 'R² Score', metrics['r2'], colors.warning),
        (col4, 'CV Score', metrics['cv_scores'].mean(), metrics['cv_scores'].std(), colors.error)
    ]
    
    for col, title, *values, color in metrics_data:
        with col:
            value = f"{values[0]:.3f} ± {values[1]:.3f}" if len(values) > 1 else f"{values[0]:.3f}"
            st.markdown(apply_style("metric",
                value=value,
                label=title
            ), unsafe_allow_html=True)

def create_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f"""
        <div class="card">
            <a href="data:file/csv;base64,{b64}" 
               download="predictions.csv" 
               class="stButton">
                <button>Download Predictions CSV</button>
            </a>
        </div>
    """

def display_error_analysis(y_test, y_pred, metrics):
    error_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Absolute_Error': np.abs(y_test - y_pred),
        'Percent_Error': metrics['percent_errors']
    })
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(apply_style("metric",
            value=f"{metrics['percent_errors'].mean():.1f}%",
            label="Average Error %"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(apply_style("metric",
            value=f"{np.percentile(metrics['percent_errors'], 90):.1f}%",
            label="90th Percentile Error"
        ), unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Largest Prediction Errors")
    error_display = error_df.nlargest(10, 'Percent_Error')
    formatted_errors = pd.DataFrame({
        'Actual': error_display['Actual'].round(2),
        'Predicted': error_display['Predicted'].round(2),
        'Absolute_Error': error_display['Absolute_Error'].round(2),
        'Percent_Error': error_display['Percent_Error'].round(1).astype(str) + '%'
    })
    st.dataframe(formatted_errors, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_model_details(selected_metadata):
    st.subheader("Model Details")
    timestamp = selected_metadata['timestamp']
    
    # Convert stored timestamp format to desired display format
    formatted_date = pd.to_datetime(timestamp, format='%Y%m%d%H%M%S').strftime('%m-%d-%Y at %H:%M:%S')
    
    st.markdown(apply_style("metric",
        value=selected_metadata['type'],
        label="Model Type"
    ), unsafe_allow_html=True)
    st.markdown(apply_style("metric",
        value=formatted_date,
        label="Trained on"
    ), unsafe_allow_html=True)

def display_visualizations(model, X_test, y_test, y_pred, target_col):
    col1, col2 = st.columns(2)
    with col1:
        residuals = y_test - y_pred
        fig_residuals = px.histogram(
            residuals,
            title="Distribution of Prediction Errors",
            labels={'value': 'Error', 'count': 'Frequency'},
            template='plotly_white'
        )
        fig_residuals.update_layout(
            showlegend=False,
            title_x=0.5,
            title_font_size=20,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_residuals, use_container_width=True)
    
    with col2:
        fig_scatter = px.scatter(
            x=y_test, y=y_pred,
            labels={'x': f'Actual {target_col}', 'y': f'Predicted {target_col}'},
            title="Actual vs Predicted Values"
        )
        fig_scatter.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        fig_scatter.update_layout(
            template='plotly_white',
            title_x=0.5,
            title_font_size=20,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    if hasattr(model[-1], 'feature_importances_'):
        feature_names = model[0].get_feature_names_out()
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model[-1].feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig_importance = px.bar(
            importance_df.tail(10),
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 10 Most Important Features'
        )
        fig_importance.update_layout(
            template='plotly_white',
            title_x=0.5,
            title_font_size=20,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_importance, use_container_width=True)



def display_all_tabs(model, metrics, X_test, y_test, y_pred, target_col):
    tab1, tab2, tab3 = st.tabs([
        "📊 Key Metrics", 
        "📈 Visualizations", 
        "🎯 Error Analysis"
    ])
    
    with tab1:
        display_detailed_metrics(model, metrics, y_test, y_pred, target_col)
    with tab2:
        display_visualizations(model, X_test, y_test, y_pred, target_col)
    with tab3:
        display_error_analysis(y_test, y_pred, metrics)


def main():
   st.set_page_config(
       page_title="ML Prediction Platform",
       page_icon="🤖",
       layout="wide",
       initial_sidebar_state="expanded"
   )
   st.markdown(get_css(), unsafe_allow_html=True)
   colors = ColorPalette()

   # Initialize session states
   if 'model_manager' not in st.session_state:
       st.session_state.model_manager = ModelManager()
   for state in ['model', 'model_config', 'was_log_transformed', 'previous_tab']:
       if state not in st.session_state:
           st.session_state[state] = None

   # Header
   st.markdown("""
       <div class="app-header">
           <h1>🤖 ML Prediction Platform</h1>
           <p>Build, train, and manage your machine learning models</p>
       </div>
   """, unsafe_allow_html=True)

   # Model count metric
   st.markdown(apply_style("metric",
       value=len(st.session_state.model_manager.metadata),
       label="Active Models"
   ), unsafe_allow_html=True)

   # Main tabs
   tab1, tab2 = st.tabs(["📊 Model Management", "🔧 Train New Model"])
   current_tab = "Model Management" if tab1._active else "Train New Model"
   
   if current_tab == "Model Management" and st.session_state.previous_tab != current_tab:
       st.session_state.model_manager = ModelManager()
   st.session_state.previous_tab = current_tab

   # Model Management Tab
   with tab1:
       if st.session_state.model_manager.metadata:
           st.markdown('<div class="card">', unsafe_allow_html=True)
           st.session_state.model_manager.display_comparison()
           st.markdown('</div>', unsafe_allow_html=True)
           
           st.markdown('<div class="card">', unsafe_allow_html=True)
           st.header("Select Model for Predictions")
           model_options = {f"{k} ({v['type']} - {v['timestamp']})": k 
                       for k, v in st.session_state.model_manager.metadata.items()}
           selected_model_name = st.selectbox("Select Model", list(model_options.keys()))
           selected_model_id = model_options[selected_model_name]
           
           if selected_model_id:
               selected_model = st.session_state.model_manager.get_model(selected_model_id)
               selected_metadata = st.session_state.model_manager.metadata[selected_model_id]
               
               col1, col2 = st.columns(2)
               with col1:
                   st.subheader("Model Details")
                   st.markdown(apply_style("metric",
                       value=selected_metadata['type'],
                       label="Type"
                   ), unsafe_allow_html=True)
                   st.markdown(apply_style("metric",
                       value=selected_metadata['timestamp'],
                       label="Trained on"
                   ), unsafe_allow_html=True)
               
               with col2:
                   st.subheader("Model Performance")
                   metrics_df = pd.DataFrame([{
                       'R² Score': selected_metadata['metrics']['r2'],
                       'RMSE': selected_metadata['metrics']['rmse'],
                       'MAE': selected_metadata['metrics']['mae'],
                       'CV Score': selected_metadata['metrics']['cv_score']
                   }])
                   st.dataframe(metrics_df.round(4), use_container_width=True)
               
               # Model performance tabs
               if selected_metadata.get('test_data'):
                   X_test, y_test, y_pred = selected_metadata['test_data']
                   display_all_tabs(
                       selected_model,
                       selected_metadata['metrics'],
                       X_test, y_test, y_pred,
                       selected_metadata['config']['target_col']
                   )
               
               # Predictions section
               st.markdown('<div class="card">', unsafe_allow_html=True)
               st.header("Make Predictions")
               new_file = st.file_uploader("Upload new data", type="csv", key="new_data")
               
               if new_file:
                   try:
                       # Process new data
                       new_df = pd.read_csv(new_file)
                       config = st.session_state.model_manager.get_model_config(selected_model_id)
                       new_df_processed, _ = preprocess_data(
                           new_df,
                           config['date_col'],
                           config['cat_cols'],
                           config['target_col']
                       )
                       
                       new_df_processed, _ = add_features(
                           new_df_processed,
                           config['date_col'],
                           config['cat_cols'],
                           config['target_col']
                       )
                       
                       # Make predictions
                       predictions = selected_model.predict(
                           new_df_processed[config['required_columns']]
                       )
                       
                       # Handle log transformation
                       if st.session_state.was_log_transformed:
                           predictions = np.expm1(predictions)
                           actual_values = np.expm1(new_df_processed[config['target_col']])
                       else:
                           actual_values = new_df_processed[config['target_col']]
                       
                       # Display results
                       results_df = pd.DataFrame({
                           'Date': new_df_processed[config['date_col']],
                           'FinalCategory': new_df_processed[config['cat_cols'][0]],
                           'Actual_Spend': actual_values,
                           'Predicted_Spend': predictions,
                           'Difference': actual_values - predictions,
                           'Percent_Error': abs((actual_values - predictions) / actual_values) * 100
                       })
                       
                       st.markdown('<div class="card">', unsafe_allow_html=True)
                       st.dataframe(results_df.round(2), use_container_width=True)
                       st.markdown(create_download_link(results_df), unsafe_allow_html=True)
                       st.markdown('</div>', unsafe_allow_html=True)
                       
                   except Exception as e:
                       st.error(f"Error making predictions: {str(e)}")
               st.markdown('</div>', unsafe_allow_html=True)
           st.markdown('</div>', unsafe_allow_html=True)
       else:
           st.markdown(apply_style("status",
               status="warning",
               text="No trained models available. Switch to 'Train New Model' tab to train one."
           ), unsafe_allow_html=True)

   # Training Tab
   with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Training Data", type="csv")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            
            # Main content area: Data Preview
            st.header("Data Preview")
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(df.head(), use_container_width=True)
            with col2:
                st.code(display_df_info(df))

            # Sidebar: Model Configuration
            with st.sidebar:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.header("Model Configuration")
                date_col = st.selectbox("Date Column", df.columns)
                cat_cols = st.multiselect("Categorical Columns", df.columns)
                target_col = st.selectbox("Target Column", df.columns)
                model_type = st.selectbox(
                    "Model Type",
                    ["random_forest", "gradient_boosting", "stacking", 
                        "xgboost", "lightgbm", "catboost"]
                )
                
                train_model_btn = st.button("Train Model", type="primary")
                st.markdown('</div>', unsafe_allow_html=True)

            # Main content area: Model Training Results
            if train_model_btn:
                with st.spinner("Training model..."):
                    try:
                        df_processed, log_transformed = preprocess_data(df, date_col, cat_cols, target_col)
                        st.session_state.was_log_transformed = log_transformed
                        
                        model, metrics, X_test, y_test, y_pred = train_model(
                            df_processed, date_col, cat_cols, target_col, model_type
                        )
                        
                        st.session_state.model = model
                        st.session_state.model_config = {
                            'date_col': date_col,
                            'cat_cols': cat_cols,
                            'target_col': target_col,
                            'required_columns': df_processed.columns.tolist()
                        }
                        
                        model_id = st.session_state.model_manager.save_model(
                            model, 
                            metrics, 
                            model_type,
                            test_data=(X_test, y_test, y_pred),
                            config=st.session_state.model_config
                        )
                        
                        st.markdown(apply_style("status",
                            status="success",
                            text="Model trained and saved successfully!"
                        ), unsafe_allow_html=True)
                        
                        st.header("Model Performance")
                        display_all_tabs(model, metrics, X_test, y_test, y_pred, target_col)
                        
                        with st.expander("Best Model Parameters"):
                            st.json(metrics['best_params'])
                        
                    except Exception as e:
                        st.markdown(apply_style("status",
                            status="error",
                            text=f"Error training model: {str(e)}"
                        ), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
   main()