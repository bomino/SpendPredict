import pandas as pd
import streamlit as st
import json
import os
from datetime import datetime
import pickle
import plotly.graph_objects as go

class ModelManager:
    def __init__(self, storage_path='models'):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.metadata_file = os.path.join(storage_path, 'model_metadata.json')
        self.load_metadata()

    def load_metadata(self):
        try:
            print(f"Storage path: {self.storage_path}")  # Debug
            print(f"Files in storage: {os.listdir(self.storage_path)}")  # Debug
            
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}
        except Exception as e:
            print(f"Error loading metadata: {str(e)}")
            self.metadata = {}
            if os.path.exists(self.metadata_file):
                os.remove(self.metadata_file)

    def save_metadata(self):
        metadata_copy = {}
        for model_id, data in self.metadata.items():
            metadata_copy[model_id] = {
                'type': str(data['type']),
                'timestamp': str(data['timestamp']),
                'metrics': {k: float(v) for k, v in data['metrics'].items()},
                'path': str(data['path']),
                'test_data_path': str(data['test_data_path']) if data.get('test_data_path') else None,
                'config': data.get('config')
            }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata_copy, f, indent=2)


    def save_model(self, model, metrics, model_type, test_data=None, config=None):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_id = f"{model_type}_{timestamp}"
        
        model_path = os.path.join(self.storage_path, f"{model_id}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Save test data separately
        test_data_path = None
        if test_data:
            test_data_path = os.path.join(self.storage_path, f"{model_id}_test_data.pkl")
            with open(test_data_path, 'wb') as f:
                pickle.dump(test_data, f)
        
        self.metadata[model_id] = {
            'type': model_type,
            'timestamp': timestamp,
            'metrics': {
                'r2': float(metrics['r2']),
                'rmse': float(metrics['rmse']),
                'mae': float(metrics['mae']),
                'cv_score': float(metrics['cv_scores'].mean())
            },
            'path': model_path,
            'test_data_path': test_data_path,
            'config': config
        }
        self.save_metadata()
        return model_id

    def get_model(self, model_id):
        if model_id in self.metadata:
            with open(self.metadata[model_id]['path'], 'rb') as f:
                return pickle.load(f)
        return None

    def get_model_test_data(self, model_id):
        if model_id in self.metadata and self.metadata[model_id].get('test_data_path'):
            with open(self.metadata[model_id]['test_data_path'], 'rb') as f:
                return pickle.load(f)
        return None

    def get_model_config(self, model_id):
        return self.metadata[model_id].get('config', None)

    def delete_model(self, model_id):
        if model_id in self.metadata:
            # Delete model file
            if os.path.exists(self.metadata[model_id]['path']):
                os.remove(self.metadata[model_id]['path'])
            
            # Delete test data if exists
            if self.metadata[model_id].get('test_data_path') and \
               os.path.exists(self.metadata[model_id]['test_data_path']):
                os.remove(self.metadata[model_id]['test_data_path'])
            
            del self.metadata[model_id]
            self.save_metadata()
            return True
        return False

    def display_comparison(self):
        if not self.metadata:
            st.warning("No models saved yet.")
            return

        df = pd.DataFrame.from_dict(
            {k: v['metrics'] for k, v in self.metadata.items()}, 
            orient='index'
        )
        df['model_type'] = [v['type'] for v in self.metadata.values()]
        df['timestamp'] = [v['timestamp'] for v in self.metadata.values()]
        
        st.subheader("Model Comparison")
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df.round(4))
        
        with col2:
            fig = go.Figure()
            for metric in ['r2', 'rmse', 'mae']:
                fig.add_trace(go.Bar(
                    name=metric.upper(),
                    x=df.index,
                    y=df[metric],
                    text=df[metric].round(3)
                ))
            
            fig.update_layout(
                barmode='group',
                title="Model Metrics Comparison",
                xaxis_title="Model ID",
                yaxis_title="Value"
            )
            st.plotly_chart(fig)
        
        st.subheader("Model Management")
        col1, col2 = st.columns(2)
        
        with col1:
            models_to_delete = st.multiselect(
                "Select models to delete:",
                options=list(self.metadata.keys())
            )
        
        with col2:
            if st.button("Delete Selected Models"):
                for model_id in models_to_delete:
                    if self.delete_model(model_id):
                        st.success(f"Model {model_id} deleted successfully!")
                    else:
                        st.error(f"Failed to delete model {model_id}")
                st.rerun()