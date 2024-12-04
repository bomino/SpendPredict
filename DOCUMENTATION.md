# ML Prediction Platform Documentation

## Table of Contents
1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Data Requirements & Preprocessing](#data-requirements--preprocessing)
4. [Model Types & Configurations](#model-types--configurations)
5. [Feature Engineering & Selection](#feature-engineering--selection)
6. [Model Management & Versioning](#model-management--versioning)
7. [Performance Metrics & Evaluation](#performance-metrics--evaluation)
8. [API Reference](#api-reference)
9. [Troubleshooting & FAQs](#troubleshooting--faqs)
10. [UI Components & Navigation](#ui-components--navigation)

## Overview

### Purpose
The ML Prediction Platform is designed for enterprise-grade machine learning workflow management, specifically focusing on spending prediction tasks. It provides an end-to-end solution from data preprocessing to model deployment.

### Key Features
- Automated ML pipeline
- Multiple algorithm support
- Interactive visualizations
- Model version control
- Real-time predictions
- Performance monitoring

## Installation & Setup

### System Requirements
- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- Modern web browser
- Internet connection for dependencies

### Dependencies
```
streamlit
streamlit-card
pandas
scikit-learn
plotly
matplotlib
xgboost
lightgbm
catboost
```

### Installation Steps
```bash
# Clone repository
git clone https://github.com/bomino/SpendPredic.git
cd ml-prediction-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run app.py
```

## Data Requirements & Preprocessing

### Data Format Specifications
- File format: CSV
- Required columns:
  - Date column: YYYY-MM-DD format
  - Categorical columns: String or numerical categories
  - Target column: Numerical values
- No missing values in critical columns
- Consistent category naming

### Sample Data Format
```csv
Date,Category,SubCategory,Amount
2023-01-01,Electronics,Laptops,1500.00
2023-01-01,Clothing,Shirts,800.50
2023-01-02,Electronics,Phones,2000.75
```

### Preprocessing Pipeline
1. **Data Validation**
   - Date format checking
   - Missing value detection
   - Category consistency verification
   - Numerical range validation

2. **Data Cleaning**
   - Missing value handling
   - Outlier detection
   - Category standardization
   - Date parsing and standardization

3. **Feature Preparation**
   - Date feature extraction
   - Category encoding
   - Numerical scaling
   - Feature interaction generation

## Model Types & Configurations

### Random Forest
```python
parameters = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10]
}
```
Best for: Balanced performance and interpretability

### XGBoost
```python
parameters = {
    'n_estimators': [500, 1000],
    'learning_rate': [0.01, 0.05],
    'max_depth': [4, 6, 8],
    'subsample': [0.8, 0.9]
}
```
Best for: High performance on large datasets

### LightGBM
```python
parameters = {
    'n_estimators': [500, 1000],
    'learning_rate': [0.01, 0.05],
    'max_depth': [4, 6, 8],
    'num_leaves': [31, 63]
}
```
Best for: Fast training on large datasets

### CatBoost
```python
parameters = {
    'iterations': [1000],
    'learning_rate': [0.01, 0.03],
    'depth': [4, 6],
    'min_data_in_leaf': [10],
    'grow_policy': ['Depthwise']
}
```
Best for: Handling categorical variables

### Gradient Boosting
```python
parameters = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0]
}
```
Best for: Robust performance on various datasets

### Stacking Ensemble
```python
base_models = [
    ('rf', RandomForestRegressor()),
    ('gb', GradientBoostingRegressor())
]
```
Best for: Combining multiple model strengths

## Feature Engineering & Selection

### Temporal Features
- Year, Month, Day extraction
- Day of week
- Week of year
- Is weekend/holiday
- Seasonal indicators
- Rolling statistics

### Categorical Features
- One-hot encoding
- Target encoding
- Frequency encoding
- Category grouping
- Interaction features

### Numerical Features
- Scaling (RobustScaler)
- Log transformation
- Polynomial features
- Binning
- Interaction terms

## Model Management & Versioning

### Model Storage
- Binary format (.pkl)
- Metadata JSON
- Performance metrics
- Configuration parameters
- Feature importance

### Version Control
```python
{
    'model_id': {
        'type': 'model_type',
        'timestamp': 'YYYY-MM-DD HH:MM:SS',
        'metrics': {
            'r2': float,
            'rmse': float,
            'mae': float,
            'cv_score': float
        },
        'path': 'model_path',
        'config': {...}
    }
}
```

### Model Selection
- Performance-based ranking
- Cross-validation scores
- Time-based filtering
- Category-specific metrics

## Performance Metrics & Evaluation

### Key Metrics
- R² Score (Coefficient of determination)
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Cross-validation scores
- Percent errors

### Visualization Components
- Actual vs Predicted plots
- Residual analysis
- Feature importance charts
- Error distribution
- Performance comparison

## API Reference

### Model Training
```python
def train_model(df, date_col, cat_cols, target_col, model_type='random_forest'):
    """
    Train a new model with specified parameters.
    
    Args:
        df (pd.DataFrame): Input dataset
        date_col (str): Date column name
        cat_cols (list): Categorical column names
        target_col (str): Target column name
        model_type (str): Algorithm selection
        
    Returns:
        tuple: (model, metrics, X_test, y_test, y_pred)
    """
```

### Model Management
```python
class ModelManager:
    def save_model(self, model, metrics, model_type, test_data=None, config=None):
        """Save trained model with metadata"""
        
    def get_model(self, model_id):
        """Retrieve trained model"""
        
    def delete_model(self, model_id):
        """Remove model from storage"""
        
    def display_comparison(self):
        """Compare model performances"""
```

### Feature Engineering
```python
def add_features(df, date_col, cat_cols, target_col):
    """
    Generate advanced features.
    
    Args:
        df (pd.DataFrame): Input dataset
        date_col (str): Date column name
        cat_cols (list): Categorical column names
        target_col (str): Target column name
        
    Returns:
        tuple: (processed_df, feature_list)
    """
```

## Troubleshooting & FAQs

### Common Issues

1. **Data Loading Errors**
   - Solution: Verify CSV format
   - Check column names
   - Validate date formats
   - Ensure no special characters

2. **Model Training Failures**
   - Solution: Check memory usage
   - Verify data types
   - Review feature engineering
   - Validate parameter ranges

3. **Prediction Errors**
   - Solution: Match input format
   - Check categorical values
   - Verify date formatting
   - Ensure all required features

### Performance Optimization
- Batch processing for large datasets
- Feature selection for dimensionality reduction
- Memory management best practices
- Computation optimization techniques

## UI Components & Navigation

### Main Interface
1. Model Management Tab
   - Model selection
   - Performance metrics
   - Prediction interface
   - Model deletion

2. Training Tab
   - Data upload
   - Column selection
   - Model configuration
   - Training progress

### Visualization Components
- Interactive plots
- Metric cards
- Error analysis
- Feature importance

### Navigation Tips
- Use sidebar for configuration
- Expand/collapse sections
- Download functionality
- Error notifications

## Support & Updates

### Getting Help
- GitHub issues
- Documentation updates
- Error reporting
- Feature requests

### Best Practices
- Regular model retraining
- Performance monitoring
- Data quality checks
- Version control management