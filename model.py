import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import streamlit as st

def calculate_metrics(y_true, y_pred):
   return {
       'mae': mean_absolute_error(y_true, y_pred),
       'mse': mean_squared_error(y_true, y_pred),
       'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
       'r2': r2_score(y_true, y_pred),
       'percent_errors': np.abs((y_true - y_pred) / y_true) * 100
   }


def get_model_config(model_type):
   if model_type == 'random_forest':
       base_model = RandomForestRegressor(random_state=42)
       param_grid = {
           'regressor__n_estimators': [100, 200, 500],
           'regressor__max_depth': [10, 20, 30, None],
           'regressor__min_samples_leaf': [1, 2, 4],
           'regressor__min_samples_split': [2, 5, 10]
       }
   elif model_type == 'gradient_boosting':
       base_model = GradientBoostingRegressor(random_state=42, loss='huber')
       param_grid = {
           'regressor__n_estimators': [100, 200, 500],
           'regressor__learning_rate': [0.01, 0.05, 0.1],
           'regressor__max_depth': [3, 5, 7],
           'regressor__subsample': [0.8, 0.9, 1.0]
       }
   elif model_type == 'stacking':
       estimators = [
           ('rf', RandomForestRegressor(random_state=42)),
           ('gb', GradientBoostingRegressor(random_state=42))
       ]
       base_model = StackingRegressor(
           estimators=estimators,
           final_estimator=RandomForestRegressor(n_estimators=100, random_state=42),
           cv=5
       )
       param_grid = {
           'regressor__final_estimator__n_estimators': [100, 200],
           'regressor__final_estimator__max_depth': [10, 20]
       }
   elif model_type == 'xgboost':
       base_model = XGBRegressor(random_state=42)
       param_grid = {
           'regressor__n_estimators': [500, 1000],
           'regressor__learning_rate': [0.01, 0.05],
           'regressor__max_depth': [4, 6, 8],
           'regressor__subsample': [0.8, 0.9]
       }
   elif model_type == 'lightgbm':
       base_model = LGBMRegressor(random_state=42)
       param_grid = {
           'regressor__n_estimators': [500, 1000],
           'regressor__learning_rate': [0.01, 0.05],
           'regressor__max_depth': [4, 6, 8],
           'regressor__num_leaves': [31, 63]
       }
   elif model_type == 'catboost':
        base_model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3,
            min_data_in_leaf=10,
            grow_policy='Depthwise',
            random_strength=1,
            bagging_temperature=1,
            od_type='Iter',
            od_wait=20,
            verbose=False,
            random_state=42,
            allow_writing_files=False,
            task_type='CPU',
            thread_count=-1,
            loss_function='RMSE',
            eval_metric='RMSE',
            bootstrap_type='Bayesian',
            leaf_estimation_method='Newton'
        )
        
        param_grid = {
            'regressor__iterations': [1000],
            'regressor__learning_rate': [0.01, 0.03],
            'regressor__depth': [4, 6],
            'regressor__min_data_in_leaf': [10],
            'regressor__grow_policy': ['Depthwise']
        }

   # Add log transform status for all models
   st.session_state.was_log_transformed = True
   return base_model, param_grid

# In model.py

def create_advanced_features(df, date_col, cat_cols, target_col):
    # Enhanced temporal features
    df['DayOfMonth'] = df[date_col].dt.day
    df['WeekOfMonth'] = df[date_col].dt.day.apply(lambda x: (x-1)//7 + 1)
    df['DayOfYear'] = df[date_col].dt.dayofyear
    
    # Category-based features
    for cat in cat_cols:
        # Target encoding with cross-validation
        df[f'{cat}_mean_spend'] = df.groupby(cat)[target_col].transform('mean')
        df[f'{cat}_median_spend'] = df.groupby(cat)[target_col].transform('median')
        
        # Time-based category features
        df[f'{cat}_month_spend'] = df.groupby([cat, 'Month'])[target_col].transform('mean')
        df[f'{cat}_quarter_spend'] = df.groupby([cat, 'Quarter'])[target_col].transform('mean')
        
        # Frequency features
        df[f'{cat}_count'] = df.groupby(cat)[target_col].transform('count')
        
    return df


def add_features(df, date_col, cat_cols, target_col):
    numeric_features = ['Year', 'Month', 'DayOfWeek', 'Quarter', 'WeekOfYear',
                       'IsWeekend', 'Season', 'IsHoliday']
    numeric_features.extend([col for col in df.columns if 'lag_' in col or 'rolling_' in col])
    
    for cat in cat_cols:
        df[f'{cat}_mean_spend'] = df.groupby(cat)[target_col].transform('mean')
        df[f'{cat}_median_spend'] = df.groupby(cat)[target_col].transform('median')
        df[f'{cat}_month_spend'] = df.groupby([cat, 'Month'])[target_col].transform('mean')
        df[f'{cat}_quarter_spend'] = df.groupby([cat, 'Quarter'])[target_col].transform('mean')
        df[f'{cat}_count'] = df.groupby(cat)[target_col].transform('count')
        numeric_features.extend([f'{cat}_mean_spend', f'{cat}_median_spend',
                              f'{cat}_month_spend', f'{cat}_quarter_spend', f'{cat}_count'])
    
    return df, numeric_features

def train_model(df, date_col, cat_cols, target_col, model_type='random_forest'):
    df, numeric_features = add_features(df, date_col, cat_cols, target_col)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
        ]
    )
    
    base_model, param_grid = get_model_config(model_type)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', base_model)
    ])
    
    X = df[[date_col] + cat_cols + numeric_features]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'],
        refit='r2',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    y_pred = grid_search.predict(X_test)
    
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'cv_scores': cross_val_score(grid_search.best_estimator_, X, y, cv=5, scoring='r2'),
        'percent_errors': np.abs((y_test - y_pred) / y_test) * 100,
        'best_params': grid_search.best_params_,
        'feature_names': numeric_features + cat_cols
    }
    
    return grid_search.best_estimator_, metrics, X_test, y_test, y_pred
