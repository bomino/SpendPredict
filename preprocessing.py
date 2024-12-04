import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from datetime import datetime

def display_df_info(df):
   info = []
   info.append(f"Total Rows: {len(df)}")
   info.append(f"Total Columns: {len(df.columns)}")
   info.append("\nColumn Info:")
   for col in df.columns:
       dtype = str(df[col].dtype)
       nulls = df[col].isnull().sum()
       info.append(f"{col}: {dtype} | Null Values: {nulls}")
   return "\n".join(info)

def create_date_features(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col])
    
    try:
        df['Year'] = df[date_col].dt.year
        df['Month'] = df[date_col].dt.month 
        df['DayOfWeek'] = df[date_col].dt.dayofweek
        df['Quarter'] = df[date_col].dt.quarter
        df['WeekOfYear'] = df[date_col].dt.isocalendar().week
        df['IsWeekend'] = df[date_col].dt.dayofweek.isin([5, 6]).astype(int)
        df['Season'] = df[date_col].dt.month % 12 // 3 + 1
        
        holidays = pd.DataFrame({
            'date': pd.date_range(start=df[date_col].min(), end=df[date_col].max()),
            'is_holiday': 0
        })
        
        df['IsHoliday'] = df[date_col].map(
            holidays.set_index('date')['is_holiday']).fillna(0)
            
        return df
    except Exception as e:
        raise Exception(f"Error creating date features: {str(e)}")

def create_lag_features(df, date_col, target_col, cat_cols):
   """Create lagged and rolling features"""
   df = df.sort_values(date_col)
   
   for cat in cat_cols:
       # Lag features
       for lag in [1, 7, 30]:
           df[f'lag_{lag}_{cat}'] = df.groupby(cat)[target_col].shift(lag)
       
       # Rolling statistics
       for window in [7, 30]:
           df[f'rolling_mean_{window}_{cat}'] = df.groupby(cat)[target_col].transform(
               lambda x: x.rolling(window=window, min_periods=1).mean())
           df[f'rolling_std_{window}_{cat}'] = df.groupby(cat)[target_col].transform(
               lambda x: x.rolling(window=window, min_periods=1).std())
   
   return df

def handle_outliers(df, target_col):
   """Remove outliers using RobustScaler"""
   scaler = RobustScaler()
   scaled_target = scaler.fit_transform(df[[target_col]])
   return df[abs(scaled_target.ravel()) < 3]

def preprocess_data(df, date_col, cat_cols, target_col):
   """Main preprocessing pipeline"""
   df = df.copy()
   
   # Handle outliers
   df = handle_outliers(df, target_col)
   
   # Create features
   df = create_date_features(df, date_col)
   df = create_lag_features(df, date_col, target_col, cat_cols)
   
   # Fill missing values
   df = df.ffill().bfill()
   
   # Log transform if skewed
   was_log_transformed = False
   if df[target_col].skew() > 1:
       df[target_col] = np.log1p(df[target_col])
       was_log_transformed = True
   
   return df, was_log_transformed