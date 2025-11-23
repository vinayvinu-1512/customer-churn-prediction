# src/data_preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    df = df.dropna()
    return df

def split_xy(df, target_col='Churn'):
    X = df.drop(columns=[target_col])
    y = df[target_col].apply(lambda x: 1 if str(x).strip().lower() in ['yes','y','true','1'] else 0)
    return X, y

def build_preprocessor(X):
    numeric_feats = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_feats = X.select_dtypes(include=['object','bool']).columns.tolist()

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_feats),
        ('cat', categorical_pipeline, categorical_feats)
    ])

    return preprocessor
