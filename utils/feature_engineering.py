# utils/feature_engineering.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np

def handle_nulls(df, strategy):
    """
    Handles missing values in the DataFrame based on the specified strategy.

    Args:
        df (pd.DataFrame): The input DataFrame.
        strategy (str): The strategy to use ('drop', 'mean', 'median', 'mode').

    Returns:
        pd.DataFrame: The DataFrame with missing values handled.
    """
    df_copy = df.copy()
    if strategy == 'drop':
        # Drop rows where any column has a missing value
        df_copy.dropna(inplace=True)
    elif strategy == 'mean':
        # Impute numerical columns with their mean
        for col in df_copy.select_dtypes(include=np.number).columns:
            df_copy[col].fillna(df_copy[col].mean(), inplace=True)
    elif strategy == 'median':
        # Impute numerical columns with their median
        for col in df_copy.select_dtypes(include=np.number).columns:
            df_copy[col].fillna(df_copy[col].median(), inplace=True)
    elif strategy == 'mode':
        # Impute all columns (numerical and categorical) with their mode
        for col in df_copy.columns:
            # Mode can return multiple values if there's a tie, take the first
            mode_val = df_copy[col].mode()[0]
            df_copy[col].fillna(mode_val, inplace=True)
    return df_copy

def encode_categoricals(df, method):
    """
    Encodes categorical features in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        method (str): The encoding method ('onehot', 'label').

    Returns:
        pd.DataFrame: The DataFrame with categorical features encoded.
    """
    df_copy = df.copy()
    categorical_cols = df_copy.select_dtypes(include='object').columns

    if method == 'onehot':
        # Apply One-Hot Encoding
        for col in categorical_cols:
            # Handle potential NaN values in categorical columns before one-hot encoding
            # Convert NaNs to a string 'Missing' or similar, or drop them if preferred
            df_copy[col] = df_copy[col].fillna('Missing_Category').astype(str)
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_data = encoder.fit_transform(df_copy[[col]])
            encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([col]), index=df_copy.index)
            df_copy = pd.concat([df_copy.drop(columns=[col]), encoded_df], axis=1)
    elif method == 'label':
        # Apply Label Encoding
        for col in categorical_cols:
            # Handle potential NaN values in categorical columns before label encoding
            df_copy[col] = df_copy[col].fillna('Missing_Category').astype(str)
            encoder = LabelEncoder()
            df_copy[col] = encoder.fit_transform(df_copy[col])
    return df_copy

def scale_features(df, method):
    """
    Scales numerical features in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        method (str): The scaling method ('standard', 'minmax').

    Returns:
        pd.DataFrame: The DataFrame with numerical features scaled.
    """
    df_copy = df.copy()
    numerical_cols = df_copy.select_dtypes(include=np.number).columns

    if method == 'standard':
        # Apply Standard Scaling
        scaler = StandardScaler()
        df_copy[numerical_cols] = scaler.fit_transform(df_copy[numerical_cols])
    elif method == 'minmax':
        # Apply Min-Max Scaling
        scaler = MinMaxScaler()
        df_copy[numerical_cols] = scaler.fit_transform(df_copy[numerical_cols])
    return df_copy

def handle_outliers(df, method, factor=1.5):
    """
    Handles outliers in numerical columns using the IQR method.

    Args:
        df (pd.DataFrame): The input DataFrame.
        method (str): The method to handle outliers ('cap', 'remove').
                      'cap': Caps outliers at the upper/lower bounds.
                      'remove': Removes rows containing outliers.
        factor (float): The IQR multiplier (default is 1.5).

    Returns:
        pd.DataFrame: The DataFrame with outliers handled.
    """
    df_copy = df.copy()
    numerical_cols = df_copy.select_dtypes(include=np.number).columns

    for col in numerical_cols:
        Q1 = df_copy[col].quantile(0.25)
        Q3 = df_copy[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        if method == 'cap':
            df_copy[col] = np.where(df_copy[col] < lower_bound, lower_bound, df_copy[col])
            df_copy[col] = np.where(df_copy[col] > upper_bound, upper_bound, df_copy[col])
        elif method == 'remove':
            df_copy = df_copy[~((df_copy[col] < lower_bound) | (df_copy[col] > upper_bound))]
    return df_copy
