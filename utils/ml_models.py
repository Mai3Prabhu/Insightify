# utils/ml_models.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor # Import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # New: Import Random Forest models
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # New: Import K-Neighbors models
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

def train_and_evaluate_model(df, target_column, model_type, hyperparameters=None):
    """
    Trains and evaluates a machine learning model with basic hyperparameter support.

    Args:
        df (pd.DataFrame): The input DataFrame (should be preprocessed/engineered).
        target_column (str): The name of the target column.
        model_type (str): The type of model to train ('linear_regression', 'decision_tree_classifier',
                          'random_forest_classifier', 'knn_classifier',
                          'decision_tree_regressor', 'random_forest_regressor', 'knn_regressor').
        hyperparameters (dict, optional): A dictionary of hyperparameters for the model. Defaults to None.

    Returns:
        dict: A dictionary containing model evaluation metrics.
              Returns an error message if target column or model type is invalid.
    """
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found in the dataset."}

    # Separate features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Ensure X contains only numerical features for model training
    X_numerical = X.select_dtypes(include=np.number)

    if X_numerical.empty:
        return {"error": "No numerical features available for model training after dropping target."}

    # Handle potential NaNs in X_numerical by filling with mean
    X_numerical = X_numerical.fillna(X_numerical.mean())

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_numerical, y, test_size=0.2, random_state=42)

    metrics = {'model_type': model_type.replace('_', ' ').title()} # Nicer display name
    model = None
    hp_used = {} # To store hyperparameters actually used by the model

    # Default hyperparameters
    if hyperparameters is None:
        hyperparameters = {}

    # --- Regression Models ---
    if model_type == 'linear_regression':
        if not pd.api.types.is_numeric_dtype(y):
            y = pd.to_numeric(y, errors='coerce')
            y.fillna(y.mean(), inplace=True)
            y_train = pd.to_numeric(y_train, errors='coerce').fillna(y_train.mean())
            y_test = pd.to_numeric(y_test, errors='coerce').fillna(y_test.mean())
            if not pd.api.types.is_numeric_dtype(y):
                return {"error": f"Target column '{target_column}' could not be converted to numerical for regression model."}
        model = LinearRegression()
        # Linear Regression typically doesn't have common hyperparameters to tune in this basic context
        
    elif model_type == 'decision_tree_regressor':
        if not pd.api.types.is_numeric_dtype(y):
            y = pd.to_numeric(y, errors='coerce')
            y.fillna(y.mean(), inplace=True)
            y_train = pd.to_numeric(y_train, errors='coerce').fillna(y_train.mean())
            y_test = pd.to_numeric(y_test, errors='coerce').fillna(y_test.mean())
            if not pd.api.types.is_numeric_dtype(y):
                return {"error": f"Target column '{target_column}' could not be converted to numerical for regression model."}
        
        dt_reg_params = {}
        if 'max_depth' in hyperparameters and hyperparameters['max_depth'] is not None:
            try:
                dt_reg_params['max_depth'] = int(hyperparameters['max_depth'])
                hp_used['max_depth'] = dt_reg_params['max_depth']
            except ValueError:
                pass # Ignore if not a valid integer

        model = DecisionTreeRegressor(random_state=42, **dt_reg_params)

    elif model_type == 'random_forest_regressor':
        if not pd.api.types.is_numeric_dtype(y):
            y = pd.to_numeric(y, errors='coerce')
            y.fillna(y.mean(), inplace=True)
            y_train = pd.to_numeric(y_train, errors='coerce').fillna(y_train.mean())
            y_test = pd.to_numeric(y_test, errors='coerce').fillna(y_test.mean())
            if not pd.api.types.is_numeric_dtype(y):
                return {"error": f"Target column '{target_column}' could not be converted to numerical for regression model."}
        
        rf_reg_params = {}
        if 'n_estimators' in hyperparameters and hyperparameters['n_estimators'] is not None:
            try:
                rf_reg_params['n_estimators'] = int(hyperparameters['n_estimators'])
                hp_used['n_estimators'] = rf_reg_params['n_estimators']
            except ValueError:
                pass
        if 'max_depth' in hyperparameters and hyperparameters['max_depth'] is not None:
            try:
                rf_reg_params['max_depth'] = int(hyperparameters['max_depth'])
                hp_used['max_depth'] = rf_reg_params['max_depth']
            except ValueError:
                pass

        model = RandomForestRegressor(random_state=42, **rf_reg_params)

    elif model_type == 'knn_regressor':
        if not pd.api.types.is_numeric_dtype(y):
            y = pd.to_numeric(y, errors='coerce')
            y.fillna(y.mean(), inplace=True)
            y_train = pd.to_numeric(y_train, errors='coerce').fillna(y_train.mean())
            y_test = pd.to_numeric(y_test, errors='coerce').fillna(y_test.mean())
            if not pd.api.types.is_numeric_dtype(y):
                return {"error": f"Target column '{target_column}' could not be converted to numerical for regression model."}
        
        knn_reg_params = {}
        if 'n_neighbors' in hyperparameters and hyperparameters['n_neighbors'] is not None:
            try:
                knn_reg_params['n_neighbors'] = int(hyperparameters['n_neighbors'])
                hp_used['n_neighbors'] = knn_reg_params['n_neighbors']
            except ValueError:
                pass
        
        model = KNeighborsRegressor(**knn_reg_params) # No random_state for KNN

    # --- Classification Models ---
    elif model_type == 'decision_tree_classifier':
        if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
            try:
                le = LabelEncoder()
                y_train = le.fit_transform(y_train)
                y_test = le.transform(y_test)
            except Exception as e:
                return {"error": f"Target column '{target_column}' could not be encoded for classification. Error: {e}"}
        elif not pd.api.types.is_integer_dtype(y):
            return {"error": f"Target column '{target_column}' is not suitable for classification. Please ensure it's categorical or integer labels."}
        
        dt_clf_params = {}
        if 'max_depth' in hyperparameters and hyperparameters['max_depth'] is not None:
            try:
                dt_clf_params['max_depth'] = int(hyperparameters['max_depth'])
                hp_used['max_depth'] = dt_clf_params['max_depth']
            except ValueError:
                pass

        model = DecisionTreeClassifier(random_state=42, **dt_clf_params)

    elif model_type == 'random_forest_classifier':
        if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
            try:
                le = LabelEncoder()
                y_train = le.fit_transform(y_train)
                y_test = le.transform(y_test)
            except Exception as e:
                return {"error": f"Target column '{target_column}' could not be encoded for classification. Error: {e}"}
        elif not pd.api.types.is_integer_dtype(y):
            return {"error": f"Target column '{target_column}' is not suitable for classification. Please ensure it's categorical or integer labels."}
        
        rf_clf_params = {}
        if 'n_estimators' in hyperparameters and hyperparameters['n_estimators'] is not None:
            try:
                rf_clf_params['n_estimators'] = int(hyperparameters['n_estimators'])
                hp_used['n_estimators'] = rf_clf_params['n_estimators']
            except ValueError:
                pass
        if 'max_depth' in hyperparameters and hyperparameters['max_depth'] is not None:
            try:
                rf_clf_params['max_depth'] = int(hyperparameters['max_depth'])
                hp_used['max_depth'] = rf_clf_params['max_depth']
            except ValueError:
                pass

        model = RandomForestClassifier(random_state=42, **rf_clf_params)

    elif model_type == 'knn_classifier':
        if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
            try:
                le = LabelEncoder()
                y_train = le.fit_transform(y_train)
                y_test = le.transform(y_test)
            except Exception as e:
                return {"error": f"Target column '{target_column}' could not be encoded for classification. Error: {e}"}
        elif not pd.api.types.is_integer_dtype(y):
            return {"error": f"Target column '{target_column}' is not suitable for classification. Please ensure it's categorical or integer labels."}
        
        knn_clf_params = {}
        if 'n_neighbors' in hyperparameters and hyperparameters['n_neighbors'] is not None:
            try:
                knn_clf_params['n_neighbors'] = int(hyperparameters['n_neighbors'])
                hp_used['n_neighbors'] = knn_clf_params['n_neighbors']
            except ValueError:
                pass
        
        model = KNeighborsClassifier(**knn_clf_params) # No random_state for KNN

    else:
        return {"error": "Invalid model type. Choose from available regression or classification models."}

    if model:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if 'regression' in model_type: # Check if it's a regression model
            metrics['Mean Squared Error (MSE)'] = mean_squared_error(y_test, y_pred)
            metrics['R2 Score'] = r2_score(y_test, y_pred)
        elif 'classification' in model_type: # Check if it's a classification model
            metrics['Accuracy'] = accuracy_score(y_test, y_pred)
            metrics['Precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['Recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['F1 Score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        if hp_used:
            metrics['Hyperparameters Used'] = hp_used

        return metrics
    else:
        return {"error": "Model initialization failed."}
