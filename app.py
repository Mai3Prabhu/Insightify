# app.py
from flask import Flask, render_template, request, redirect, send_file, jsonify
import pandas as pd
import os
from utils.eda import generate_report
from utils.feature_engineering import handle_nulls, encode_categoricals, scale_features, handle_outliers
from utils.ml_models import train_and_evaluate_model # Import new ML function
import requests # For making HTTP requests to the Gemini API

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handles file upload, saves it, and displays a preview."""
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    df = None
    # Support both CSV and Excel
    if file.filename.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif file.filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(filepath)
    else:
        return "Unsupported file format! Please upload a CSV or Excel file."

    # Get column names to pass to the template for ML feature selection
    column_names = df.columns.tolist()
    
    return render_template(
        "eda.html",
        tables=[df.head().to_html(classes='data', header="true")],
        filename=file.filename,
        column_names=column_names # Pass column names
    )

@app.route('/eda/<filename>')
def eda(filename):
    """Generates and serves the EDA report for the uploaded file."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return "File not found!", 404

    try:
        report_path = generate_report(filepath)
        return send_file(report_path, as_attachment=False)
    except Exception as e:
        return f"Error generating EDA report: {e}", 500

@app.route('/transform/<filename>', methods=['POST'])
def transform(filename):
    """Applies selected feature engineering transformations and provides the transformed file for download."""
    null_option = request.form.get('null_option')
    encoding_option = request.form.get('encoding_option')
    scaling_option = request.form.get('scaling_option')
    outlier_option = request.form.get('outlier_option') # New: Get outlier option

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return "File not found!", 404

    df = None
    if filename.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(filepath)
    else:
        return "Unsupported file format for transformation!", 400

    try:
        # Apply transformations in a logical order
        if null_option:
            df = handle_nulls(df, null_option)
        if outlier_option: # New: Apply outlier handling
            df = handle_outliers(df, outlier_option)
        if encoding_option:
            df = encode_categoricals(df, encoding_option)
        if scaling_option:
            df = scale_features(df, scaling_option)

        output_filename = 'transformed_' + filename
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        df.to_csv(output_path, index=False) # Save as CSV for simplicity

        return send_file(output_path, as_attachment=True, download_name=output_filename)
    except Exception as e:
        return f"Error during transformation: {e}", 500

@app.route('/train_model/<filename>', methods=['POST'])
def train_model(filename):
    """
    Handles training and evaluation of a machine learning model.
    Assumes the file has already been preprocessed if needed.
    """
    request_data = request.json
    target_column = request_data.get('target_column')
    model_type = request_data.get('model_type')
    hyperparameters = request_data.get('hyperparameters', {}) # New: Get hyperparameters

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found!"}), 404

    df = None
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(filepath)
    else:
        return jsonify({"error": "Unsupported file format for model training!"}), 400

    # Call the ML model training function, passing hyperparameters
    metrics = train_and_evaluate_model(df, target_column, model_type, hyperparameters)
    
    return jsonify(metrics)


@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    """
    Handles AI assistant queries by sending them to the Gemini API.
    Provides context about Insightify to the AI.
    """
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'response': 'No message provided.'}), 400

    # API key is now an empty string, allowing Canvas to inject it or for local hardcoding
    api_key = ""
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    # Provide context to the AI about what "Insightify" is and instruct it to use Markdown
    chat_history = [
        {
            "role": "user",
            "parts": [
                {"text": "You are an AI assistant for a web application called 'Insightify'. Insightify helps users perform Exploratory Data Analysis (EDA) and Feature Engineering on their uploaded CSV or Excel datasets. Users can upload a file, view a data preview, generate a full EDA report, and apply transformations like handling missing values (drop, mean, median, mode imputation), encoding categorical features (One-Hot, Label encoding), scaling numerical features (Standard, Min-Max scaling), and handling outliers (cap, remove). You can also train basic machine learning models (Linear Regression for regression, Decision Tree for classification, Random Forest, K-Nearest Neighbors) and get evaluation metrics, including basic hyperparameter tuning. Your goal is to guide users on how to use these specific functionalities within Insightify. Do not mention that you are an AI model or that you do not have access to the UI. Just provide guidance as if you are part of the application itself. Always format your responses using clear and concise Markdown. Prioritize bullet points for lists and use bold text sparingly for key terms, not for entire phrases or sentences. Avoid excessive use of asterisks or other special characters for emphasis."}
            ]
        },
        {"role": "model", "parts": [{"text": "Understood. I am ready to assist users with Insightify's features for data analysis and feature engineering, and now also with basic machine learning model training and evaluation. I will ensure my responses are clearly formatted using concise Markdown, prioritizing bullet points and using bold text judiciously."}]},
        {
            "role": "user",
            "parts": [
                {"text": user_message}
            ]
        }
    ]

    payload = {
        "contents": chat_history
    }

    try:
        response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        result = response.json()

        if result.get('candidates') and len(result['candidates']) > 0 and \
           result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts') and \
           len(result['candidates'][0]['content']['parts']) > 0:
            ai_response = result['candidates'][0]['content']['parts'][0]['text']
        else:
            ai_response = "I'm sorry, I couldn't generate a response. Please try again."

        return jsonify({'response': ai_response})

    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return jsonify({'response': f"Failed to connect to AI assistant: {e}"}), 500
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({'response': f"An unexpected error occurred: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
