# utils/eda.py
import pandas as pd
from ydata_profiling import ProfileReport # Updated to ydata_profiling
import os
import warnings

def generate_report(filepath):
    """
    Generates a comprehensive EDA report for a given dataset.

    Args:
        filepath (str): The path to the input CSV or Excel file.

    Returns:
        str: The path to the generated HTML report.
    """
    df = None
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format for EDA. Please provide a CSV or Excel file.")

    # Generate the YData Profiling report
    # Ensure a unique report name to avoid conflicts if multiple files are uploaded
    report_filename = os.path.basename(filepath).replace('.', '_') + '_eda_report.html'
    report_path = os.path.join('uploads', report_filename) # Save reports in the uploads folder

    # Suppress warnings from ydata-profiling if desired
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Use profile_report method directly as it's the standard for ydata-profiling
    profile = df.profile_report(title=f"EDA Report for {os.path.basename(filepath)}")

    profile.to_file(report_path)
    return report_path
