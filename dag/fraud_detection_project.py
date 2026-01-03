"""
Fraud Detection Pipeline Project
Safe Mode: Imports are inside functions to prevent DAG loading errors.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

# We dynamically find your airflow_home
AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME", os.path.expanduser("~/airflow_home"))
BASE_PATH = os.path.join(AIRFLOW_HOME, "fraud_project")

# Define paths
DATA_PATH = os.path.join(BASE_PATH, "data")
RESULTS_PATH = os.path.join(BASE_PATH, "results")
FIGURES_PATH = os.path.join(RESULTS_PATH, "figures")
TABLES_PATH = os.path.join(RESULTS_PATH, "tables")

# Create directories if they don't exist
for path in [DATA_PATH, FIGURES_PATH, TABLES_PATH]:
    os.makedirs(path, exist_ok=True)

default_args = {
    'owner': 'fraud-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 0,
    'start_date': days_ago(1),
}

dag = DAG(
    'fraud_detection_project',
    default_args=default_args,
    description='Final Fraud Detection Pipeline',
    schedule_interval=None,
    catchup=False,
    tags=['fraud_project'],
)

# =============================================================================
# TASKS
# =============================================================================

def task_generate_data():
    """Generates mock data for the project."""
    import pandas as pd
    import numpy as np
    
    print(f"Generating data in: {DATA_PATH}")
    
    # Mock Credit Card Data
    cc_file = os.path.join(DATA_PATH, "creditcard.csv")
    if not os.path.exists(cc_file):
        n = 1000
        df = pd.DataFrame({
            'Time': np.arange(n),
            'Amount': np.random.uniform(10, 500, n),
            'Class': np.random.choice([0, 1], n, p=[0.9, 0.1]),
            'V1': np.random.normal(0, 1, n),
            'V2': np.random.normal(0, 1, n)
        })
        df.to_csv(cc_file, index=False)
        print("Created mock creditcard.csv")
    
    return "Data Generated"

def task_clean_data():
    """Cleans the raw data."""
    import pandas as pd
    
    cc_file = os.path.join(DATA_PATH, "creditcard.csv")
    df = pd.read_csv(cc_file)
    
    # Simple cleaning: Drop duplicates
    df_clean = df.drop_duplicates()
    
    output_file = os.path.join(DATA_PATH, "clean_data.csv")
    df_clean.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")

def task_train_model():
    """Trains a simple logistic regression model."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    
    # Load data
    data_file = os.path.join(DATA_PATH, "clean_data.csv")
    df = pd.read_csv(data_file)
    
    X = df[['Time', 'Amount', 'V1', 'V2']]
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc}")
    
    # Save a simple plot
    fig, ax = plt.subplots()
    ax.bar(['Accuracy'], [acc])
    ax.set_title(f"Model Accuracy: {acc:.2f}")
    plot_path = os.path.join(FIGURES_PATH, "accuracy_plot.png")
    fig.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    
    # Save a text summary
    with open(os.path.join(TABLES_PATH, "summary.txt"), "w") as f:
        f.write(f"Model run completed.\nAccuracy: {acc}\n")

# =============================================================================
# OPERATORS
# =============================================================================

t1 = PythonOperator(
    task_id='1_generate_mock_data',
    python_callable=task_generate_data,
    dag=dag,
)

t2 = PythonOperator(
    task_id='2_clean_data',
    python_callable=task_clean_data,
    dag=dag,
)

t3 = PythonOperator(
    task_id='3_train_model',
    python_callable=task_train_model,
    dag=dag,
)

t1 >> t2 >> t3