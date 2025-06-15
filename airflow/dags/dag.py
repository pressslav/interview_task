from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# Define the project root directory to ensure scripts are found reliably.
PROJ_ROOT = "/home/press/coding/acronis_interview_v02"
ACCURACY_DIR = f"{PROJ_ROOT}/tmp/accuracy"

def _compare_accuracies():
    """
    Reads accuracy scores from files and prints a comparison.
    """
    small_accuracy_path = f"{ACCURACY_DIR}/small_dataset_accuracy.txt"
    full_accuracy_path = f"{ACCURACY_DIR}/full_dataset_accuracy.txt"

    with open(small_accuracy_path, 'r') as f:
        small_accuracy = float(f.read())
    
    with open(full_accuracy_path, 'r') as f:
        full_accuracy = float(f.read())
    
    print(f"Accuracy Report:")
    print(f"Small dataset accuracy: {small_accuracy}")
    print(f"Full dataset accuracy:  {full_accuracy}")
    
    if full_accuracy >= small_accuracy:
        print("Model trained on the full dataset performs better, use it further")
    else:
        print("Model trained on the small dataset performs better, use it further")

with DAG(
    dag_id='customer_churn_training_dag',
    schedule_interval=None,
    catchup=False
) as dag:

    # Train on small dataset
    train_small = BashOperator(
        task_id='train_on_small_dataset',
        bash_command=f"""
            cd {PROJ_ROOT} && \
            python scripts/model_training.py \
                --data_path data/customer_churn_small.csv \
                --n_estimators 50 \
                --max_depth 5 \
                --top_n_features 10 \
                --mlflow_tracking_uri http://127.0.0.1:5000 \
                --output_path {ACCURACY_DIR}/small_dataset_accuracy.txt
        """
    )

    # --- Task to train on the full dataset ---
    train_full = BashOperator(
        task_id='train_on_full_dataset',
        bash_command=f"""
            cd {PROJ_ROOT} && \
            python scripts/model_training.py \
                --data_path kaggle_datasets/telecom_customer_churn.csv \
                --n_estimators 100 \
                --max_depth 10 \
                --top_n_features 20 \
                --mlflow_tracking_uri http://127.0.0.1:5000 \
                --output_path {ACCURACY_DIR}/full_dataset_accuracy.txt
        """
    )
    
    # Task to compare the accuracies from the two runs
    compare_results = PythonOperator(
        task_id='compare_accuracies',
        python_callable=_compare_accuracies,
    )
    
    # Define task dependencies: run training in parallel, then compare.
    [train_small, train_full] >> compare_results
