from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
}

with DAG(
    '00_test_connection',
    default_args=default_args,
    description='A simple hello world to test if Airflow works',
    schedule_interval=None,
    catchup=False,
    tags=['test'],
) as dag:

    t1 = BashOperator(
        task_id='print_hello',
        bash_command='echo "Hello! Airflow is working correctly."',
    )

    t2 = BashOperator(
        task_id='print_date',
        bash_command='date',
    )

    t1 >> t2
