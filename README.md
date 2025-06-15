1/  pip install -r requirements.txt' for dep

2/ run python prepare_data.py

3/ start mlflow server  mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

then go to http://localhost:5000/

4/ make a training run on full dataset

python model_training.py --data_path ../kaggle_datasets/telecom_customer_churn.csv --n_estimators 150 --max_depth 20

5/ make a training run on small dataset

python model_training.py --data_path ../data/customer_churn_small.csv --n_estimators 150 --max_depth 20c

6/ stop mlflow server with ^C

7/ run  terraform init and then terraform apply to spin up local docker registry and mlflow server

8/ run  docker container ls to prove both a spun up

9/ (optional) airflow standalone to spin up airflow server

10/  #(optional for running the Airflow UI) (must be constraint to local version of python
otherwise it breaks)

pip install "apache-airflow==2.9.*" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.0/constraints-3.9.txt"

then run airflow db migrate

then airflow standalone

11/ run python model_training.py --data_path ../kaggle_datasets/telecom_customer_churn.csv --n_estimators 150 --max_depth 20 --model_output_path ../models/model.pkl

to get the .pkl and json files in /models

12/ build the docker image w/ 

docker build -t churn-predictor -f Dockerfile .

13/ push to local docker registry

14/  run the docker service w/ 
docker run -d -p 8000:8000 churn-predictor

15/ - input customer data sample to get prediction

curl -X 'POST' 'http://localhost:8000/predict' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"Gender": "Female", "Paperless_Billing": "Yes", "Payment_Method": "Credit Card", "Avg_Monthly_GB_Download": 50, "Total_Long_Distance_Charges": 450}

curl -X 'POST' 'http://localhost:8000/predict' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"City": "San Diego", "Zip_Code": 92101, "Latitude": 32.7157, "Longitude": -117.1611, "Tenure_in_Months": 24, "Contract": "One Year", "Internet_Service": "Yes", "Streaming_TV": "Yes", "Streaming_Music": "Yes", "Unlimited_Data": "Yes", "Payment_Method": "Bank Withdrawal", "Monthly_Charge": 85.00}'

16/show latency and number of runs by running

docker exec $(docker ps -q) cat /app/serve/app.log