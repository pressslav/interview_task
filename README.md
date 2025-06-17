# Project Setup and Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
Python version 3.9 is needed to run the specific version of the Airflow UI I'm using

### 2. Prepare Data
```bash
python prepare_data.py
```

### 3. Start MLflow Server
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```
Then go to `http://localhost:5000/`

### 4. Make a Training Run on Full Dataset
```bash
python model_training.py --data_path ../kaggle_datasets/telecom_customer_churn.csv --n_estimators 150 --max_depth 20
```

### 5. Make a Training Run on Small Dataset
```bash
python model_training.py --data_path ../data/customer_churn_small.csv --n_estimators 150 --max_depth 20
```

### 6. Stop MLflow Server
Stop the server with `^C`

### 7. Run Terraform
Run `terraform init` and then `terraform apply` to spin up local docker registry and mlflow server

### 8. Verify Containers
Run `docker container ls` to prove both are spun up

### 9. (Optional) Run Airflow
Use `airflow standalone` to spin up airflow server

### 10. (Optional) Install Airflow with Constraints
(Must be constraint to local version of python otherwise it breaks)
```bash
pip install "apache-airflow==2.9.*" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.0/constraints-3.9.txt"
```
Then run `airflow db migrate`
Then run `airflow standalone`

### 11. Generate Model Artifacts
Run the following to get the `.pkl` and `.json` files in `/models`:
```bash
python model_training.py --data_path ../kaggle_datasets/telecom_customer_churn.csv --n_estimators 150 --max_depth 20 --model_output_path ../models/model.pkl
```

### 12. Build the Docker Image
```bash
docker build -t churn-predictor -f Dockerfile .
```

### 13. Push to Local Docker Registry

### 14. Run the Docker Service
```bash
docker run -d -p 8000:8000 churn-predictor
```

### 15. Get a Prediction
Input customer data sample to get prediction.
```bash
curl -X 'POST' 'http://localhost:8000/predict' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"Gender": "Female", "Paperless_Billing": "Yes", "Payment_Method": "Credit Card", "Avg_Monthly_GB_Download": 50, "Total_Long_Distance_Charges": 450}'

curl -X 'POST' 'http://localhost:8000/predict' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"City": "San Diego", "Zip_Code": 92101, "Latitude": 32.7157, "Longitude": -117.1611, "Tenure_in_Months": 24, "Contract": "One Year", "Internet_Service": "Yes", "Streaming_TV": "Yes", "Streaming_Music": "Yes", "Unlimited_Data": "Yes", "Payment_Method": "Bank Withdrawal", "Monthly_Charge": 85.00}'
```

### 16. Show Latency and Number of Runs
```bash
docker exec $(docker ps -q) cat /app/serve/app.log
```
