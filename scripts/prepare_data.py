import pandas as pd
import argparse

def prepare_data(full_data_path, sample_data_path, n_rows):
    """
    Reads the full dataset, takes a random sample, and saves it to a new CSV file in /data.
    """
    print(f"Reading full dataset from {full_data_path}")
    df = pd.read_csv(full_data_path)
    
    print(f"Taking a random sample of {n_rows} rows")
    sample_df = df.sample(n=n_rows, random_state=42)
    
    print(f"Saving sample dataset to {sample_data_path}")
    sample_df.to_csv(sample_data_path, index=False)
    print("Sample dataset created successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a smaller sample from the full dataset.")
    parser.add_argument("--full_data_path", type=str, default="../kaggle_datasets/telecom_customer_churn.csv", help="Path to the full dataset CSV.")
    parser.add_argument("--sample_data_path", type=str, default="../data/customer_churn_small.csv", help="Path to save the sampled dataset CSV.")
    parser.add_argument("--n_rows", type=int, default=1000, help="Number of rows to sample. Default is set to 1000 rows as per the task requirements.")
    
    args = parser.parse_args()
    
    prepare_data(args.full_data_path, args.sample_data_path, args.n_rows)