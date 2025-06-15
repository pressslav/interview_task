import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import mlflow.data
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib
import json

def plot_feature_importance(model, feature_names, top_n=20):
    """Generates and saves a feature importance plot for the top N features."""
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)

    #Select top N features to use
    top_n_idx = sorted_idx[-top_n:]
    top_n_features = np.array(feature_names)[top_n_idx]
    top_n_importance = feature_importance[top_n_idx]

    fig, ax = plt.subplots(figsize=(10, top_n / 2.5))
    ax.barh(range(len(top_n_features)), top_n_importance, align='center')
    ax.set_yticks(range(len(top_n_features)))
    ax.set_yticklabels(top_n_features)
    ax.set_title(f"Top {top_n} Feature Importance")

    plt.tight_layout()
    return fig

def main(args):
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment("churn_prediction")

    with mlflow.start_run():
        mlflow.log_param("data_path", args.data_path)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("top_n_features", args.top_n_features)

        print(f"Loading data from {args.data_path}")
        df = pd.read_csv(args.data_path)

        #I converted churned and stayed to 1 and 0 respectively to have a numerial representation 
        #at the end I take all the 1s and 0s and create a new column called Churn
        df['Churn'] = df['Customer Status'].apply(lambda x: 1 if x == 'Churned' else 0).astype(np.float64)
        
        #Drop irrelevant columns and the original target which I'm trying to predict
        df = df.drop(columns=['Customer ID', 'Customer Status', 'Churn Category', 'Churn Reason'])
        
        #creating a new dataframe with the one-hot encoded  categorical features in 1 and 0
        X = pd.get_dummies(df.drop('Churn', axis=1), drop_first=True)
        y = df['Churn']
        
        #I'm converting integer columns to float to avoid schema enforcement errors with missing values
        print("Converting integer columns to float to handle potential missing values")
        for col in X.columns:
            if pd.api.types.is_integer_dtype(X[col]):
                X[col] = X[col].astype(np.float64)
        
        feature_names = X.columns.tolist()

        #Data spliting for  20% test, 80% train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        #Log training data as an input
        print("Logging training data...")
        training_dataset = mlflow.data.from_pandas(X_train.join(y_train), targets="Churn", name="customer_churn_training")
        mlflow.log_input(training_dataset)

        #Training the random forest classifier
        print("Training RandomForestClassifier...")
        classifier_model = RandomForestClassifier(
            n_estimators=args.n_estimators, 
            max_depth=args.max_depth, 
            random_state=42
        )
        classifier_model.fit(X_train, y_train)
        
        #I'm saving the names by taking the python object and converting it and saving it to a JSON file
        if args.feature_names_output_path:
            output_dir = os.path.dirname(args.feature_names_output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(args.feature_names_output_path, 'w') as f:
                json.dump(feature_names, f)
            print(f"Feature names saved to {args.feature_names_output_path}")

        #Evaluation of the model
        y_pred = classifier_model.predict(X_test)
        y_proba = classifier_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy}")
        print(f"AUC: {auc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        #logging the metrics to MLflow to be able to see them in the dashboard
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        
        ##added a limit to the number of features to plot because they didn't fit in one picture
        ##set the default to 20, but it's changeable via cli
        print("Creating feature importance chart...")
        feature_importance_fig = plot_feature_importance(
            classifier_model, 
            feature_names, 
            top_n=args.top_n_features
        )
        mlflow.log_figure(feature_importance_fig, "feature_importance.png")
        
        mlflow.sklearn.log_model(
            sk_model=classifier_model,
            name="random_forest_model",
            registered_model_name="random_forest_model",
            input_example=X_train.head()
        )
        
        #Save the model to a .pkl file if path is provided so i can use it after
        if args.model_output_path:
            # Ensure the directory exists
            output_dir = os.path.dirname(args.model_output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            joblib.dump(classifier_model, args.model_output_path)
            print(f"Model saved to {args.model_output_path}")
        
        #Save the accuracy to a file so the dag.py script can read it and make a decision
        if args.output_path:
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            with open(args.output_path, 'w') as f:
                f.write(str(accuracy))
            print(f"Accuracy ({accuracy}) saved to {args.output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a customer churn prediction model.")
    
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data CSV (mandatory).")
    parser.add_argument("--output_path", type=str, help="Path to save the accuracy score.")
    parser.add_argument("--model_output_path", type=str, help="Path to save the trained model as a .pkl file.")
    parser.add_argument("--feature_names_output_path", type=str, help="Path to save the feature names as a .json file.")
    parser.add_argument("--run_name", type=str, default=None, help="Name for the MLflow run.")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees.")
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum depth of the trees.")
    parser.add_argument("--top_n_features", type=int, default=20, help="Number of top features to plot.")
    parser.add_argument("--mlflow_tracking_uri", type=str, default="http://127.0.0.1:5000", help="MLflow tracking server URI.")
    
    args = parser.parse_args()
    main(args)