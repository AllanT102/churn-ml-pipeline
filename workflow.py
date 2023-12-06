from model_pipeline import process_training_data
from model_training import train_logistic_regression, train_random_forest
from model_evaluator import evaluate_model
from pyspark.ml.classification import RandomForestClassificationModel, LogisticRegressionModel
import os

def analyze_churn():
    # Set file location
    file_location = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    # process the data
    (trainprepDF, testprepDF) = process_training_data(file_location=file_location)

    # # Train Logistic Regression model
    lr_model = train_logistic_regression(trainprepDF)
    rf_model = train_random_forest(trainprepDF)

    # Evaluate Logistic Regression model
    evaluate_model(lr_model, LogisticRegressionModel, testprepDF)
    
    # Evaluate Random Forest model
    evaluate_model(rf_model, RandomForestClassificationModel, testprepDF)

if __name__ == "__main__":
    if not os.path.exists("data/lrModel") and not os.path.exists("data/rfModel"):
        analyze_churn()