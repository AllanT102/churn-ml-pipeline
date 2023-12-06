from pyspark.sql import SparkSession
from flask import Flask, request, jsonify
from pyspark.ml.classification import RandomForestClassificationModel, LogisticRegressionModel
from model_pipeline import process_input_data

app = Flask(__name__)

def predict_churn(df, type):
    try:
        # Load the trained model
        model = None
        if type == "lr":
            model_path = "data/lrModel"
            model = LogisticRegressionModel.load(model_path)
        elif type == "rf":
            model_path = "data/rfModel"
            model = RandomForestClassificationModel.load(model_path)

        # Transform the data
        model_data = process_input_data(df)

        # Make predictions
        predictions = model.transform(model_data)
        if predictions.count() == 0:
            return {"error": "No predictions was made"}    
        
        results = predictions.select(['prediction', 'customerID'])
        return results.toPandas().to_json(orient="records")
    except Exception as e:
        print(e)

def prepare_dataframe(csv_data):
    spark = SparkSession.builder.appName("ChurnPredictorMLPipeline").getOrCreate()
    dt = spark.sparkContext.parallelize(csv_data)
    df = spark.read.csv(dt, header=True, inferSchema=True, sep=",", nanValue=" ", nullValue=" ")
    return df

# do the prediction with a csv file
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'data_csv' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['data_csv']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        csv_content = file.read()
        csv_ut8_encoded = csv_content.decode('utf-8')
        csv_data = csv_ut8_encoded.splitlines()

        df = prepare_dataframe(csv_data)
        model_type = request.form.get('model_type')

        results = predict_churn(df, model_type)
        return results, 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
