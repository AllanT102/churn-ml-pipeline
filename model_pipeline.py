from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml import Pipeline

def process_training_data(file_location):
    try:
        spark = SparkSession.builder.appName("ChurnPredictorMLPipeline").getOrCreate()
        df = spark.read.format("csv") \
            .option("inferSchema", "true") \
            .option("header", "true") \
            .option("sep", ",") \
            .option("nanValue", " ") \
            .option("nullValue", " ") \
            .load(file_location)
        
        (train_data, test_data) = df.randomSplit([0.7, 0.3], 24)

        catColumns = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
                    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                    "Contract", "PaperlessBilling", "PaymentMethod"]

        stages = []

        for catCol in catColumns:
            stringIndexer = StringIndexer(inputCol=catCol, outputCol=catCol + "Index")
            encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[catCol + "catVec"])
            stages += [stringIndexer, encoder]

        imputer = Imputer(inputCols=["TotalCharges"], outputCols=["Out_TotalCharges"])
        stages += [imputer]

        label_Idx = StringIndexer(inputCol="Churn", outputCol="label")
        stages += [label_Idx]

        tenure_bin = QuantileDiscretizer(numBuckets=3, inputCol="tenure", outputCol="tenure_bin")
        stages += [tenure_bin]

        numericCols = ["tenure_bin", "Out_TotalCharges", "MonthlyCharges"]
        assembleInputs = [c + "catVec" for c in catColumns] + numericCols
        assembler = VectorAssembler(inputCols=assembleInputs, outputCol="features")
        stages += [assembler]

        pipeline = Pipeline().setStages(stages)
        pipelineModel = pipeline.fit(train_data)

        trainprepDF = pipelineModel.transform(train_data)
        testprepDF = pipelineModel.transform(test_data)
        return (trainprepDF, testprepDF)
    except Exception as e:
        print(e)

def process_input_data(df):
    try:
        catColumns = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
                    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                    "Contract", "PaperlessBilling", "PaymentMethod"]

        stages = []

        for catCol in catColumns:
            stringIndexer = StringIndexer(inputCol=catCol, outputCol=catCol + "Index")
            encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[catCol + "catVec"])
            stages += [stringIndexer, encoder]

        imputer = Imputer(inputCols=["TotalCharges"], outputCols=["Out_TotalCharges"])
        stages += [imputer]

        tenure_bin = QuantileDiscretizer(numBuckets=3, inputCol="tenure", outputCol="tenure_bin")
        stages += [tenure_bin]

        numericCols = ["tenure_bin", "Out_TotalCharges", "MonthlyCharges"]
        assembleInputs = [c + "catVec" for c in catColumns] + numericCols
        assembler = VectorAssembler(inputCols=assembleInputs, outputCol="features")
        stages += [assembler]

        pipeline = Pipeline().setStages(stages)
        pipelineModel = pipeline.fit(df)

        return pipelineModel.transform(df)
    except Exception as e:
        print(e)