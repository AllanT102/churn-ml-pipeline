from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def train_logistic_regression(trainprepDF):
    try:
        lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)
        evaluatorLR = BinaryClassificationEvaluator(rawPredictionCol="prediction")

        paramGrid = (ParamGridBuilder()
                    .addGrid(lr.regParam, [0.01, 0.5, 2.0])
                    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
                    .addGrid(lr.maxIter, [5, 10, 20])
                    .build())

        cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluatorLR, numFolds=5)
        cvModel = cv.fit(trainprepDF)
        model_path = "data/lrModel"
        cvModel.bestModel.save(model_path)
        return model_path
    except Exception as e:
        print(e)
        raise e

def train_random_forest(trainprepDF):
    try:
        rf = RandomForestClassifier(labelCol="label", featuresCol="features").setImpurity("gini").setMaxDepth(6).setNumTrees(50).setFeatureSubsetStrategy("auto").setSeed(1010)
        evaluatorRF = BinaryClassificationEvaluator(rawPredictionCol="prediction")

        paramGrid = (ParamGridBuilder()
                    .addGrid(rf.maxDepth, [2, 4, 6])
                    .addGrid(rf.numTrees, [20, 50, 100])
                    .build())

        cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluatorRF, numFolds=5)
        cvModel = cv.fit(trainprepDF)

        model_path = "data/rfModel"
        cvModel.bestModel.save(model_path)
        return model_path
    except Exception as e:
        print(e)
        raise e