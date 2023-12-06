from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql import SparkSession

def evaluate_model(model_path, model, testprepDF):
    try:
        spark = SparkSession.builder.appName("ChurnPredictorMLPipeline").getOrCreate()
        model = model.load(model_path)
        predictions = model.transform(testprepDF)

        evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
        area_under_curve = evaluator.evaluate(predictions)
        print("Area under curve = %g" % area_under_curve)

        sc = spark.sparkContext
        results = predictions.select(['prediction', 'label'])
        results_collect = results.collect()
        results_list = [(float(i[0]), float(i[1])) for i in results_collect]
        predictionAndLabels = sc.parallelize(results_list)

        metrics = BinaryClassificationMetrics(predictionAndLabels)

        # Area under precision-recall curve
        print("Area under PR = %s" % metrics.areaUnderPR)

        # Area under ROC curve
        print("Area under ROC = %s" % metrics.areaUnderROC)

        count = predictions.count()
        correct = results.filter(results.prediction == results.label).count()
        wrong = results.filter(results.prediction != results.label).count()
        tp = results.filter(results.prediction == 1.0).filter(results.prediction == results.label).count()
        fp = results.filter(results.prediction == 1.0).filter(results.prediction != results.label).count()
        fn = results.filter(results.prediction == 0.0).filter(results.prediction != results.label).count()
        tn = results.filter(results.prediction == 0.0).filter(results.prediction == results.label).count()

        accuracy = (tp + tn) / count
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        print("Correct: %s\nWrong: %s\nTP: %s\nFP: %s\nFN: %s\nTN: %s\nAccuracy: %s\nPrecision: %s\nRecall:%s" % (
        correct, wrong, tp, fp, fn, tn, accuracy, precision, recall))
    except Exception as e:
        print(e)