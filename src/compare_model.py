from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd

# --- 1. Spark session ---
spark = SparkSession.builder \
    .appName("Model_Comparison") \
    .master("local[*]") \
    .getOrCreate()

# --- 2. Load processed CSV ---
data_path = "D:/PhanTichNDSpark/data/dataRow_csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

label_col = "sales"
feature_cols = [c for c in df.columns if c != label_col]

# --- 3. VectorAssembler ---
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

df = assembler.transform(df).select("features", label_col)

# --- 4. Train / Test split (FIXED) ---
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# --- 5. Models ---
models = {
    "Logistic Regression": LogisticRegression(
        labelCol=label_col,
        featuresCol="features"
    ),
    "Decision Tree": DecisionTreeClassifier(
        labelCol=label_col,
        featuresCol="features",
        maxDepth=10
    ),
    "Random Forest": RandomForestClassifier(
        labelCol=label_col,
        featuresCol="features",
        numTrees=300,
        maxDepth=10,
        seed=42
    )
}

# --- 6. Evaluators ---
metrics = ["accuracy", "weightedPrecision", "weightedRecall", "f1"]
evaluators = {
    m: MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName=m
    )
    for m in metrics
}

# --- 7. Train & Evaluate ---
results = []

for model_name, model in models.items():
    fitted_model = model.fit(train_df)
    preds = fitted_model.transform(test_df)

    row = {"Model": model_name}
    for metric, evaluator in evaluators.items():
        row[metric] = round(evaluator.evaluate(preds), 4)

    results.append(row)

# --- 8. Save result table ---
pdf = pd.DataFrame(results)
output_path = "D:/PhanTichNDSpark/report/model_comparison_results.csv"
pdf.to_csv(output_path, index=False)

print("\n=== MODEL COMPARISON RESULTS ===")
print(pdf)
print(f"\nSaved to {output_path}")

spark.stop()
