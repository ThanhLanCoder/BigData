from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
import os

# --- 1. Init Spark ---
spark = SparkSession.builder \
    .appName("RF_Testing_Report") \
    .master("local[*]") \
    .getOrCreate()

# --- 2. Load model ---
model = PipelineModel.load("D:/PhanTichNDSpark/spark_model/sales_rf_model")

# --- 3. Load processed data ---
df = spark.read.csv(
    "D:/PhanTichNDSpark/data/dataRow_for_training_csv",
    header=True,
    inferSchema=True
)

# --- 4. Load FIXED 20% test ---
fixed_test_df = spark.read.parquet(
    "D:/PhanTichNDSpark/data/test_fixed"
)

# --- 5. Evaluators ---
evaluators = {
    "accuracy": MulticlassClassificationEvaluator(
        labelCol="sales", predictionCol="prediction", metricName="accuracy"
    ),
    "precision": MulticlassClassificationEvaluator(
        labelCol="sales", predictionCol="prediction", metricName="weightedPrecision"
    ),
    "recall": MulticlassClassificationEvaluator(
        labelCol="sales", predictionCol="prediction", metricName="weightedRecall"
    ),
    "f1": MulticlassClassificationEvaluator(
        labelCol="sales", predictionCol="prediction", metricName="f1"
    )
}

# =========================
# FUNCTION: add mean & std rows
# =========================
def add_mean_std(df, test_type):
    mean_row = {"run": "mean", "type": test_type}
    std_row = {"run": "std", "type": test_type}

    for col in ["accuracy", "precision", "recall", "f1"]:
        mean_row[col] = df[col].mean()
        std_row[col] = df[col].std()

    return pd.concat(
        [df, pd.DataFrame([mean_row, std_row])],
        ignore_index=True
    )

# =========================
# PART 1 – FIXED 20%
# =========================
fixed_results = []

for run in range(1, 11):
    pred = model.transform(fixed_test_df)
    row = {"run": run, "type": "fixed_20"}
    for name, eval in evaluators.items():
        row[name] = eval.evaluate(pred)
    fixed_results.append(row)

fixed_df = pd.DataFrame(fixed_results)
fixed_df = add_mean_std(fixed_df, "fixed_20")

# =========================
# PART 2 – RANDOM 20%
# =========================
random_results = []

for run in range(1, 11):
    _, test_df = df.randomSplit([0.8, 0.2], seed=run)
    pred = model.transform(test_df)

    row = {"run": run, "type": "random_20"}
    for name, eval in evaluators.items():
        row[name] = eval.evaluate(pred)
    random_results.append(row)

random_df = pd.DataFrame(random_results)
random_df = add_mean_std(random_df, "random_20")

# =========================
# SAVE CSV
# =========================
final_df = pd.concat([fixed_df, random_df], ignore_index=True)

report_dir = "D:/PhanTichNDSpark/report"
os.makedirs(report_dir, exist_ok=True)

final_df.to_csv(f"{report_dir}/rf_test_full_report.csv", index=False)

print("✅ Test completed")
print("📄 Saved: rf_test_full_report.csv")

spark.stop()
