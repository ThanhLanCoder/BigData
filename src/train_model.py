from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, trim, when
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
import shutil, os

#Spark session
spark = SparkSession.builder \
    .appName("SalesRFTraining") \
    .master("local[*]") \
    .getOrCreate()

#Load data
df = spark.read.csv(
    "D:/PhanTichNDSpark/data/datatrain.csv",
    header=True,
    inferSchema=True
)

df = df.withColumnRenamed("duration_(secs)", "duration_secs")


df = df.withColumn(
    "accessed_from",
    lower(trim(col("accessed_from")))
)


df = df.withColumn(
    "sales",
    when(col("sales") > 0, 1).otherwise(0)
)

#Indexers
accessed_indexer = StringIndexer(
    inputCol="accessed_from",
    outputCol="accessed_from_idx",
    handleInvalid="keep"
)

pay_indexer = StringIndexer(
    inputCol="pay_method",
    outputCol="pay_method_idx",
    handleInvalid="keep"
)

gender_indexer = StringIndexer(
    inputCol="gender",
    outputCol="gender_idx",
    handleInvalid="keep"
)

membership_indexer = StringIndexer(
    inputCol="membership",
    outputCol="membership_idx",
    handleInvalid="keep"
)

# OneHotEncoder
encoder = OneHotEncoder(
    inputCols=["accessed_from_idx", "pay_method_idx"],
    outputCols=["accessed_ohe", "pay_ohe"]
)

#Vector Assembler
assembler = VectorAssembler(
    inputCols=[
        "duration_secs",
        "bytes",
        "accessed_ohe",
        "pay_ohe",
        "gender_idx",
        "membership_idx"
    ],
    outputCol="features"
)

#Model
rf = RandomForestClassifier(
    labelCol="sales",
    featuresCol="features",
    numTrees=300,
    maxDepth=10,
    seed=42
)

pipeline = Pipeline(stages=[
    accessed_indexer,
    pay_indexer,
    gender_indexer,
    membership_indexer,
    encoder,
    assembler,
    rf
])


train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

#Train model
model = pipeline.fit(train_df)

#Save model
model_path = "D:/PhanTichNDSpark/spark_model/sales_rf_model"
if os.path.exists(model_path):
    shutil.rmtree(model_path)

model.write().overwrite().save(model_path)

#save test
test_path = "D:/PhanTichNDSpark/data/test_fixed"
if os.path.exists(test_path):
    shutil.rmtree(test_path)

test_df.write.mode("overwrite").parquet(test_path)

print("TRAIN DONE")
print("TEST_FIXED SAVED")

spark.stop()
