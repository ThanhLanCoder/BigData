# data_processing_for_csv.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import trim, lower, regexp_replace, col, when, udf
from pyspark.sql.types import DoubleType, ArrayType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.linalg import DenseVector
import shutil
import os

# --- 1. Init Spark ---
spark = SparkSession.builder \
    .appName("DataProcessing_For_CSV") \
    .getOrCreate()

# --- 2. Load CSV ---
data_path = "/PhanTichNDSpark/data/dataRowOld.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# --- 3. Rename column nếu bị sai ---
if 'accessed_Ffom' in df.columns:
    df = df.withColumnRenamed('accessed_Ffom', 'accessed_from')

# --- 4. Drop các cột không dùng ---
drop_cols = ['network_protocol', 'accessed_date', 'ip', 'country', 'language', 'returned', 'returned_amount']
df = df.drop(*[c for c in drop_cols if c in df.columns])

# --- 5. Clean accessed_from ---
df = df.withColumn('accessed_from', trim(col('accessed_from')))
df = df.withColumn('accessed_from', regexp_replace(col('accessed_from'), r'\s+', ' '))
df = df.withColumn('accessed_from', regexp_replace(col('accessed_from'), r'[\r\n\t]', ''))
df = df.withColumn('accessed_from', lower(col('accessed_from')))
df = df.withColumn('accessed_from', when(col('accessed_from') == 'saffri', 'safari').otherwise(col('accessed_from')))

# --- 6. Cast numeric columns ---
numeric_cols = ['duration_(secs)', 'bytes', 'age', 'sales']
for c in numeric_cols:
    if c in df.columns:
        df = df.withColumn(c, col(c).cast('double'))

# --- 7. Define label ---
df = df.withColumn('sales', (col('sales') > 0).cast('integer'))

# --- 8. Categorical columns ---
categorical_cols = ['accessed_from', 'gender', 'membership', 'pay_method']

# --- 9. Preprocessing pipeline ---
# StringIndexer cho tất cả categorical
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid='keep') for c in categorical_cols]

# OneHotEncoder cho high-cardinality categorical
onehot_cols = ['accessed_from', 'pay_method']
encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in onehot_cols]

# VectorAssembler numeric + OHE index columns
assembler_input = numeric_cols[:2] + [f"{c}_ohe" for c in onehot_cols] + [f"{c}_idx" for c in ['gender', 'membership']]
assembler = VectorAssembler(inputCols=assembler_input, outputCol='features_raw')

# StandardScaler để chuẩn hóa numeric + vector OHE
scaler = StandardScaler(inputCol='features_raw', outputCol='features', withMean=True, withStd=True)

# Full pipeline
preprocess_pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])

# --- 10. Fit và transform ---
preprocess_model = preprocess_pipeline.fit(df)
df_transformed = preprocess_model.transform(df)

# --- 11. Chuyển cột Vector (OHE) thành array để lưu CSV ---
vector_to_array = udf(lambda v: v.toArray().tolist() if v else [], ArrayType(DoubleType()))

for c in onehot_cols:
    df_transformed = df_transformed.withColumn(f"{c}_ohe_arr", vector_to_array(f"{c}_ohe"))

# --- 12. Split array thành nhiều cột ---
from pyspark.sql.functions import col
def explode_array(df, array_col, prefix):
    # Lấy số phần tử
    n = len(df.select(array_col).first()[0])
    for i in range(n):
        df = df.withColumn(f"{prefix}_{i}", col(array_col)[i])
    return df

for c in onehot_cols:
    df_transformed = explode_array(df_transformed, f"{c}_ohe_arr", c)

# --- 13. Select final columns để CSV ---
final_cols = numeric_cols[:2] + \
             [f"{c}_idx" for c in categorical_cols] + \
             [f"{c}_{i}" for c in onehot_cols for i in range(len(df_transformed.select(f"{c}_ohe_arr").first()[0]))] + \
             ['sales']

df_out = df_transformed.select(final_cols)

# --- 14. Save CSV ---
output_csv_path = "/PhanTichNDSpark/data/dataRow_for_training_csv"
df_out.coalesce(1).write.csv(output_csv_path, header=True, mode='overwrite')

# --- 15. Save pipeline ---
pipeline_save_path = "/PhanTichNDSpark/data/spark_preprocess_pipeline_csv"
if os.path.exists(pipeline_save_path):
    shutil.rmtree(pipeline_save_path)
preprocess_model.write().overwrite().save(pipeline_save_path)

print(f"Processed CSV saved at {output_csv_path}")
print(f"Preprocessing pipeline saved at {pipeline_save_path}")

spark.stop()
