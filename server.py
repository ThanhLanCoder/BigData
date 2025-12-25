from flask import Flask, request, jsonify, render_template
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

app = Flask(__name__)

spark = SparkSession.builder \
    .appName("SalesPredictionAPI") \
    .master("local[*]") \
    .getOrCreate()

model = PipelineModel.load(
    "D:/PhanTichNDSpark/spark_model/sales_rf_model"
)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    df = spark.createDataFrame([data])

    preds = model.transform(df)
    row = preds.select("prediction", "probability").first()

    return jsonify({
        "prediction": int(row.prediction),
        "probability": float(row.probability[1])
    })

if __name__ == "__main__":
    app.run(debug=True)
