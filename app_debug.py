from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Khởi tạo Flask app
app = Flask(__name__)

# Load preprocessor và model
preprocessor = joblib.load('../PhanTichND/data/preprocessor.pkl')
model = joblib.load('../PhanTichND/model/RandomForest.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        transformed = preprocessor.transform(df)
        prediction = model.predict(transformed)[0]
        probability = model.predict_proba(transformed)[0][1]
        return jsonify({
            'prediction': int(prediction),
            'probability': round(float(probability), 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)