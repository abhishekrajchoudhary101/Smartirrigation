from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return "XGBoost API Running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # get input
        
        soil_moisture = float(data['soil_moisture'])-60.00
        soil_humidity=float(data['soil_humidity'])-60.00
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])

        # IMPORTANT: same order as training
        features = np.array([[soil_moisture,soil_humidity,temperature, humidity]])

        # scale
        features = scaler.transform(features)

        # predict
        prediction = model.predict(features)

        return jsonify({
            "pump": int(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
