from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import pickle, numpy as np, time

app = Flask(__name__)

model  = pickle.load(open("models/model.pkl",  "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# ── Prometheus counters ────────────────────────────
REQUEST_COUNT   = Counter('prediction_requests_total', 'Total predictions')
FAILURE_COUNT   = Counter('failure_predictions_total', 'Total failure predictions')
REQUEST_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')

@app.route('/predict', methods=['POST'])
def predict():
    start = time.time()
    REQUEST_COUNT.inc()

    data = request.json
    features = np.array([[
        data['air_temp'], data['process_temp'],
        data['rotational_speed'], data['torque'], data['tool_wear']
    ]])

    scaled = scaler.transform(features)
    pred   = int(model.predict(scaled)[0])
    prob   = float(model.predict_proba(scaled)[0][1])

    if pred == 1:
        FAILURE_COUNT.inc()

    REQUEST_LATENCY.observe(time.time() - start)

    return jsonify({
        "prediction": pred,
        "failure_probability": round(prob, 4),
        "status": "FAILURE" if pred == 1 else "NORMAL"
    })

@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)