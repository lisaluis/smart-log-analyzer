import os, pickle, json, time, numpy as np
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model  = pickle.load(open(os.path.join(MODELS_DIR, "model.pkl"),  "rb"))
scaler = pickle.load(open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb"))

REQUEST_COUNT   = Counter("prediction_requests_total",   "Total prediction requests")
FAILURE_COUNT   = Counter("failure_predictions_total",   "Total failure predictions")
REQUEST_LATENCY = Histogram("prediction_latency_seconds","Prediction latency")

@app.route("/")
def home():
    return jsonify({
        "service": "Smart Log Analyzer",
        "version": "1.0.0",
        "status":  "running",
        "endpoints": {
            "POST /predict": "Predict machine failure",
            "GET  /health":  "Health check",
            "GET  /metrics": "Prometheus metrics",
            "GET  /info":    "Model information"
        }
    })

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()
    REQUEST_COUNT.inc()
    try:
        data = request.get_json(force=True)
        features = np.array([[
            float(data.get("air_temp", data.get("Air temperature [K]"))),
            float(data.get("process_temp", data.get("Process temperature [K]"))),
            float(data.get("rotational_speed", data.get("Rotational speed [rpm]"))),
            float(data.get("torque", data.get("Torque [Nm]"))),
            float(data.get("tool_wear", data.get("Tool wear [min]"))),
        ]])
        scaled  = scaler.transform(features)
        pred    = int(model.predict(scaled)[0])
        prob    = float(model.predict_proba(scaled)[0][1])
        if pred == 1:
            FAILURE_COUNT.inc()
        REQUEST_LATENCY.observe(time.time() - start)
        return jsonify({
            "prediction":          pred,
            "status":              "FAILURE" if pred == 1 else "NORMAL",
            "failure_probability": round(prob, 4),
            "latency_seconds":     round(time.time() - start, 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

@app.route("/info")
def info():
    eval_path = os.path.join(MODELS_DIR, "evaluation.json")
    eval_data = json.load(open(eval_path)) if os.path.exists(eval_path) else {}
    return jsonify({"model_type": type(model).__name__, "metrics": eval_data})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)