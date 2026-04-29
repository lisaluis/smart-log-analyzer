import os, pickle, json, time, numpy as np
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# ── Load model & scaler at startup ────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model  = pickle.load(open(os.path.join(MODELS_DIR, "model.pkl"),  "rb"))
scaler = pickle.load(open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb"))

FEATURES = ['Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# ── Prometheus metrics ─────────────────────────────────────────────────
REQUEST_COUNT   = Counter("prediction_requests_total",   "Total prediction requests")
FAILURE_COUNT   = Counter("failure_predictions_total",   "Total failure predictions")
NORMAL_COUNT    = Counter("normal_predictions_total",    "Total normal predictions")
REQUEST_LATENCY = Histogram("prediction_latency_seconds","Prediction latency in seconds")
ERROR_COUNT     = Counter("prediction_errors_total",     "Total prediction errors")

# ── Routes ─────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({
        "service":     "Smart Log Analyzer",
        "version":     "1.0.0",
        "description": "ML-powered system failure prediction API",
        "endpoints": {
            "POST /predict": "Predict machine failure",
            "GET  /health":  "Health check",
            "GET  /metrics": "Prometheus metrics",
            "GET  /info":    "Model information"
        }
    })

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route("/info")
def info():
    eval_path = os.path.join(MODELS_DIR, "evaluation.json")
    eval_data = {}
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_data = json.load(f)
    return jsonify({
        "model_type": type(model).__name__,
        "features":   FEATURES,
        "metrics":    eval_data
    })

@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()
    REQUEST_COUNT.inc()
    try:
        data = request.get_json(force=True)

        # Accept both key names (friendly + exact column name)
        def get_val(friendly, exact):
            return data.get(friendly, data.get(exact))

        features = np.array([[
            float(get_val("air_temp",          "Air temperature [K]")),
            float(get_val("process_temp",       "Process temperature [K]")),
            float(get_val("rotational_speed",   "Rotational speed [rpm]")),
            float(get_val("torque",             "Torque [Nm]")),
            float(get_val("tool_wear",          "Tool wear [min]")),
        ]])

        scaled   = scaler.transform(features)
        pred     = int(model.predict(scaled)[0])
        prob     = float(model.predict_proba(scaled)[0][1])
        latency  = round(time.time() - start, 4)

        if pred == 1:
            FAILURE_COUNT.inc()
        else:
            NORMAL_COUNT.inc()
        REQUEST_LATENCY.observe(latency)

        return jsonify({
            "prediction":          pred,
            "status":              "FAILURE" if pred == 1 else "NORMAL",
            "failure_probability": round(prob, 4),
            "normal_probability":  round(1 - prob, 4),
            "latency_seconds":     latency,
            "input": {
                "air_temp":        features[0][0],
                "process_temp":    features[0][1],
                "rotational_speed":features[0][2],
                "torque":          features[0][3],
                "tool_wear":       features[0][4],
            }
        })

    except Exception as e:
        ERROR_COUNT.inc()
        return jsonify({"error": str(e), "hint": "Check field names and types"}), 400

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
