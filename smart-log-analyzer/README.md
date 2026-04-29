# 🔍 Smart Log Analyzer
### System Failure Prediction using Machine Learning & MLOps
**Lisa Luis & Shannon Coelho | MSC AI, Goa University**

---

## 🚀 Live Demo
| Service | URL |
|---------|-----|
| Flask API | https://smart-log-analyzer-n3v3.onrender.com |
| Health Check | https://smart-log-analyzer-n3v3.onrender.com/health |
| Metrics | https://smart-log-analyzer-n3v3.onrender.com/metrics |

---

## 🏗️ Architecture

```
Sensor Data → Flask API → ML Model → Prediction
                ↓               ↑
           Prometheus     scikit-learn
                ↓
            Grafana Dashboard
```

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| ML | scikit-learn (Random Forest, Logistic Regression) |
| API | Flask + Gunicorn |
| UI | Streamlit |
| Experiment Tracking | MLflow |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Deployment | Render |
| Monitoring | Prometheus + Grafana |

## 📦 Setup (Local)

```bash
# 1. Clone
git clone https://github.com/lisaluis/smart-log-analyzer.git
cd smart-log-analyzer

# 2. Add dataset
# Download ai4i2020.csv from Kaggle and put it in data/

# 3. Train
python src/preprocess.py
python src/train.py
python src/evaluate.py

# 4. Run API
python app/app.py

# 5. Run UI
streamlit run app/streamlit_app.py
```

## 🐳 Docker (All Services)

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Flask API | http://localhost:5000 |
| Streamlit UI | http://localhost:8501 |
| MLflow | http://localhost:5001 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

## 📊 Dataset

AI4I 2020 Predictive Maintenance Dataset (Kaggle)
- 10,000 machine readings
- 5 features: Air Temp, Process Temp, Rotational Speed, Torque, Tool Wear
- Target: Machine failure (binary)

## 🔁 CI/CD Pipeline

Every push to `main` triggers:
1. ✅ Data preprocessing
2. ✅ Model training
3. ✅ Model evaluation
4. ✅ Docker image build & push
5. ✅ Auto-deploy to Render
