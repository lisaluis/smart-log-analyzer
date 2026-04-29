FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Train model during build so pkl files exist at runtime
RUN mkdir -p models && \
    python src/preprocess.py && \
    python src/train.py && \
    python src/evaluate.py

EXPOSE 5000

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "--timeout", "120", "app.app:app"]
