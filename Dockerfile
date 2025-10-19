FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline/ ./pipeline/
COPY steps/ ./steps/
COPY strategy/ ./strategy/
COPY saved_model/ ./saved_model/
# Single files copied to /app (working dir)
COPY accuracy.py ./accuracy.py
COPY message.py ./message.py
COPY api.py ./api.py
COPY run_pipeline.py ./run_pipeline.py
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
