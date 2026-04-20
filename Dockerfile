FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train model and generate all outputs during build
RUN python ChurnPrediction.py

EXPOSE 5000

CMD ["python", "app.py"]
