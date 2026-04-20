from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load model artifacts
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")
model_columns = joblib.load("model_columns.joblib")
meta = joblib.load("model_meta.joblib")

scale_cols = meta["scale_cols"]
churn_means = meta["churn_means"]
no_churn_means = meta["no_churn_means"]
importance_pct = meta["importance_percent"]


def get_clues(row_dict, top_n=3):
    """Identify top features pushing this customer toward the churner profile."""
    clues = []
    for feat in scale_cols:
        if feat not in row_dict:
            continue
        val = float(row_dict[feat])
        churn_dist = abs(val - churn_means[feat])
        no_churn_dist = abs(val - no_churn_means[feat])
        if churn_dist < no_churn_dist:
            weight = importance_pct.get(feat, 0)
            clues.append({
                "feature": feat,
                "value": round(val, 1),
                "churner_avg": round(churn_means[feat], 1),
                "importance": round(weight, 2)
            })

    clues.sort(key=lambda x: x["importance"], reverse=True)
    return clues[:top_n] if clues else [{"feature": "none", "message": "No strong churn indicators"}]


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if isinstance(data, dict):
        data = [data]

    df = pd.DataFrame(data)

    # One-hot encode categoricals
    categorical_cols = ["Gender", "Region", "Payment_Method"]
    df_encoded = pd.get_dummies(
        df.drop("Customer_ID", axis=1, errors="ignore"),
        columns=categorical_cols,
        drop_first=True
    )

    # Align columns with training data
    for col in model_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[model_columns]

    # Scale if logistic regression
    if meta["best_model_name"] == "Logistic Regression":
        df_scaled = df_encoded.copy()
        df_scaled[scale_cols] = scaler.transform(df_encoded[scale_cols])
        probs = model.predict_proba(df_scaled)[:, 1]
    else:
        probs = model.predict_proba(df_encoded)[:, 1]

    # Build response
    results = []
    for i, row in df.iterrows():
        row_dict = row.to_dict()
        results.append({
            "customer_id": row_dict.get("Customer_ID", f"customer_{i}"),
            "churn_probability": round(float(probs[i]), 6),
            "clues": get_clues(row_dict)
        })

    return jsonify({"predictions": results})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": meta["best_model_name"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
