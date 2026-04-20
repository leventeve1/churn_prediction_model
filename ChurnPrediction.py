import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load dataset
# -----------------------------

input_file = "Streaming.csv"
df = pd.read_csv(input_file)

customer_ids = df["Customer_ID"]

# -----------------------------
# 2. Create dimension and fact tables (star schema)
# -----------------------------

dim_customer = df[["Customer_ID", "Age", "Gender", "Region"]].copy()
fact_payment = df[["Customer_ID", "Payment_Method", "Monthly_Spend", "Discount_Offered"]].copy()
fact_usage = df[["Customer_ID", "Subscription_Length", "Last_Activity",
                  "Support_Tickets_Raised", "Satisfaction_Score"]].copy()

dim_customer.to_csv("dim_customer.csv", index=False)
fact_payment.to_csv("fact_payment.csv", index=False)
fact_usage.to_csv("fact_usage.csv", index=False)

print("Dimension and fact tables saved.")

# -----------------------------
# 3. Data preparation
# -----------------------------

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Satisfaction_Score"] = df["Satisfaction_Score"].fillna(df["Satisfaction_Score"].median())

df_model = df.drop("Customer_ID", axis=1)

categorical_cols = ["Gender", "Region", "Payment_Method"]
df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

# -----------------------------
# 4. Split features and target
# -----------------------------

X = df_model.drop("Churned", axis=1)
y = df_model["Churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Feature scaling
# -----------------------------

scale_cols = [
    "Age",
    "Subscription_Length",
    "Support_Tickets_Raised",
    "Satisfaction_Score",
    "Discount_Offered",
    "Last_Activity",
    "Monthly_Spend"
]

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

# -----------------------------
# 6. Train multiple models
# -----------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = {}

for name, model in models.items():

    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    results[name] = acc

# -----------------------------
# 7. Select best model
# -----------------------------

best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]
best_model = models[best_model_name]

print("Model results:")
for k, v in results.items():
    print(k, "accuracy:", round(v, 4))

print("\nBest model:", best_model_name)

# -----------------------------
# 8. Predict churn probability
# -----------------------------

if best_model_name == "Logistic Regression":

    best_model.fit(X_train_scaled, y_train)

    X_full = X.copy()
    X_full[scale_cols] = scaler.transform(X_full[scale_cols])

    churn_prob = best_model.predict_proba(X_full)[:, 1]

else:

    best_model.fit(X_train, y_train)
    churn_prob = best_model.predict_proba(X)[:, 1]

# -----------------------------
# 9. Feature importance
# -----------------------------

if best_model_name == "Logistic Regression":
    importance = abs(best_model.coef_[0])
else:
    importance = best_model.feature_importances_

importance_series = pd.Series(importance, index=X.columns)

importance_percent = (importance_series / importance_series.sum()) * 100

top_factors = importance_percent.sort_values(ascending=False).head(7)

# -----------------------------
# 10. Generate per-customer clues
# -----------------------------

numerical_features = scale_cols
churn_means = df[df["Churned"] == 1][numerical_features].mean()
no_churn_means = df[df["Churned"] == 0][numerical_features].mean()


def get_clues(row, top_n=3):
    """Identify top features pushing this customer toward the churner profile."""
    clues = []
    for feat in numerical_features:
        val = row[feat]
        churn_dist = abs(val - churn_means[feat])
        no_churn_dist = abs(val - no_churn_means[feat])
        if churn_dist < no_churn_dist:
            weight = importance_percent.get(feat, 0)
            clues.append((feat, val, churn_means[feat], weight))

    clues.sort(key=lambda x: x[3], reverse=True)

    result = []
    for feat, val, churn_avg, _ in clues[:top_n]:
        result.append(f"{feat}={round(val, 1)} (churner avg: {round(churn_avg, 1)})")

    if not result:
        result.append("No strong churn indicators")

    return "; ".join(result)


clues_list = df.apply(get_clues, axis=1)

# -----------------------------
# 11. Create output table
# -----------------------------

output = pd.DataFrame({
    "Customer_ID": customer_ids,
    "Churn_Probability": churn_prob,
    "Clues": clues_list
})

output = output.sort_values(by="Churn_Probability", ascending=False)

# -----------------------------
# 12. Save output as CSV
# -----------------------------

output_file = "OUTPUT_PREDICTION.csv"

with open(output_file, "w") as f:

    f.write("Best Model Used," + best_model_name + "\n")
    f.write("Model Accuracy," + str(round(best_accuracy, 4)) + "\n\n")

    f.write("Top Factors Indicating Churn (importance %):\n")

    for factor, value in top_factors.items():
        f.write(f"{factor},{round(value,2)}%\n")

    f.write("\n")

output.to_csv(output_file, mode="a", index=False)

# -----------------------------
# 13. Save output as JSON
# -----------------------------

json_output = {
    "model_info": {
        "best_model": best_model_name,
        "accuracy": round(best_accuracy, 4),
        "top_factors": {k: round(v, 2) for k, v in top_factors.items()}
    },
    "predictions": [
        {
            "customer_id": row["Customer_ID"],
            "churn_probability": round(float(row["Churn_Probability"]), 6),
            "clues": row["Clues"]
        }
        for _, row in output.iterrows()
    ]
}

with open("OUTPUT_PREDICTION.json", "w") as f:
    json.dump(json_output, f, indent=2, default=str)

# -----------------------------
# 14. Save output as XML
# -----------------------------

root = ET.Element("ChurnPredictions")

model_info_el = ET.SubElement(root, "ModelInfo")
ET.SubElement(model_info_el, "BestModel").text = best_model_name
ET.SubElement(model_info_el, "Accuracy").text = str(round(best_accuracy, 4))

factors_el = ET.SubElement(model_info_el, "TopFactors")
for factor, value in top_factors.items():
    factor_el = ET.SubElement(factors_el, "Factor")
    factor_el.set("name", factor)
    factor_el.text = str(round(value, 2))

predictions_el = ET.SubElement(root, "Predictions")
for _, row in output.iterrows():
    pred = ET.SubElement(predictions_el, "Customer")
    ET.SubElement(pred, "CustomerID").text = str(row["Customer_ID"])
    ET.SubElement(pred, "ChurnProbability").text = str(round(row["Churn_Probability"], 6))
    ET.SubElement(pred, "Clues").text = str(row["Clues"])

tree = ET.ElementTree(root)
ET.indent(tree, space="  ")
tree.write("OUTPUT_PREDICTION.xml", encoding="unicode", xml_declaration=True)

# -----------------------------
# 15. Save model artifacts for API deployment
# -----------------------------

model_meta = {
    "best_model_name": best_model_name,
    "scale_cols": scale_cols,
    "churn_means": churn_means.to_dict(),
    "no_churn_means": no_churn_means.to_dict(),
    "importance_percent": importance_percent.to_dict()
}

joblib.dump(best_model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(list(X.columns), "model_columns.joblib")
joblib.dump(model_meta, "model_meta.joblib")

print("\nPrediction completed.")
print("Results saved to OUTPUT_PREDICTION.csv, .json, and .xml")
print("Fact tables saved: dim_customer.csv, fact_payment.csv, fact_usage.csv")
print("Model artifacts saved for deployment.")
