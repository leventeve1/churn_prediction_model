import pandas as pd

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
# 2. Data preparation
# -----------------------------

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Satisfaction_Score"] = df["Satisfaction_Score"].fillna(df["Satisfaction_Score"].median())

df_model = df.drop("Customer_ID", axis=1)

categorical_cols = ["Gender", "Region", "Payment_Method"]
df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

# -----------------------------
# 3. Split features and target
# -----------------------------

X = df_model.drop("Churned", axis=1)
y = df_model["Churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Feature scaling
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
# 5. Train multiple models
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
# 6. Select best model
# -----------------------------

best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]
best_model = models[best_model_name]

print("Model results:")
for k, v in results.items():
    print(k, "accuracy:", round(v, 4))

print("\nBest model:", best_model_name)

# -----------------------------
# 7. Predict churn probability
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
# 8. Feature importance
# -----------------------------

if best_model_name == "Logistic Regression":
    importance = abs(best_model.coef_[0])
else:
    importance = best_model.feature_importances_

importance_series = pd.Series(importance, index=X.columns)

importance_percent = (importance_series / importance_series.sum()) * 100

top_factors = importance_percent.sort_values(ascending=False).head(7)

# -----------------------------
# 9. Create output table
# -----------------------------

output = pd.DataFrame({
    "Customer_ID": customer_ids,
    "Churn_Probability": churn_prob
})

output = output.sort_values(by="Churn_Probability", ascending=False)

# -----------------------------
# 10. Save output file
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

print("\nPrediction completed.")
print("Results saved to OUTPUT_PREDICTION.csv")
