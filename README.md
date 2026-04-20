# Customer Churn Prediction for a Streaming Service

This project demonstrates a machine learning pipeline to predict customer churn for a fictional video streaming company.

The goal is to identify customers who are likely to cancel their subscription so the company can take targeted retention actions such as discounts, emails, or personalized recommendations.

The script trains multiple machine learning models, selects the best performing one, and outputs churn probabilities with per-customer explanations for each prediction.

---

# Project Idea

Subscription-based businesses often lose customers every month. Predicting which users are likely to churn allows companies to proactively intervene.

This project uses customer data such as age, subscription length, payment method, support tickets, satisfaction score, last activity, and monthly spending to estimate the probability that a customer will churn.

---

# Dataset

Dataset used: Streaming Service Data  
https://www.kaggle.com/datasets/akashanandt/streaming-service-data

Example columns in the dataset:

| Column | Description |
|------|------|
| Customer_ID | Unique identifier |
| Age | Customer age |
| Gender | Male/Female |
| Subscription_Length | Months subscribed |
| Region | Customer region |
| Payment_Method | Payment method used |
| Support_Tickets_Raised | Number of support tickets |
| Satisfaction_Score | Customer satisfaction rating |
| Discount_Offered | Discount offered to the customer |
| Last_Activity | Days since last activity |
| Monthly_Spend | Monthly spending |
| Churned | Target variable (1 = churned) |

---

# Star Schema (Fact and Dimension Tables)

The dataset is split into a star schema for structured analysis:

- **dim_customer.csv** — Customer dimension (Customer_ID, Age, Gender, Region)
- **fact_payment.csv** — Payment fact table (Customer_ID, Payment_Method, Monthly_Spend, Discount_Offered)
- **fact_usage.csv** — Usage fact table (Customer_ID, Subscription_Length, Last_Activity, Support_Tickets_Raised, Satisfaction_Score)

---

# Machine Learning Approach

### 1. Data Preparation
- Missing values are filled using the median.
- Categorical variables are encoded into numerical format.
- Feature scaling is applied for algorithms that require normalized input.

### 2. Train/Test Split
The dataset is split into training and testing sets (80/20) to evaluate model performance.

### 3. Model Training
Multiple models are trained and compared:
- Logistic Regression
- Random Forest
- Gradient Boosting

### 4. Model Selection
The model with the highest accuracy on the test dataset is automatically selected.

### 5. Prediction
The selected model predicts the churn probability for each customer.

### 6. Feature Importance
The system calculates the most important factors influencing churn and their relative importance percentages.

### 7. Per-Customer Clues
For each customer, the system identifies the top 3 features that push them toward the churner profile. Each clue shows the customer's value compared to the average churner value, helping explain why that customer is at risk.

Example clue: `Satisfaction_Score=2.0 (churner avg: 4.5)`

---

# Output

The script generates predictions in three formats:

- **OUTPUT_PREDICTION.csv** — CSV with model info, top factors, and per-customer predictions with clues
- **OUTPUT_PREDICTION.json** — Same data in JSON format
- **OUTPUT_PREDICTION.xml** — Same data in XML format

Example output:

```
Customer_ID,Churn_Probability,Clues
CUST000221,0.9999,Satisfaction_Score=1.0 (churner avg: 4.5); Last_Activity=333 (churner avg: 220.7)
```

---

# Docker Deployment

The model is deployed as a containerized REST API using Flask.

### Build and run the container

```
docker build -t churn-prediction .
docker run -p 5000:5000 churn-prediction
```

### API Endpoints

**POST /predict** — Submit customer data, receive churn probability and clues

## 🔌 Example API Request (POST /predict)

```bash
curl -X POST http://localhost:5000/predict ^
-H "Content-Type: application/json" ^
-d "{\"Customer_ID\":\"CUST999999\",\"Age\":35,\"Gender\":\"Male\",\"Subscription_Length\":12,\"Region\":\"East\",\"Payment_Method\":\"PayPal\",\"Support_Tickets_Raised\":5,\"Satisfaction_Score\":3,\"Discount_Offered\":10.5,\"Last_Activity\":200,\"Monthly_Spend\":45.0}"

**GET /health** — Health check

---

# Scheduling

The batch prediction script can be scheduled to run automatically using **Windows Task Scheduler**:

1. Open Task Scheduler (`taskschd.msc`)
2. Create Basic Task
3. Set a trigger (e.g. daily, weekly)
4. Action: Start a Program
5. Program/script: path to `run_schedule.bat`

Each run is logged to `run_log.txt`.

---

# How to Run Locally

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Place dataset in project folder

```
Streaming.csv
```

### 3. Run the script

```
python ChurnPrediction.py
```

### 4. Output

The script will generate:
- `OUTPUT_PREDICTION.csv`, `.json`, `.xml`
- `dim_customer.csv`, `fact_payment.csv`, `fact_usage.csv`
- Model artifacts (`model.joblib`, `scaler.joblib`, etc.)

---

# Project Structure

```
project/
  Streaming.csv              # Input dataset
  ChurnPrediction.py         # Main prediction script
  app.py                     # Flask API for Docker deployment
  Dockerfile                 # Container definition
  run_schedule.bat           # Windows Task Scheduler script
  requirements.txt           # Python dependencies
  README.md
  dim_customer.csv           # Customer dimension table
  fact_payment.csv           # Payment fact table
  fact_usage.csv             # Usage fact table
  OUTPUT_PREDICTION.csv      # Predictions (CSV)
  OUTPUT_PREDICTION.json     # Predictions (JSON)
  OUTPUT_PREDICTION.xml      # Predictions (XML)
```

---

# Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn (Logistic Regression, Random Forest, Gradient Boosting)
- Flask (REST API)
- Docker (containerization)
- Windows Task Scheduler (scheduling)

---

# Team Members

Levente Selley

Gergo Zalavary

Fabo Leon Tompos
