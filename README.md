# Customer Churn Prediction for a Streaming Service

This project demonstrates a simple machine learning pipeline to predict customer churn for a fictional video streaming company.

The goal is to identify customers who are likely to cancel their subscription so the company can take targeted retention actions such as discounts, emails, or personalized recommendations.

The script trains multiple machine learning models, selects the best performing one, and outputs churn probabilities for each customer.

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

# Machine Learning Approach

The pipeline performs the following steps:

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

---

# Output

After running the script, an output file is generated:

OUTPUT_PREDICTION.csv

The file contains:
1. The best model used
2. Model accuracy
3. Top factors influencing churn
4. Customers sorted by churn probability

Example output:

Best Model Used,Random Forest  
Model Accuracy,0.84  

Top Factors Indicating Churn (importance %):  
Last_Activity,21.3%  
Satisfaction_Score,18.7%  
Subscription_Length,16.5%  

Customer_ID,Churn_Probability  
CUST000241,0.93  
CUST000982,0.91  

This allows a marketing team to quickly identify high-risk customers.

---

# How to Run

### 1 Install dependencies

pip install -r requirements.txt

### 2 Place dataset in project folder

Streaming.csv

### 3 Run the script

python ChurnPrediction.py

### 4 Output

The script will generate:

OUTPUT_PREDICTION.csv

---

# Project Structure

project/

Streaming.csv  
ChurnPrediction.py  
requirements.txt  
README.md  

---

# Technologies Used

Python  
Pandas  
Scikit-learn  
Machine Learning (classification)

---

# Educational Purpose

This project was created as part of a data science / machine learning university assignment demonstrating a complete workflow including data preparation, model training, model comparison, churn probability prediction, and business-oriented output.

---

# Possible Future Improvements

- Add more advanced models (XGBoost, LightGBM)
- Use cross-validation for model selection
- Add explainability tools (SHAP)
- Build a dashboard for churn monitoring
- Deploy the model as an API or web application

# Team members

Levente Sélley

Gergő Zalaváry

Fabó Leon Tompos
