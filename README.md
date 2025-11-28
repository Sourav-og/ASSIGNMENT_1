# ASSIGNMENT_1
📘 Churn Prediction for E-Commerce Marketplace
📌 Project Overview

This project builds a churn prediction model for a Turkish e-commerce marketplace.
A customer is considered churned if they do not purchase again within 60 days after their last order.

The goal:
✔️ Predict customers likely to churn
✔️ Help marketing teams run targeted retention campaigns
✔️ Use a single interpretable supervised ML model
✔️ Provide explainability & fairness analysis

This project includes the full ML pipeline, PR-AUC evaluation, thresholding, and subgroup analysis across Device Type and City.

🚀 Features

✔️ End-to-end ML pipeline
✔️ Custom churn definition using time horizon
✔️ Feature engineering (RFM + behavioral + demographics)
✔️ Preprocessing with ColumnTransformer
✔️ Two model families (Logistic Regression, Random Forest)
✔️ Hyperparameter tuning via GridSearchCV
✔️ PR-AUC curve visualization
✔️ Threshold selection (Max F1)
✔️ Subgroup fairness evaluation
✔️ GitHub-ready documentation

📊 Churn Definition

A customer is labeled:

1 → Churned
No purchase within 60 days after their last order

0 → Not churned
At least one purchase within the next 60 days

A cutoff date prevents future data leakage.

🛠️ Technologies Used

Python

Pandas

NumPy

Scikit-Learn

Matplotlib

Google Colab / VS Code

📁 Project Structure
churn-prediction/
│
├── churn_model.py
├── SOLUTION_REPORT.md
├── requirements.txt
└── README.md   ← (this file)

🧠 How to Run the Code
Option 1 — Google Colab

Upload marketplace_transactions.csv

Upload churn_model.py

Run the script

View PR curve, results, and subgroup analysis

Option 2 — VS Code

Install dependencies:

pip install -r requirements.txt


Run:

python churn_model.py


Graphs appear in a pop-up Matplotlib window.

📈 Outputs You Will See

The model prints:

Best model family

Best hyperparameters

Validation PR-AUC

Test PR-AUC

Precision–Recall Curve

Selected threshold

Classification report

Device-Type subgroup analysis

City-level subgroup analysis

🧩 Key Insights (Varies by data)

High recency strongly correlates with churn

Lower browsing activity (pages viewed, session duration) increases churn risk

Mobile vs desktop users show different behavioral patterns

Certain cities show lower prediction confidence → need targeted strategies

🔮 Future Improvements

Add SHAP explainability

Add ROC curve + confusion matrix

Deploy model as API

Build dashboard for marketing team

Introduce survival models

📝 Acknowledgment

This project follows a real-world ML workflow focusing on churn prediction in e-commerce with a strong emphasis on explainability and fairness.

📌 2. requirements.txt (upload to repo)
numpy
pandas
scikit-learn
matplotlib

📌 3. Final Folder Structure for GitHub
churn-prediction/
│
├── churn_model.py
├── SOLUTION_REPORT.md
├── README.md
├── requirements.txt
└── marketplace_transactions.csv  (OPTIONAL – usually not uploaded)
