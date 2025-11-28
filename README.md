# ASSIGNMENT_1
Loan Default Prediction – Machine Learning Project

A supervised machine learning project that predicts whether a customer will pay back a loan or default, based on demographic, financial, and loan-related attributes.
This project follows the business requirement of using a single ML model with explainable results, and also includes fairness-based subgroup analysis.

⭐ Project Highlights

✔ Predicts loan default using a single supervised ML model

✔ Uses Logistic Regression for explainability

✔ Full preprocessing pipeline (scaling + encoding + missing value handling)

✔ Computes overall AUC

✔ Computes AUC by education level

✔ Computes AUC by loan purpose

✔ Identifies Top 3 and Bottom 3 loan purposes

✔ Includes a complete solution report for documentation

📁 Repository Structure
📦 Loan-Default-Prediction
 ┣ 📄 loan_default_prediction.py     → Main project code
 ┣ 📄 loan_data.csv                  → Dataset (optional to upload)
 ┣ 📄 Solution_Report.md             → Full project report
 ┣ 📄 README.md                      → Documentation file
 ┗ 📂 images/                        → (Optional) graphs or screenshots

📌 1. Problem Statement

A financial institution wants to predict which customers are most likely to default on loans.
Leadership requires an interpretable, single-model solution, along with subgroup-based fairness analysis for business trust.

Your tasks:

Build a single supervised model

Predict the loan_paid_back outcome

Evaluate using AUC

Compute AUC for:

Education levels

Loan purposes

Identify top and bottom 3 loan purposes

📌 2. Dataset Description

The dataset loan_data.csv includes:

Input features

Customer demographics

Financial variables

Loan amount, purpose, interest rate

Credit score indicators

Payment behavior patterns

Target
loan_paid_back  
1 → Loan paid  
0 → Default

📌 3. Approach / Methodology
✔ Data Preprocessing

Missing value handling

Numeric scaling (StandardScaler)

Categorical encoding (OneHotEncoder)

Pipeline created with ColumnTransformer

✔ Model Used

Logistic Regression (interpretable + business-friendly)

No ensembles or multiple models (as per instruction)

✔ Evaluation Metric

AUC (Area Under ROC Curve)

Subgroup AUC for fairness analysis

📌 4. Key Results
✔ Overall AUC

Replace with your output:

Overall AUC: X.XXXX

✔ Subgroup AUC Results

Education Level AUC Table

Loan Purpose AUC Table

Top 3 Loan Purposes

Bottom 3 Loan Purposes

(Add your actual results here)

📌 5. How to Run the Project
👉 Option 1: Google Colab

Upload loan_data.csv

Upload/run loan_default_prediction.py

View outputs in console

👉 Option 2: Local Machine / VS Code
Install requirements:
pip install pandas numpy scikit-learn

Run the script:
python loan_default_prediction.py


Make sure loan_data.csv is in the same folder, or update the full path in the code.

📌 6. Technologies Used

Python

Pandas

NumPy

Scikit-learn

Jupyter / Google Colab

Logistic Regression

📌 7. Conclusion

The model successfully predicts loan default using a single interpretable ML model.

Overall AUC and subgroup AUC analysis help assess performance fairness across customer groups.

This project demonstrates a complete ML workflow: preprocessing → modeling → evaluation → fairness analysis → reporting.

📌 8. Author

Sourav Adhikari
B.Tech Student
Machine Learning & Data Analytics Enthusiast
