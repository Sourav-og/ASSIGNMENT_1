"""
Churn prediction for Turkish e-commerce marketplace
Implements:
- Section 1: Churn definition, snapshot, target construction
- Section 2: Feature engineering + preprocessing pipeline
- Section 3: Two model families + tuning using PR-AUC (average_precision)
- Section 4: Threshold selection + subgroup analysis (Device_Type, City)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

RANDOM_STATE = 42
HORIZON_DAYS = 60  # churn horizon: 60 days without purchase => churn


def load_data(path: str) -> pd.DataFrame:
    """
    Load marketplace transaction data and parse date column.
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def build_customer_features_and_labels(df: pd.DataFrame,
                                       horizon_days: int = HORIZON_DAYS
                                       ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build one snapshot per customer:
    - Use all orders up to a global cutoff date (no future leakage).
    - Define churn as: no purchase within `horizon_days` after last observed order.
    Returns:
        X: feature matrix (per customer)
        y: churn labels (1 = churn, 0 = non-churn)
        meta: metadata for subgroup analysis (Device_Type, City)
    """

    # ----- 1) Define cutoff date to avoid looking too far into the future -----
    # Use the maximum date in the dataset as reference
    ref_date = df["Date"].max()
    cutoff_date = ref_date - pd.Timedelta(days=horizon_days)
    print(f"Reference date in data: {ref_date}")
    print(f"Cutoff date for feature construction: {cutoff_date}")

    # Only use data up to cutoff for feature engineering (no leakage)
    df_hist = df[df["Date"] <= cutoff_date].copy()

    # If some customers have no orders before cutoff, they are ignored
    # because we don't have enough history for them.

    # ----- 2) Aggregate features per customer (RFM + behavioral) -----
    # Numeric aggregates
    numeric_aggs = df_hist.groupby("Customer_ID").agg(
        orders_count=("Order_ID", "nunique"),
        total_spend=("Total_Amount", "sum"),
        avg_order_value=("Total_Amount", "mean"),
        total_quantity=("Quantity", "sum"),
        avg_quantity=("Quantity", "mean"),
        avg_discount=("Discount_Amount", "mean"),
        avg_session_duration=("Session_Duration_Minutes", "mean"),
        avg_pages_viewed=("Pages_Viewed", "mean"),
        avg_delivery_time=("Delivery_Time_Days", "mean"),
        avg_rating=("Customer_Rating", "mean"),
        avg_unit_price=("Unit_Price", "mean"),
    )

    # Helper to compute mode safely
    def safe_mode(x):
        m = x.mode()
        return m.iloc[0] if not m.empty else np.nan

    # Demographic & preference aggregates (categorical)
    demo_aggs = df_hist.groupby("Customer_ID").agg(
        age=("Age", "median"),
        gender=("Gender", safe_mode),
        city=("City", safe_mode),
        preferred_category=("Product_Category", safe_mode),
        preferred_payment=("Payment_Method", safe_mode),
        preferred_device=("Device_Type", safe_mode),
        is_returning=("Is_Returning_Customer", safe_mode),
    )

    # Last order date (for recency & label construction)
    last_order_date = df_hist.groupby("Customer_ID")["Date"].max().to_frame("last_order_date")

    # Combine features
    features = numeric_aggs.join(demo_aggs).join(last_order_date)

    # Recency: days since last order until cutoff date
    features["recency_days"] = (cutoff_date - features["last_order_date"]).dt.days

    # ----- 3) Construct churn label (no purchase within horizon_days after last order) -----
    # We now look at *all* orders, but only to see if there is a future order
    # within the horizon after the last observed order for each customer.

    last_order = features["last_order_date"]

    df_future = df.merge(
        last_order.rename("last_order_date"),
        on="Customer_ID",
        how="inner"
    )

    horizon = pd.Timedelta(days=horizon_days)

    # Orders that fall in the churn look-ahead window after last order
    in_future_window = (df_future["Date"] > df_future["last_order_date"]) & \
                       (df_future["Date"] <= df_future["last_order_date"] + horizon)

    future_orders = df_future[in_future_window]

    # Customers with at least one order in the future window => non-churn (0)
    has_future_order = future_orders.groupby("Customer_ID").size().rename("has_future_order")

    features = features.join(has_future_order, how="left")
    features["has_future_order"] = features["has_future_order"].fillna(0).astype(int)

    # Churn label: 1 = no future order within horizon
    features["churn"] = (features["has_future_order"] == 0).astype(int)

    # ----- 4) Prepare X, y, meta -----
    # Keep Device_Type & City for subgroup analysis (use aggregated values)
    meta = features[["preferred_device", "city"]].rename(
        columns={"preferred_device": "Device_Type", "city": "City"}
    )

    # Drop helper columns from X
    X = features.drop(columns=["churn", "last_order_date", "has_future_order"])
    y = features["churn"]

    print("Class distribution (1=churn, 0=non-churn):")
    print(y.value_counts(normalize=True))

    return X, y, meta


def build_preprocessing_pipeline(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Build a preprocessing pipeline:
    - Impute missing values
    - Scale numeric features
    - One-hot encode categorical features
    """

    # Identify categorical and numeric features
    categorical_features = [
        "gender",
        "city",
        "preferred_category",
        "preferred_payment",
        "preferred_device",
        "is_returning",
    ]

    # Make sure only existing columns are used
    categorical_features = [c for c in categorical_features if c in X.columns]
    numeric_features = [c for c in X.columns if c not in categorical_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def tune_models(X_train, y_train, preprocessor):
    """
    Train and tune two model families using PR-AUC (average_precision):
    - Model 1: Logistic Regression
    - Model 2: Random Forest
    Returns:
        best_model_name, best_model, results_dict
    """

    # ----- Model family 1: Logistic Regression -----
    log_reg_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )),
    ])

    log_reg_param_grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__penalty": ["l2"],       # using lbfgs, so only l2
        "model__solver": ["lbfgs"],
    }

    log_reg_grid = GridSearchCV(
        estimator=log_reg_pipeline,
        param_grid=log_reg_param_grid,
        scoring="average_precision",   # PR-AUC
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    print("\nTuning Logistic Regression...")
    log_reg_grid.fit(X_train, y_train)
    print(f"Best PR-AUC (Logistic Regression): {log_reg_grid.best_score_:.4f}")
    print("Best params (Logistic Regression):", log_reg_grid.best_params_)

    # ----- Model family 2: Random Forest -----
    rf_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    rf_param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [5, 10, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }

    rf_grid = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=rf_param_grid,
        scoring="average_precision",   # PR-AUC
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    print("\nTuning Random Forest...")
    rf_grid.fit(X_train, y_train)
    print(f"Best PR-AUC (Random Forest): {rf_grid.best_score_:.4f}")
    print("Best params (Random Forest):", rf_grid.best_params_)

    # ----- Choose best model family -----
    results = {
        "logistic_regression": {
            "best_score": log_reg_grid.best_score_,
            "best_estimator": log_reg_grid.best_estimator_,
        },
        "random_forest": {
            "best_score": rf_grid.best_score_,
            "best_estimator": rf_grid.best_estimator_,
        },
    }

    if rf_grid.best_score_ > log_reg_grid.best_score_:
        best_name = "random_forest"
        best_model = rf_grid.best_estimator_
    else:
        best_name = "logistic_regression"
        best_model = log_reg_grid.best_estimator_

    print(f"\nBest model family: {best_name} "
          f"with PR-AUC = {results[best_name]['best_score']:.4f}")

    return best_name, best_model, results


def choose_threshold(y_true, y_scores):
    """
    Choose a decision threshold based on F1 score
    (you can later override with a business-specific threshold if needed).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # Avoid division by zero in F1 computation
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    # thresholds has length = len(precision) - 1
    best_idx = np.nanargmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"Chosen threshold (max F1): {best_threshold:.4f}, F1 = {best_f1:.4f}")
    return best_threshold, precision, recall, thresholds


def plot_pr_curve(y_true, y_scores, model_name: str):
    """
    Plot Precision-Recall curve.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    plt.figure()
    plt.step(recall, precision, where='post', label=f"PR curve (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({model_name})")
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate_subgroups(y_true, y_scores, meta: pd.DataFrame, threshold: float):
    """
    Evaluate fairness/subgroup performance:
    - By Device_Type: Mobile, Desktop, Tablet
    - By City: report metrics per city (you can later pick top/bottom 3 as needed)

    Metrics: PR-AUC, Precision, Recall, F1
    """
    # Prepare predicted labels at given threshold
    y_pred = (y_scores >= threshold).astype(int)

    # --- Device_Type subgroup ---
    print("\nSubgroup performance by Device_Type:")
    device_results = []

    for device in meta["Device_Type"].dropna().unique():
        mask = meta["Device_Type"] == device
        if mask.sum() < 10:
            # Too few samples; skip or still print with warning
            print(f"  Device {device}: too few samples ({mask.sum()}) – skipping detailed metrics.")
            continue

        y_true_d = y_true[mask]
        y_scores_d = y_scores[mask]
        y_pred_d = y_pred[mask]

        pr_auc_d = average_precision_score(y_true_d, y_scores_d)
        precision_d = precision_score(y_true_d, y_pred_d, zero_division=0)
        recall_d = recall_score(y_true_d, y_pred_d, zero_division=0)
        f1_d = f1_score(y_true_d, y_pred_d, zero_division=0)

        device_results.append({
            "Device_Type": device,
            "support": mask.sum(),
            "pr_auc": pr_auc_d,
            "precision": precision_d,
            "recall": recall_d,
            "f1": f1_d,
        })

    device_df = pd.DataFrame(device_results).sort_values("pr_auc", ascending=False)
    print(device_df.to_string(index=False))

    # --- City subgroup ---
    print("\nSubgroup performance by City:")
    city_results = []

    for city in meta["City"].dropna().unique():
        mask = meta["City"] == city
        if mask.sum() < 10:
            continue

        y_true_c = y_true[mask]
        y_scores_c = y_scores[mask]
        y_pred_c = y_pred[mask]

        pr_auc_c = average_precision_score(y_true_c, y_scores_c)
        precision_c = precision_score(y_true_c, y_pred_c, zero_division=0)
        recall_c = recall_score(y_true_c, y_pred_c, zero_division=0)
        f1_c = f1_score(y_true_c, y_pred_c, zero_division=0)

        city_results.append({
            "City": city,
            "support": mask.sum(),
            "pr_auc": pr_auc_c,
            "precision": precision_c,
            "recall": recall_c,
            "f1": f1_c,
        })

    city_df = pd.DataFrame(city_results).sort_values("pr_auc", ascending=False)
    print("\nAll cities (sorted by PR-AUC):")
    print(city_df.to_string(index=False))

    if not city_df.empty:
        print("\nTop 3 cities by PR-AUC:")
        print(city_df.head(3).to_string(index=False))
        print("\nBottom 3 cities by PR-AUC:")
        print(city_df.tail(3).to_string(index=False))


def main():
    # ========= Section 1 & 2: Load data, define churn, feature engineering =========
    data_path = "marketplace_transactions.csv"   # adjust path if needed
    df = load_data(data_path)

    X, y, meta = build_customer_features_and_labels(df, horizon_days=HORIZON_DAYS)

    preprocessor, num_cols, cat_cols = build_preprocessing_pipeline(X)
    print("\nNumeric features:", num_cols)
    print("Categorical features:", cat_cols)

    # Train/test split (stratified to respect churn imbalance)
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # ========= Section 3: Model training & tuning (PR-AUC) =========
    best_name, best_model, results = tune_models(X_train, y_train, preprocessor)

    # Evaluate on test set using best model
    y_scores_test = best_model.predict_proba(X_test)[:, 1]
    test_pr_auc = average_precision_score(y_test, y_scores_test)
    print(f"\nTest PR-AUC ({best_name}): {test_pr_auc:.4f}")

    # ========= Section 4: Thresholding & PR curve =========
    plot_pr_curve(y_test, y_scores_test, best_name)

    # Choose threshold (here: max F1; you can fix to 0.35 or something else if needed)
    threshold, precision_arr, recall_arr, thresholds = choose_threshold(y_test, y_scores_test)

    y_pred_test = (y_scores_test >= threshold).astype(int)
    print("\nClassification report on test set:")
    print(classification_report(y_test, y_pred_test, digits=4))

    # ========= Subgroup analysis: Device_Type & City =========
    evaluate_subgroups(y_test.reset_index(drop=True),
                       y_scores_test,
                       meta_test.reset_index(drop=True),
                       threshold=threshold)


if __name__ == "__main__":
    main()
