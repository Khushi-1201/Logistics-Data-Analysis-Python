"""
Week 4: Predictive Modeling and Optimization in Logistics Systems
Dataset: logistics_dataset.csv (500 simulated shipments)

Run:
    pip install pandas numpy matplotlib scikit-learn
    python week4_predictive_modeling.py

The script trains Linear Regression and Random Forest models to predict
Delivery_Time_hours, evaluates them on a hold-out test set, performs 5-fold
cross-validation, and saves plots plus a model_metrics.csv file.

Important: Shipment_ID is intentionally excluded because it is an identifier,
not a meaningful predictor. Delivery_Status is excluded to avoid target leakage:
it may be determined after delivery and would not be available at prediction time.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "logistics_dataset.csv"
OUTPUT_DIR = "week4_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
target = "Delivery_Time_hours"
features = [
    "Distance_km", "Shipment_Weight_kg", "Shipment_Volume",
    "Transport_Mode", "Transportation_Cost"
]
numeric_features = [
    "Distance_km", "Shipment_Weight_kg", "Shipment_Volume",
    "Transportation_Cost"
]
categorical_features = ["Transport_Mode"]

# Basic data-quality checks
print("Dataset shape:", df.shape)
print("Missing values by column:\\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\\nDelivery status counts:\\n", df["Delivery_Status"].value_counts())

X, y = df[features], df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

models = {
    "Linear Regression": Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", LinearRegression())
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        ))
    ])
}

results = []
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred
    cv_mae = -cross_val_score(
        model, X, y,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_mean_absolute_error"
    ).mean()
    results.append({
        "Model": name,
        "MAE_hours": mean_absolute_error(y_test, pred),
        "RMSE_hours": mean_squared_error(y_test, pred) ** 0.5,
        "R2": r2_score(y_test, pred),
        "CV_MAE_hours": cv_mae
    })

results_df = pd.DataFrame(results).sort_values("MAE_hours")
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"), index=False)
print("\\nModel evaluation results:\\n", results_df.to_string(index=False))

# Actual-versus-predicted plot
plt.figure(figsize=(7, 5))
for name, pred in predictions.items():
    plt.scatter(y_test, pred, alpha=0.65, label=name)
lo = min(y_test.min(), min(p.min() for p in predictions.values()))
hi = max(y_test.max(), max(p.max() for p in predictions.values()))
plt.plot([lo, hi], [lo, hi], "--")
plt.xlabel("Actual delivery time (hours)")
plt.ylabel("Predicted delivery time (hours)")
plt.title("Actual vs Predicted Delivery Time")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=180)
plt.close()

# Residual plot
plt.figure(figsize=(7, 5))
for name, pred in predictions.items():
    plt.scatter(pred, y_test - pred, alpha=0.65, label=name)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted delivery time (hours)")
plt.ylabel("Residual (actual - predicted, hours)")
plt.title("Residual Analysis")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residual_analysis.png"), dpi=180)
plt.close()

# Compare model error metrics
plt.figure(figsize=(7, 5))
x = np.arange(len(results_df))
w = 0.35
plt.bar(x - w/2, results_df["MAE_hours"], w, label="MAE")
plt.bar(x + w/2, results_df["RMSE_hours"], w, label="RMSE")
plt.xticks(x, results_df["Model"], rotation=10)
plt.ylabel("Error in hours (lower is better)")
plt.title("Model Comparison on Hold-Out Test Set")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=180)
plt.close()

# Random Forest feature importance (importance is descriptive, not causal)
rf = models["Random Forest"]
names = rf.named_steps["preprocessor"].get_feature_names_out()
importances = rf.named_steps["model"].feature_importances_
importance_df = pd.DataFrame({
    "Feature": names, "Importance": importances
}).sort_values("Importance", ascending=False)
importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
plt.figure(figsize=(7, 5))
plt.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1])
plt.xlabel("Feature importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=180)
plt.close()

print("\\nOutputs saved in:", OUTPUT_DIR)
print("Interpret results cautiously: this is a small simulated dataset.")
