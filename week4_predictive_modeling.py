"""
Week 4: Predictive Modeling and Optimization in Logistics Systems
Dataset: logistics_dataset.csv (simulated shipment data)

Run from the folder containing this script and logistics_dataset.csv:
    pip install pandas numpy matplotlib scikit-learn
    python week4_predictive_modeling.py

The script compares a mean baseline, Linear Regression, and a tuned Random Forest.
Hyperparameter tuning uses cross-validation on the training set only. The test set is
held out for final evaluation. Outputs are saved in week4_outputs/.

Important:
- Shipment_ID is excluded because it is an identifier.
- Delivery_Status is excluded to avoid target leakage.
- This dataset is simulated; results should not be generalized to real operations.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "logistics_dataset.csv"
OUTPUT_DIR = "week4_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
target = "Delivery_Time_hours"
features = ["Distance_km", "Shipment_Weight_kg", "Shipment_Volume",
            "Transport_Mode", "Transportation_Cost"]
numeric_features = ["Distance_km", "Shipment_Weight_kg",
                    "Shipment_Volume", "Transportation_Cost"]
categorical_features = ["Transport_Mode"]

print("Dataset shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))
if "Delivery_Status" in df.columns:
    print("Delivery status counts:\\n", df["Delivery_Status"].value_counts())

X, y = df[features], df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features)
    ])

baseline = DummyRegressor(strategy="mean")
linear = Pipeline([
    ("preprocessor", build_preprocessor()),
    ("model", LinearRegression())
])
rf_pipe = Pipeline([
    ("preprocessor", build_preprocessor()),
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
])

param_grid = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_leaf": [2, 4, 8],
    "model__max_features": [1.0, "sqrt"]
}
tuned_rf = GridSearchCV(
    rf_pipe, param_grid=param_grid, scoring="neg_mean_absolute_error",
    cv=cv, n_jobs=-1, refit=True
)

models = {
    "Mean Baseline": baseline,
    "Linear Regression": linear,
    "Tuned Random Forest": tuned_rf
}
results = []
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred
    if name == "Tuned Random Forest":
        cv_mae = -model.best_score_
        params = str(model.best_params_)
    else:
        cv_mae = -cross_val_score(
            model, X_train, y_train, cv=cv,
            scoring="neg_mean_absolute_error", n_jobs=-1
        ).mean()
        params = "strategy=mean" if name == "Mean Baseline" else "LinearRegression"
    results.append({
        "Model": name,
        "Test_MAE_hours": mean_absolute_error(y_test, pred),
        "Test_RMSE_hours": mean_squared_error(y_test, pred) ** 0.5,
        "Test_R2": r2_score(y_test, pred),
        "Training_CV_MAE_hours": cv_mae,
        "Best_Parameters": params
    })

results_df = pd.DataFrame(results).sort_values("Test_MAE_hours")
results_df.to_csv(os.path.join(OUTPUT_DIR, "week4_model_metrics.csv"), index=False)
print("\\nEvaluation results:\\n", results_df.to_string(index=False))
print("\\nTuned Random Forest parameters:", tuned_rf.best_params_)

# Actual vs predicted plot
plt.figure(figsize=(7.5, 5.4))
for name, pred in predictions.items():
    plt.scatter(y_test, pred, alpha=0.65, label=name)
lo = min(y_test.min(), min(p.min() for p in predictions.values()))
hi = max(y_test.max(), max(p.max() for p in predictions.values()))
plt.plot([lo, hi], [lo, hi], "--")
plt.xlabel("Actual delivery time (hours)")
plt.ylabel("Predicted delivery time (hours)")
plt.title("Actual vs Predicted Delivery Time (Test Set)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "week4_actual_vs_predicted.png"), dpi=180)
plt.close()

# Residual analysis for the model selected by training cross-validation MAE
best_name = min(results, key=lambda row: row["Training_CV_MAE_hours"])["Model"]
best_pred = predictions[best_name]
plt.figure(figsize=(7.5, 5.4))
plt.scatter(best_pred, y_test - best_pred, alpha=0.7)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted delivery time (hours)")
plt.ylabel("Residual (actual - predicted, hours)")
plt.title("Residual Analysis: " + best_name)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "week4_residual_analysis.png"), dpi=180)
plt.close()

# Compare MAE and RMSE
plt.figure(figsize=(8, 5.4))
x = np.arange(len(results_df)); width = 0.34
plt.bar(x - width/2, results_df["Test_MAE_hours"], width, label="MAE")
plt.bar(x + width/2, results_df["Test_RMSE_hours"], width, label="RMSE")
plt.xticks(x, results_df["Model"], rotation=10)
plt.ylabel("Error in hours (lower is better)")
plt.title("Model Comparison on Hold-Out Test Set")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "week4_model_comparison.png"), dpi=180)
plt.close()

# Random Forest feature importance
rf = tuned_rf.best_estimator_
feature_names = rf.named_steps["preprocessor"].get_feature_names_out()
importances = rf.named_steps["model"].feature_importances_
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
importance_df = importance_df.sort_values("Importance", ascending=False)
importance_df.to_csv(os.path.join(OUTPUT_DIR, "week4_feature_importance.csv"), index=False)
plt.figure(figsize=(7.5, 5.4))
plt.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1])
plt.xlabel("Feature importance")
plt.title("Tuned Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "week4_feature_importance.png"), dpi=180)
plt.close()

print("\\nFiles saved in:", OUTPUT_DIR)
print("Remember: weak scores are valid findings. Do not claim production readiness.")
