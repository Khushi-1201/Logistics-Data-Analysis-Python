# Week 4: Predictive Modeling and Optimization in Logistics Systems

## Objective
Predict `Delivery_Time_hours` from distance, shipment weight, shipment volume, transport mode, and transportation cost.

## Models and validation
- Mean baseline, Linear Regression, and tuned Random Forest.
- 80/20 train-test split with `random_state=42`.
- Five-fold cross-validation and GridSearchCV performed on training data only.
- Final metrics: MAE, RMSE, and R² on the held-out test set.
- Residual chart corresponds to the model selected by training cross-validation MAE.

## Important interpretation
The dataset is simulated. In this run, the mean baseline slightly outperformed the machine-learning models on the held-out test set. The models show weak predictive value and should not be used for operational decisions without better real-world data and further validation.

## Files
- `week4_predictive_modeling.py` — reproducible modeling workflow
- `logistics_dataset.csv` — simulated dataset used by the script
- `week4_model_metrics.csv` — model metrics and selected parameters
- `week4_feature_importance.csv` — Random Forest feature importance values
- `week4_actual_vs_predicted.png` — held-out test predictions
- `week4_residual_analysis.png` — residual analysis for the CV-selected model
- `week4_model_comparison.png` — test MAE/RMSE comparison
- `week4_feature_importance.png` — feature importance chart
- `Week_4_Predictive_Modeling_Report.docx` — detailed report

## Run
Install dependencies and run from this folder:
```bash
pip install pandas numpy matplotlib scikit-learn
python week4_predictive_modeling.py
```
