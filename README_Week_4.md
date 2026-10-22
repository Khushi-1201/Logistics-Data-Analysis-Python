# Week 4: Predictive Modeling and Optimization in Logistics Systems

## Objective
Predict `Delivery_Time_hours` using shipment distance, weight, volume, transport mode, and transportation cost.

## Methods
- Mean baseline
- Linear Regression
- Random Forest with training-only GridSearchCV tuning
- 80/20 train-test split and five-fold cross-validation on the training data
- MAE, RMSE, and R² evaluation

## Important result
The dataset is simulated and has limited predictive signal. Report model scores honestly; do not claim production readiness or guaranteed cost savings.

## Files
- `week4_predictive_modeling.py` — full reproducible workflow
- `week4_model_metrics.csv` — metrics and tuned parameters
- `week4_feature_importance.csv` — feature importance values
- `week4_actual_vs_predicted.png` — test predictions
- `week4_residual_analysis.png` — residual analysis for best test-MAE model
- `week4_model_comparison.png` — test MAE/RMSE comparison
- `week4_feature_importance.png` — Random Forest feature importance
- `Week_4_Predictive_Modeling_Report.docx` — detailed report

## Run
```bash
pip install pandas numpy matplotlib scikit-learn
python week4_predictive_modeling.py
```
Keep `logistics_dataset.csv` in the same folder as the Python script.
