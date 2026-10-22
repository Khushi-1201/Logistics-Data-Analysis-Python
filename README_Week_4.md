# Week 4: Predictive Modeling and Optimization in Logistics Systems

## Objective
Predict `Delivery_Time_hours` using shipment distance, weight, volume, transport mode, and transportation cost. Compare Linear Regression and Random Forest Regression.

## Dataset
- 500 rows and 8 columns
- Missing values: 0
- Duplicate rows: 0
- Target: `Delivery_Time_hours`
- `Shipment_ID` is excluded as an identifier.
- `Delivery_Status` is excluded from predictors to avoid target leakage.

## Model evaluation
See `week4_model_metrics.csv`. Metrics are calculated on a fixed 80/20 train/test split (random_state=42), with 5-fold cross-validation MAE.

## Important interpretation
The tested models show weak test-set R² on this simulated dataset. Do not claim the model is production-ready or promise operational savings. Collect more useful real-world features and validate on historical operational data before deployment.

## Files
- `week4_predictive_modeling.py` — reproducible training/evaluation script
- `week4_model_metrics.csv` — evaluation table
- `week4_actual_vs_predicted.png` — test-set predictions
- `week4_residual_analysis.png` — residual plot
- `week4_model_comparison.png` — MAE/RMSE comparison
- `week4_feature_importance.png` — Random Forest feature importance
- `Week_4_Predictive_Modeling_Report.docx` — detailed report

## Run
Install dependencies:
```bash
pip install pandas numpy matplotlib scikit-learn
```
Place the script and `logistics_dataset.csv` in the same directory, then run:
```bash
python week4_predictive_modeling.py
```
