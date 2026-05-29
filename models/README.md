# Models

Serialized model artifacts produced by `scripts/train_model.py`. Not committed.

## Naming convention

`{model_type}-v{version}.joblib`

| Example | Description |
|---------|-------------|
| `xgb-v1.0.joblib` | XGBoost classifier |
| `rf-v1.0.joblib` | Random forest |
| `lr-v1.0.joblib` | Logistic regression |

Run `make train` to regenerate from processed data.
