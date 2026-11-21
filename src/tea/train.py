"""Model pipeline and evaluation for MP210 (regression, metric = MAE)."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL = ["flush", "section"]

MODELS = {
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


def build_pipeline(model_name, params):
    """One-hot encode the two text columns, pass the numbers through, then the model."""
    if model_name not in MODELS:
        raise ValueError(f"unknown model: {model_name}")
    encoder = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )
    return Pipeline([("prep", encoder), ("model", MODELS[model_name](**params))])


def evaluate(model, X, y):
    """Mean Absolute Error in kg."""
    return float(mean_absolute_error(y, model.predict(X)))
