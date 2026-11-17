"""Split data and build pipelines."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from ml_lab.utils.reproducibility import DEFAULT_SEED


def split_three_way(X, y, task="classification", seed=DEFAULT_SEED, test_size=0.2, val_size=0.2):
    """Train, validation, test split."""
    stratify = y if task == "classification" else None
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )
    stratify = y_rest if task == "classification" else None
    relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=relative, random_state=seed, stratify=stratify
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def column_types(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(X, scale=True, encode="onehot"):
    """Impute, scale, encode."""
    numeric, categorical = column_types(X)
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    parts = [("num", Pipeline(numeric_steps), numeric)]
    if categorical:
        if encode == "onehot":
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        cat_steps = [("impute", SimpleImputer(strategy="most_frequent")), ("encode", encoder)]
        parts.append(("cat", Pipeline(cat_steps), categorical))
    return ColumnTransformer(parts)


def make_model_pipeline(X, model, scale=True, encode="onehot"):
    """Preprocessor plus model."""
    return Pipeline([("prep", build_preprocessor(X, scale, encode)), ("model", model)])


def feature_names(pipeline):
    """Names after preprocessing."""
    return list(pipeline.named_steps["prep"].get_feature_names_out())
