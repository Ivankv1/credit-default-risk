from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from src.preprocessing import preprocess
from src.feature_engineering import create_features
from src.config import MODELS_DIR

def prepare_data(train, bureau=None, previous=None):
    """
    Полная подготовка данных перед обучением.
    """

    train = preprocess(train)

    train = create_features(
        train,
        bureau=bureau,
        previous=previous
    )

    return train

def split_features_target(df):

    X = df.drop(columns=["TARGET"])

    y = df["TARGET"]

    return X, y

def make_train_test_split(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

from lightgbm import LGBMClassifier


def train_model(X_train, y_train, params=None):

    if params is None:

        params = {
            "random_state": 42
        }

    model = LGBMClassifier(**params)

    model.fit(X_train, y_train)

    return model

def evaluate(model, X_test, y_test):

    probabilities = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, probabilities)

    print(f"ROC-AUC: {roc:.4f}")

    return roc

def save_model(model):

    MODELS_DIR.mkdir(exist_ok=True)

    joblib.dump(
        model,
        MODELS_DIR / "lightgbm.pkl"
    )

def train_pipeline(train,
                   bureau=None,
                   previous=None):

    train = prepare_data(
        train,
        bureau,
        previous
    )

    X, y = split_features_target(train)

    X_train, X_test, y_train, y_test = make_train_test_split(
        X,
        y
    )

    model = train_model(
        X_train,
        y_train
    )

    evaluate(
        model,
        X_test,
        y_test
    )

    save_model(model)

    return model