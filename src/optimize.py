import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score


def objective(trial, X, y):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "random_state": 42,
    }

    model = LGBMClassifier(**params)

    score = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="roc_auc"
    ).mean()

    return score