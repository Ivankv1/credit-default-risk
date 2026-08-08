from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_model(model, X_test, y_test):

    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    metrics = {
        "roc_auc": roc_auc_score(y_test, proba),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }

    return metrics