import shap


def create_explainer(model):

    return shap.TreeExplainer(model)


def shap_values(model, X):

    explainer = create_explainer(model)

    return explainer.shap_values(X)


def summary_plot(model, X):

    values = shap_values(model, X)

    shap.summary_plot(values, X)