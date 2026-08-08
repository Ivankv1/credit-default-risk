import pandas as pd

from src.config import RAW_DATA_DIR


def load_application_train():
    return pd.read_csv(RAW_DATA_DIR / "application_train.csv")


def load_application_test():
    return pd.read_csv(RAW_DATA_DIR / "application_test.csv")


def load_bureau():
    return pd.read_csv(RAW_DATA_DIR / "bureau.csv")


def load_previous_application():
    return pd.read_csv(RAW_DATA_DIR / "previous_application.csv")


def load_all():
    return {
        "train": load_application_train(),
        "test": load_application_test(),
        "bureau": load_bureau(),
        "previous": load_previous_application(),
    }