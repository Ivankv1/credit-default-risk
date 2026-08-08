import numpy as np
import pandas as pd

def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание признаков возраста.
    """
    df = df.copy()

    df["AGE_YEARS"] = (-df["DAYS_BIRTH"]) / 365

    return df

def add_employment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Признаки, связанные со стажем работы.
    """
    df = df.copy()

    df["YEARS_EMPLOYED"] = (-df["DAYS_EMPLOYED"]) / 365

    return df

def add_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Финансовые коэффициенты.
    """
    df = df.copy()

    df["CREDIT_INCOME_RATIO"] = (
        df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    )

    df["ANNUITY_INCOME_RATIO"] = (
        df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    )

    df["CREDIT_ANNUITY_RATIO"] = (
        df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    )

    return df

def merge_bureau(train, bureau):
    """
    Добавление агрегированных признаков bureau.
    """

    bureau_agg = (
        bureau
        .groupby("SK_ID_CURR")
        .agg({
            "AMT_CREDIT_SUM": ["mean", "max"],
            "DAYS_CREDIT": ["min", "max"],
            "SK_ID_BUREAU": "count"
        })
    )

    bureau_agg.columns = [
        "_".join(col)
        for col in bureau_agg.columns
    ]

    bureau_agg.reset_index(inplace=True)

    train = train.merge(
        bureau_agg,
        on="SK_ID_CURR",
        how="left"
    )

    return train

def merge_previous(train, previous):
    """
    Агрегация previous_application.
    """

    previous_agg = (
        previous
        .groupby("SK_ID_CURR")
        .agg({
            "AMT_APPLICATION": "mean",
            "AMT_CREDIT": "mean",
            "SK_ID_PREV": "count"
        })
    )

    previous_agg.columns = [
        "_".join(col)
        for col in previous_agg.columns
    ]

    previous_agg.reset_index(inplace=True)

    train = train.merge(
        previous_agg,
        on="SK_ID_CURR",
        how="left"
    )

    return train

def create_features(train, bureau=None, previous=None):
    """
    Полный Feature Engineering.
    """

    train = add_age_features(train)
    train = add_employment_features(train)
    train = add_financial_ratios(train)

    if bureau is not None:
        train = merge_bureau(train, bureau)

    if previous is not None:
        train = merge_previous(train, previous)

    return train

