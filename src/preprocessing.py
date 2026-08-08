import numpy as np
import pandas as pd


def fix_days_employed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заменяет аномальное значение DAYS_EMPLOYED.
    """
    df = df.copy()

    df["DAYS_EMPLOYED_ANOM"] = df["DAYS_EMPLOYED"] == 365243
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет ненужные столбцы.
    """
    df = df.copy()

    columns_to_drop = []

    df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполнение пропусков.
    """
    df = df.copy()

    numeric_columns = df.select_dtypes(include="number").columns

    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Полный preprocessing.
    """
    df = fix_days_employed(df)
    df = drop_unused_columns(df)
    df = fill_missing_values(df)

    return df