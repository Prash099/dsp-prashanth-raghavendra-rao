import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from house_prices.constants import FEATURE_COLUMNS, \
                                   NUMERICAL_FEATURES,\
                                   CATEGORICAL_FEATURES
import joblib


scaler = StandardScaler()
encoder = OrdinalEncoder()
imputer = SimpleImputer(missing_values=np.nan, 
                        strategy='most_frequent')


def feature_scaler(df: pd.DataFrame) -> pd.DataFrame:
    scaler.fit(df)
    df_scaled = scaler.transform(df)
    joblib.dump(scaler, '../models/scaler.joblib')
    return df_scaled

def feature_encoder(df: pd.DataFrame) -> pd.DataFrame:
    encoder.fit(df)
    df_encoded = encoder.transform(df)
    joblib.dump(encoder, '../models/encoder.joblib')
    return df_encoded

def feature_imputer(df: pd.DataFrame) -> pd.DataFrame:
    imputer.fit(df)
    df_imputed = pd.DataFrame(imputer.transform(df), columns=df.columns)
    return df_imputed

def train(X_train: pd.DataFrame) -> pd.DataFrame:
    X_train[NUMERICAL_FEATURES] = feature_scaler(
                                  X_train[NUMERICAL_FEATURES]
                                  )
    X_train[CATEGORICAL_FEATURES] = feature_encoder(
                                    X_train[CATEGORICAL_FEATURES]
                                    )
    return X_train

def test(X_test: pd.DataFrame) -> pd.DataFrame:
    X_test[NUMERICAL_FEATURES] = feature_scaler(
                                 X_test[NUMERICAL_FEATURES]
                                 )
    X_test[CATEGORICAL_FEATURES] = feature_encoder(
                                   X_test[CATEGORICAL_FEATURES]
                                   )
    return X_test

def inference(test_df: pd.DataFrame, scaler: object, encoder: object) -> pd.DataFrame:
    test_df = test_df[FEATURE_COLUMNS]
    test_df = feature_imputer(test_df)
    test_df[NUMERICAL_FEATURES] = scaler.transform(
                                  test_df[NUMERICAL_FEATURES]
                                  )
    test_df[CATEGORICAL_FEATURES] = encoder.transform(
                                    test_df[CATEGORICAL_FEATURES]
                                    )
    return test_df
