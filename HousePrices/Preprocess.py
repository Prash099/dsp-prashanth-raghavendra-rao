import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from HousePrices.Constants import FEATURE_COLUMNS, NUMERICAL_FEATURES,\
                                  CATEGORICAL_FEATURES, TARGET
import joblib


class Preprocess:
    def __init__(self) -> None:
        self.feature_columns = FEATURE_COLUMNS
        self.num_features = NUMERICAL_FEATURES
        self.catg_features = CATEGORICAL_FEATURES
        self.target = TARGET
        self.scaler = StandardScaler()
        self.encoder = OrdinalEncoder()
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    def feature_scaler(self, df) -> pd.DataFrame:
        self.scaler.fit(df)
        df_scaled = self.scaler.transform(df)
        joblib.dump(self.scaler, 'models/scaler.joblib')
        return df_scaled

    def feature_encoder(self, df) -> pd.DataFrame:
        self.encoder.fit(df)
        df_encoded = self.encoder.transform(df)
        joblib.dump(self.encoder, 'models/encoder.joblib')
        return df_encoded

    def feature_imputer(self, df) -> pd.DataFrame:
        self.imputer.fit(df)
        df_imputed = pd.DataFrame(self.imputer.transform(df), columns=df.columns)
        return df_imputed

    def train(self, X_train) -> pd.DataFrame:
        X_train[self.num_features] = self.feature_scaler(X_train[self.num_features])
        X_train[self.catg_features] = self.feature_encoder(X_train[self.catg_features])
        return X_train

    def test(self, X_test) -> pd.DataFrame:
        X_test[self.num_features] = self.feature_scaler(X_test[self.num_features])
        X_test[self.catg_features] = self.feature_encoder(X_test[self.catg_features])
        return X_test

    def inference(self, test_df, scaler, encoder) -> pd.DataFrame:
        test_df = test_df[self.feature_columns]
        test_df = self.feature_imputer(test_df)
        test_df[self.num_features] = scaler.transform(test_df[self.num_features])
        test_df[self.catg_features] = encoder.transform(test_df[self.catg_features])
        return test_df
