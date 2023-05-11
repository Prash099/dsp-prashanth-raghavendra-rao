import house_prices.preprocess as preprocess
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from house_prices.constants import FEATURE_COLUMNS, TARGET
import numpy as np
import pandas as pd
import joblib


model = RandomForestRegressor(random_state=0)


def build_model(train_df: pd.DataFrame) -> dict[str, str]:
    target = train_df[TARGET]
    train_df = train_df[FEATURE_COLUMNS]
    X_train, X_test, y_train, y_test = train_test_split(train_df,
                                                        target,
                                                        test_size=0.3,
                                                        random_state=42)
    X_train = preprocess.train(X_train)
    model.fit(X_train, y_train)

    X_test = preprocess.test(X_test)
    y_pred = model.predict(X_test)
    score = compute_rmsle(y_test, y_pred)

    joblib.dump(model, '../models/model.joblib')
    return {'rmse : ': score}


def compute_rmsle(y_test: np.ndarray,
                  y_pred: np.ndarray,
                  precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)
