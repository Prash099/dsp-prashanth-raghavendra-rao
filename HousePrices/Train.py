from HousePrices.Preprocess import Preprocess
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from HousePrices.Constants import FEATURE_COLUMNS, TARGET
import numpy as np
import joblib


class Train:
    def __init__(self, df) -> None:
        self.train_df = df[FEATURE_COLUMNS]
        self.target = df[TARGET]
        self.preprocess = Preprocess()
        self.model = RandomForestRegressor(random_state=0)

    def build_model(self) -> dict[str, str]:
        X_train, X_test, y_train, y_test = train_test_split(self.train_df,
                                                            self.target,
                                                            test_size=0.3,
                                                            random_state=42)
        X_train = self.preprocess.train(X_train)
        model = self.model.fit(X_train, y_train)

        X_test = self.preprocess.test(X_test)
        y_pred = model.predict(X_test)
        score = self.compute_rmsle(y_test, y_pred)

        joblib.dump(model, 'models/model.joblib')
        return {'rmse : ': score}

    def compute_rmsle(self, y_test: np.ndarray,
                      y_pred: np.ndarray, precision: int = 2) -> float:
        rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
        return round(rmsle, precision)
