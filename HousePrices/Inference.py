from HousePrices.Preprocess import Preprocess
import numpy as np
import joblib


class Inference:
    def __init__(self, df) -> None:
        self.test_df = df
        self.preprocess = Preprocess()
        self.model = joblib.load('models/model.joblib')
        self.scaler = joblib.load('models/scaler.joblib')
        self.encoder = joblib.load('models/encoder.joblib')

    def make_predictions(self) -> np.ndarray:
        self.test_df = self.preprocess.inference(self.test_df,
                                                 self.scaler,
                                                 self.encoder)
        prediction = self.model.predict(self.test_df)
        return prediction
