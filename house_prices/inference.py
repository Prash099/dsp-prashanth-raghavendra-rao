import house_prices.preprocess as preprocess
import numpy as np
import pandas as pd
import joblib

model = joblib.load('../models/model.joblib')
scaler = joblib.load('../models/scaler.joblib')
encoder = joblib.load('../models/encoder.joblib')


def make_predictions(test_df: pd.DataFrame) -> np.ndarray:
    test_df = preprocess.inference(test_df, scaler, encoder)
    prediction = model.predict(test_df)
    return prediction
