import joblib

DB_HOST = "localhost"
DB_NAME = "WINE_DB"
DB_USER = "postgres"
DB_PASSWORD = "root321"

def load_constants():
    try:
        model_obj = joblib.load("Model/model.joblib")
        feature_obj = joblib.load("Model/features.joblib")
        scaler_obj = joblib.load("Model/scaler.joblib")
        return model_obj, feature_obj, scaler_obj
    except Exception as e:
        print(e.args)
        return [], [], []