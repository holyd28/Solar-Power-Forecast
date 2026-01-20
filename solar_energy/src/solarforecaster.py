import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

class SolarForecaster:
    def __init__(self):
        # learning rate = 0.05 to prevent overfitting
        self.model = XGBRegressor(random_state = 42, n_jobs = 1)
        
    def hyperparameter_tuning(self, X_train, y_train):
        # set of parameters to tune model
        params = {
            "n_estimators": [250, 500, 750, 1000],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        }
        
        # use timeseriessplit to respect order of time
        tscv = TimeSeriesSplit(n_splits = 3)
        search = RandomizedSearchCV(
            estimator = self.model,
            param_distributions = params,
            n_iter = 10,
            scoring = "neg_mean_absolute_error",
            cv = tscv,
            verbose = 1,
            random_state = 42,
            n_jobs = -1,
        )
        search.fit(X_train, y_train)
        
        print(f"Best parameters found: {search.best_params_}")
        
        self.model = search.best_estimator_
        
        
        
        
    def preprocessing(self, gen_df, weather_df):
        
        # fix date formats to become YYYY-MM-DD
        gen_df["DATE_TIME"] = pd.to_datetime(gen_df["DATE_TIME"], dayfirst = True) # follows DD-MM-YYYY
        weather_df["DATE_TIME"] = pd.to_datetime(weather_df["DATE_TIME"])
        
        # only 1 weather station exists, hence we ensure each data is unique
        weather_unique = weather_df.drop_duplicates(subset = ["DATE_TIME"])
        
        # merge on timestamp
        merged = pd.merge(gen_df, weather_unique, on = "DATE_TIME", how = "inner")
        return merged
        
        
    def engineer_features(self, df):
        data = df.copy()
        
        # standardise column names
        data.columns = data.columns.str.lower()
        
        
        # use cyclical time encoding to preserve "closeness" relative to time
        
        # hour of day to capture day/night cycle
        data["hour_sin"] = np.sin(2 * np.pi * data["date_time"].dt.hour / 24)
        data["hour_cos"] = np.cos(2 * np.pi * data["date_time"].dt.hour / 24)
        
        # month of year to capture sun angle variations
        data["month_sin"] = np.sin(2 * np.pi * data["date_time"].dt.month / 12)
        data["month_cos"] = np.cos(2 * np.pi * data["date_time"].dt.month / 12)
        
        # thermal efficiency proxy (+1 to avoid division by zero)
        data["efficiency proxy"] = data["irradiation"] / (data["module_temperature"] + 1)
        
        # lag features
        data["lag_1"] = data.groupby("source_key_x")["dc_power"].shift(1)
        
        # volatility feature
        data["irradiance_std"] = data["irradiation"].rolling(4).std()
        
        # rolling mean
        data["rolling_mean_4"] = data.groupby("source_key_x")["dc_power"].transform(lambda t: t.rolling(window = 4).mean())
        
        # drop nas created when shifting
        data = data.dropna()
        
        features = ["hour_sin", "hour_cos", "month_sin", "month_cos", "ambient_temperature", 
                    "module_temperature", "irradiation", "efficiency proxy", "lag_1", "rolling_mean_4", "irradiance_std"]
        target = "dc_power" 
        
        return data[features], data[target], data
    
    def train(self,X_train, y_train, X_test, y_test):
        self.model.fit(X_train, y_train, eval_set = [(X_test, y_test)], verbose = False)
