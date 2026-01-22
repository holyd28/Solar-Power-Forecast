import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from src.solarforecaster import SolarForecaster

def main():
    print("Loading data...")
    # load data sets
    try:
        gen_data = pd.read_csv("data/Plant_1_Generation_Data.csv")
        weather_data = pd.read_csv("data/Plant_1_Weather_Sensor_Data.csv")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    forecaster = SolarForecaster()
    
    # preprocess & merge into a singular dataframe
    print("Preprocessing data...")
    processed_data = forecaster.preprocessing(gen_data, weather_data)
    
    # feature engineering
    print("Engineering features...")
    X, y, final_data = forecaster.engineer_features(processed_data)
    
    # time-series split, training on first 80% & testing on last 20%
    split_index = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    # finding best parameters
    forecaster.hyperparameter_tuning(X_train, y_train)
    
    # training model
    print(f"Training model on {len(X_train)} samples...")
    forecaster.train(X_train, y_train, X_test, y_test)
    
    # evaluating model
    print("Evaluating model...")
    y_pred = forecaster.model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print("\n" + "-"*30)
    print(f"R2 Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f} kW")
    print("-"*30 + "\n")
    
    # day-only testing
    daytime_mask = y_test > 0
    y_test_day = y_test[daytime_mask]
    preds_day = y_pred[daytime_mask]

    r2_day = r2_score(y_test_day, preds_day)
    mae_day = mean_absolute_error(y_test_day, preds_day)

    print("\n" + "-"*30)
    print(f"Daytime-Only R2: {r2_day:.4f}")
    print(f"Daytime-Only MAE: {mae_day:.2f} kW")   
    print("-"*30 + "\n")
    
    
    # plotting results
    plt.figure(figsize = (12, 6))
    plt.plot(y_test.values[:200], label = "Actual Power", color = "black", alpha= 0.6)
    plt.plot(y_pred[:200], label = "Predicted Power", color = "orange", linestyle = "--")
    plt.title(f"Solar Power Forecast (R2: {r2:.4f}, MAE: {mae:.4f} kW)")
    plt.xlabel("Time Steps (15-min intervals)")
    plt.ylabel("DC Power Output (kW)")
    plt.legend()
    plt.grid(True, alpha = 0.3)
    
    plt.savefig("results/solar_power_forecast_plot.png")
    print("Plot saved to results/solar_power_forecast_plot.png")
    
if __name__ == "__main__":
    main()
