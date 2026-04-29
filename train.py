from tkinter import X
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

def create_features(df, target_col):
    """Function creating time features and lags."""
    data = df.copy()

    #time features
    data['hour'] = data.index.hour
    data['dayofweek'] = data.index.dayofweek
    data['month'] = data.index.month
    data['dayofyear'] = data.index.dayofyear

    #cyclic features - model knows that 23 and 2am is close to each other
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

    # features from the past (lags)
    for lag in [1, 2, 3, 24, 48]:
        data[f'PM10_lag_{lag}'] = data[target_col].shift(lag)

    return data

def train_and_save_model(csv_path, best_station="MpKrakWadow-PM10-1g", index_name="PM10"):
    "function preparing data and training model"

    df = pd.read_csv(csv_path)
    data = df[['Time', best_station]].copy()
    data.rename(columns={best_station: 'PM10'}, inplace=True)
    
    data['Time'] = pd.to_datetime(data['Time'])
    data.sort_values('Time', inplace=True)
    data.set_index('Time', inplace=True)

    data[index_name] = data[index_name].interpolate(method='linear').bfill().ffill()

    #features engineering (lags )
    data = create_features(data, target_col=index_name)
    data.dropna(inplace=True)

    features = [
        'hour', 'dayofweek', 'month', 'dayofyear', 
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'PM10_lag_1', 'PM10_lag_2', 'PM10_lag_3', 'PM10_lag_24', 'PM10_lag_48'
    ]

    x = data[features]
    y = data[index_name]

    #training and split 80% training, 20% test
    train_size = int(len(data) * 0.8)
    x_train, x_test = x.iloc[:train_size], x.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    model = HistGradientBoostingRegressor(max_iter=200, random_state=42)

    model = HistGradientBoostingRegressor(max_iter=200, random_state=42)
    model.fit(x_train, y_train)

    #check mae = mean absolute error
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)

    #save model
    joblib.dump(model, f'{index_name}_model.joblib')
    print(f"Saved model to '{index_name}_model.joblib'")
    print(f"MAE: {mae:.2f} µg/m³")

train_and_save_model("data/PM10_1g_joint_2017-2023.csv", "MpTarRoSitko-PM10-1g", "PM10")