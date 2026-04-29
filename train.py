from tkinter import X
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

def create_features(df):
    """Function creating time features and lags."""
    data = df.copy()
    target = data.columns[0]

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
        data[f'{target}_lag_{lag}'] = data[target].shift(lag)

    return data

def train_and_save_model(csv_path):
    "function preparing data and training model"
    df = pd.read_csv(csv_path)
    target = df.columns[1]
    data = df[['Time', target]].copy()
    data.rename(columns={target: target}, inplace=True)
    
    data['Time'] = pd.to_datetime(data['Time'])
    data.sort_values('Time', inplace=True)
    data.set_index('Time', inplace=True)

    data[target] = data[target].interpolate(method='linear').bfill().ffill()

    #features engineering (lags )
    data = create_features(data)
    data.dropna(inplace=True)

    features = [
        'hour', 'dayofweek', 'month', 'dayofyear', 
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        f'{target}_lag_1', f'{target}_lag_2', f'{target}_lag_3',
        f'{target}_lag_24', f'{target}_lag_48'
    ]

    x = data[features]
    y = data[target]

    #training and split 80% training, 20% test
    train_size = int(len(data) * 0.8)
    x_train, x_test = x.iloc[:train_size], x.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    model = HistGradientBoostingRegressor(max_iter=1000, random_state=42)
    model.fit(x_train, y_train)

    #check mae = mean absolute error
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)

    #save model
    joblib.dump(model, f'models/{target}_model.joblib')
    print(f"Saved model to 'models/{target}_model.joblib'")
    print(f"MAE: {mae:.2f} µg/m³")

train_and_save_model("data/merged_PM10_2017_2023.csv")