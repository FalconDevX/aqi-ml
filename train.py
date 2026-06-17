import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


def create_features(df):
    """Function creating time features and lags."""
    data = df.copy()
    target = data.columns[0]

    data['hour'] = data.index.hour
    data['dayofweek'] = data.index.dayofweek
    data['month'] = data.index.month
    data['dayofyear'] = data.index.dayofyear

    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

    for lag in [1, 2, 3, 24, 48]:
        data[f'{target}_lag_{lag}'] = data[target].shift(lag)

    return data


def train_and_save_model(csv_path, index_name):
    """Function preparing data and training model."""
    df = pd.read_csv(csv_path)

    data = df[['Time', index_name]].copy()
    data['Time'] = pd.to_datetime(data['Time'])
    data.sort_values('Time', inplace=True)
    data.set_index('Time', inplace=True)

    data[index_name] = data[index_name].interpolate(method='linear').bfill().ffill()

    data = create_features(data)
    data.dropna(inplace=True)

    features = [
        'hour', 'dayofweek', 'month', 'dayofyear',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        f'{index_name}_lag_1',
        f'{index_name}_lag_2',
        f'{index_name}_lag_3',
        f'{index_name}_lag_24',
        f'{index_name}_lag_48',
    ]

    x = data[features]
    y = data[index_name]

    train_size = int(len(data) * 0.8)

    x_train = x.iloc[:train_size]
    x_test = x.iloc[train_size:]

    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    model = HistGradientBoostingRegressor(
        max_iter=1000,
        random_state=42,
    )

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    os.makedirs('models/exp', exist_ok=True)

    model_path = f'models/exp/{index_name}_model.joblib'
    joblib.dump(model, model_path)

    print(f"Saved model to '{model_path}'")
    print(f"MAE: {mae:.2f} µg/m³")
    print(f"R²: {r2:.3f}")


train_and_save_model("data/merged_PM10_2017_2023.csv", "PM10")
