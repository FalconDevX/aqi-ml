import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

prediction_hours = 10
lags = [1, 2, 3, 6, 12, 24, 48]


def train_direct_models(csv_path, index_name):
    """Train one direct-forecast model per prediction horizon."""
    df = pd.read_csv(csv_path)

    data = df[['Time', index_name]].copy()
    data['Time'] = pd.to_datetime(data['Time'])
    data.sort_values('Time', inplace=True)
    data.set_index('Time', inplace=True)

    data[index_name] = data[index_name].interpolate(method='linear').bfill().ffill()

    features = [
        'hour', 'dayofweek', 'month', 'dayofyear',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        f'{index_name}_lag_1',
        f'{index_name}_lag_2',
        f'{index_name}_lag_3',
        f'{index_name}_lag_6',
        f'{index_name}_lag_12',
        f'{index_name}_lag_24',
        f'{index_name}_lag_48',
    ]

    os.makedirs('models/direct', exist_ok=True)

    print(f"Training direct models for {index_name} ({prediction_hours} horizons)...")

    for horizon in range(1, prediction_hours + 1):
        horizon_data = data.copy()

        horizon_data['target'] = horizon_data[index_name].shift(-horizon)

        future_time = pd.Series(horizon_data.index, index=horizon_data.index).shift(-horizon)

        horizon_data['hour'] = future_time.dt.hour
        horizon_data['dayofweek'] = future_time.dt.dayofweek
        horizon_data['month'] = future_time.dt.month
        horizon_data['dayofyear'] = future_time.dt.dayofyear

        horizon_data['hour_sin'] = np.sin(2 * np.pi * horizon_data['hour'] / 24)
        horizon_data['hour_cos'] = np.cos(2 * np.pi * horizon_data['hour'] / 24)
        horizon_data['month_sin'] = np.sin(2 * np.pi * horizon_data['month'] / 12)
        horizon_data['month_cos'] = np.cos(2 * np.pi * horizon_data['month'] / 12)

        for lag in lags:
            horizon_data[f'{index_name}_lag_{lag}'] = horizon_data[index_name].shift(lag - 1)

        horizon_data.dropna(inplace=True)

        x = horizon_data[features]
        y = horizon_data['target']

        train_size = int(len(horizon_data) * 0.8)

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

        model_path = f'models/direct/{index_name}_h{horizon}_model.joblib'
        joblib.dump(model, model_path)

        print(f"+{horizon}h -> MAE: {mae:.2f}, R2: {r2:.3f}  (saved to '{model_path}')")


if __name__ == '__main__':
    train_direct_models('data/merged_PM10_2017_2023.csv', 'PM10')
