import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

def create_features(df, target_col='PM10'):
    """Function creating time features and lags."""
    data = df.copy()
    
    # Cechy z kalendarza i czasu
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

def train_and_save_model(csv_path, best_station='MpTarRoSitko-PM10-1g'):
    """Funkcja przygotowująca dane, ucząca model i zapisująca go do pliku."""
    print("Wczytywanie i przygotowanie danych...")
    df = pd.read_csv(csv_path)
    
    data = df[['Time', best_station]].copy()
    data.rename(columns={best_station: 'PM10'}, inplace=True)
    
    data['Time'] = pd.to_datetime(data['Time'])
    data.sort_values('Time', inplace=True)
    data.set_index('Time', inplace=True)

    # Interpolacja braków danych
    data['PM10'] = data['PM10'].interpolate(method='linear').bfill().ffill()

    # Inżynieria cech
    data = create_features(data, target_col='PM10')
    data.dropna(inplace=True) # Usunięcie pierwszych 48h z powodu braku historii

    # Podział zmiennych
    features = [
        'hour', 'dayofweek', 'month', 'dayofyear', 
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'PM10_lag_1', 'PM10_lag_2', 'PM10_lag_3', 'PM10_lag_24', 'PM10_lag_48'
    ]
    X = data[features]
    y = data['PM10']

    # Podział chronologiczny: 80% trening, 20% test
    train_size = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    print("Trenowanie modelu (HistGradientBoostingRegressor)...")
    model = HistGradientBoostingRegressor(max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    #check mae = mean absolute error
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Średni błąd bezwzględny na zbiorze testowym (MAE): {mae:.2f} µg/m³")

    #save model
    joblib.dump(model, 'pm10_model.joblib')
    print("Zapisano model do pliku 'pm10_model.joblib'")

def predict_future_pm10(target_datetime, past_48h_values):
    """
    function using saved model to predict future PM10 based on date and knowing the last 48 hours (as a list of 49 values sorted chronologically). 
    """
    model = joblib.load('pm10_model.joblib')
    
    # Tworzymy jednowierszową ramkę danych dla szukanego momentu
    target_dt = pd.to_datetime(target_datetime)
    
    # Konstruujemy cechy czasowe dla docelowej daty
    hour = target_dt.hour
    month = target_dt.month
    
    # Przekazane wartości wstecz: past_48h_values musi mieć długość co najmniej 49 
    # (indeks -1 to aktualnie szukany element, -2 to 1h wstecz, -25 to 24h, -49 to 48h)
    lag_1 = past_48h_values[-2]
    lag_2 = past_48h_values[-3]
    lag_3 = past_48h_values[-4]
    lag_24 = past_48h_values[-25]
    lag_48 = past_48h_values[-49]

    input_data = pd.DataFrame([{
        'hour': hour,
        'dayofweek': target_dt.dayofweek,
        'month': month,
        'dayofyear': target_dt.dayofyear,
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'month_sin': np.sin(2 * np.pi * month / 12),
        'month_cos': np.cos(2 * np.pi * month / 12),
        'PM10_lag_1': lag_1,
        'PM10_lag_2': lag_2,
        'PM10_lag_3': lag_3,
        'PM10_lag_24': lag_24,
        'PM10_lag_48': lag_48
    }])

    # Przewidywanie
    prediction = model.predict(input_data)[0]
    return prediction

if __name__ == "__main__":
    # 1. Krok 1: Wytrenuj model używając swojego pliku CSV
    # Odkomentuj to, aby wytrenować po raz pierwszy:
    train_and_save_model("PM10_1g_joint_2017-2023.csv")

    # 2. Krok 2: Użyj dowolnych danych
    # Wymagane jest dostarczenie ostatnich 49 wartości PM10 (gdzie ostatnia to symulowane obecne okienko) 
    # Tutaj stworzyłem fałszywą listę 49 pomiarów z oscylującą wartością do testów:
    dummy_history = [50 + (i % 5) for i in range(49)]  
    
    data_do_prognozy = "2024-12-24 18:00:00" # Zima, wieczór, wigilia - idealnie dla czułości na cykl
    
    # Gdy model został zapisany, możesz uruchomić:
    # predicted_value = predict_future_pm10(data_do_prognozy, dummy_history)
    # print(f"Przewidywana wartość PM10 dla {data_do_prognozy}: {predicted_value:.2f} µg/m³")