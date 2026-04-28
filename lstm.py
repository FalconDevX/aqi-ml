import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# ============================================================
# PARAMETRY GLOBALNE
# ============================================================
LOOK_BACK = 48       # Ile godzin wstecz widzi model
EPOCHS    = 50       # Górny limit – EarlyStopping zatrzyma wcześniej
BATCH     = 32
TRAIN_RATIO = 0.8    # 80% trening, 20% test


# ============================================================
# POMOCNICZE
# ============================================================

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje cykliczne cechy kalendarzowe."""
    data = df.copy()
    hour  = data.index.hour
    month = data.index.month

    data['hour_sin']  = np.sin(2 * np.pi * hour  / 24)
    data['hour_cos']  = np.cos(2 * np.pi * hour  / 24)
    data['month_sin'] = np.sin(2 * np.pi * month / 12)
    data['month_cos'] = np.cos(2 * np.pi * month / 12)
    data['dayofweek'] = data.index.dayofweek
    return data


def create_sequences(X: np.ndarray, y: np.ndarray, look_back: int):
    """Przekształca tablice 2D → sekwencje 3D dla LSTM."""
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i : i + look_back])
        ys.append(y[i + look_back])
    return np.array(Xs), np.array(ys)


# ============================================================
# TRENING
# ============================================================

def train_and_save_model(csv_path: str, station_col: str = 'MpTarRoSitko-PM10-1g'):
    print("=" * 60)
    print("KROK 1 – Wczytywanie i czyszczenie danych")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    data = df[['Time', station_col]].copy()
    data.rename(columns={station_col: 'PM10'}, inplace=True)
    data['Time'] = pd.to_datetime(data['Time'])
    data.sort_values('Time', inplace=True)
    data.set_index('Time', inplace=True)

    # Interpolacja braków – najpierw liniowo, potem fill resztek na brzegach
    data['PM10'] = data['PM10'].interpolate(method='linear').bfill().ffill()

    # Dodanie cech kalendarzowych
    data = create_time_features(data)

    feature_columns = ['PM10', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dayofweek']

    # ----------------------------------------------------------
    # POPRAWKA 1: Podział na train/test PRZED skalowaniem
    # (unikamy data leakage – scaler nie widzi zbioru testowego)
    # ----------------------------------------------------------
    print("\nKROK 2 – Chronologiczny podział danych (przed skalowaniem)")

    train_size = int(len(data) * TRAIN_RATIO)
    train_data = data.iloc[:train_size]
    test_data  = data.iloc[train_size:]

    print(f"  Trening : {train_data.index[0]}  →  {train_data.index[-1]}  ({len(train_data)} próbek)")
    print(f"  Test    : {test_data.index[0]}  →  {test_data.index[-1]}  ({len(test_data)} próbek)")

    # ----------------------------------------------------------
    # POPRAWKA 2: Scaler fitowany TYLKO na danych treningowych
    # ----------------------------------------------------------
    print("\nKROK 3 – Skalowanie (fit wyłącznie na zbiorze treningowym)")

    scaler = MinMaxScaler()
    scaler.fit(train_data[feature_columns])   # <-- tylko trening!

    train_scaled = scaler.transform(train_data[feature_columns])
    test_scaled  = scaler.transform(test_data[feature_columns])

    # Indeks kolumny PM10 – użyjemy go do odwrócenia skalowania
    pm10_idx = feature_columns.index('PM10')

    # ----------------------------------------------------------
    # Budowanie sekwencji 3D
    # ----------------------------------------------------------
    print("\nKROK 4 – Tworzenie sekwencji 3D dla LSTM")

    X_train, y_train = create_sequences(train_scaled, train_scaled[:, pm10_idx], LOOK_BACK)
    X_test,  y_test  = create_sequences(test_scaled,  test_scaled[:, pm10_idx],  LOOK_BACK)

    print(f"  X_train shape: {X_train.shape}  |  y_train shape: {y_train.shape}")
    print(f"  X_test  shape: {X_test.shape}   |  y_test  shape: {y_test.shape}")

    # ----------------------------------------------------------
    # POPRAWKA 3: Brak relu w LSTM (tanh jest domyślny i stabilny)
    #             Dodano Dropout przeciwko przeuczeniu
    # ----------------------------------------------------------
    print("\nKROK 5 – Budowanie modelu LSTM")

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(128, return_sequences=True),   # tanh domyślnie
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # ----------------------------------------------------------
    # POPRAWKA 4: EarlyStopping + ReduceLROnPlateau
    # ----------------------------------------------------------
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            patience=3,
            factor=0.5,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    print("\nKROK 6 – Trenowanie")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    # ----------------------------------------------------------
    # Ewaluacja
    # ----------------------------------------------------------
    print("\nKROK 7 – Ewaluacja na zbiorze testowym")

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # Odwrócenie skalowania przy użyciu parametrów kolumny PM10
    # MinMaxScaler: x_real = x_scaled * data_range + data_min
    pm10_min   = scaler.data_min_[pm10_idx]
    pm10_range = scaler.data_range_[pm10_idx]

    y_test_real = y_test       * pm10_range + pm10_min
    y_pred_real = y_pred_scaled * pm10_range + pm10_min

    mae  = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    print(f"  MAE  : {mae:.2f} µg/m³")
    print(f"  RMSE : {rmse:.2f} µg/m³")

    # Zapis
    model.save('pm10_lstm_model.keras')
    joblib.dump(scaler, 'feature_scaler.joblib')
    joblib.dump(feature_columns, 'feature_columns.joblib')
    print("\nZapisano: pm10_lstm_model.keras, feature_scaler.joblib, feature_columns.joblib")

    return history


# ============================================================
# PROGNOZA
# ============================================================

def predict_future_pm10(target_datetime, past_48h_pm10: list) -> float:
    """
    Przewiduje wartość PM10 dla wskazanej godziny.

    Parametry
    ----------
    target_datetime : str lub datetime
        Godzina, dla której chcemy prognozę (np. "2024-12-24 18:00:00").
    past_48h_pm10 : list[float]
        Dokładnie 48 wartości PM10 z kolejnych godzin poprzedzających
        target_datetime (od najstarszej do najnowszej).

    Zwraca
    -------
    float
        Prognozowane stężenie PM10 w µg/m³.
    """
    if len(past_48h_pm10) != LOOK_BACK:
        raise ValueError(
            f"Oczekiwano {LOOK_BACK} wartości historycznych, "
            f"otrzymano {len(past_48h_pm10)}."
        )

    # Wczytanie modelu i artefaktów
    model           = tf.keras.models.load_model('pm10_lstm_model.keras')
    scaler          = joblib.load('feature_scaler.joblib')
    feature_columns = joblib.load('feature_columns.joblib')

    target_dt = pd.to_datetime(target_datetime)

    # Rekonstrukcja DataFrame historycznego (48 godzin przed target)
    history_dates = pd.date_range(
        end=target_dt - pd.Timedelta(hours=1),
        periods=LOOK_BACK,
        freq='h'
    )
    df_history = pd.DataFrame({'PM10': past_48h_pm10}, index=history_dates)
    df_history = create_time_features(df_history)
    df_history = df_history[feature_columns]  # właściwa kolejność kolumn

    # Skalowanie (transform – NIE fit!)
    scaled_history = scaler.transform(df_history)

    # Wejście dla LSTM: (1 próbka, 48 kroków, n_cech)
    X_input = scaled_history.reshape(1, LOOK_BACK, len(feature_columns))

    # Prognoza
    pred_scaled = model.predict(X_input, verbose=0)[0, 0]

    # Odwrócenie skalowania
    pm10_idx   = feature_columns.index('PM10')
    pm10_min   = scaler.data_min_[pm10_idx]
    pm10_range = scaler.data_range_[pm10_idx]
    prediction = pred_scaled * pm10_range + pm10_min

    return float(prediction)


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    CSV_PATH = "PM10_1g_joint_2017-2023.csv"

    # --- Trening (odkomentuj przy pierwszym uruchomieniu) ---
    train_and_save_model(CSV_PATH)

    # --- Test predykcji ---
    if os.path.exists('pm10_lstm_model.keras'):
        dummy_history = [50 + (i % 5) for i in range(LOOK_BACK)]
        target        = "2024-12-24 18:00:00"
        result        = predict_future_pm10(target, dummy_history)
        print(f"\nPM10 dla {target}: {result:.2f} µg/m³")
    else:
        print("Model nie istnieje – najpierw uruchom train_and_save_model().")