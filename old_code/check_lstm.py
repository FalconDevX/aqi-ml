import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import os

HISTORIA_GODZIN = 48
PROGNOZA_GODZIN = 10

# 1. Wczytanie wcześniej wytrenowanego modelu LSTM i skalerów
try:
    model = tf.keras.models.load_model('pm10_lstm_model.keras')
    feature_scaler = joblib.load('feature_scaler.joblib')
    target_scaler = joblib.load('target_scaler.joblib')
    print("Pomyślnie wczytano model LSTM oraz skalery.")
except Exception as e:
    print(f"Błąd podczas wczytywania modelu lub skalerów: {e}")
    print("Upewnij się, że 'pm10_lstm_model.keras', 'feature_scaler.joblib' i 'target_scaler.joblib' są w folderze.")
    exit()

# 2. Wczytanie nowych danych testowych
df_nowe = pd.read_csv("Tarnow_Sitko_PM10_interpolated.csv")
df_nowe = df_nowe.dropna(subset=['PM10', 'Time'])
df_nowe['Time'] = pd.to_datetime(df_nowe['Time'])
df_nowe.sort_values('Time', inplace=True)
df_nowe.reset_index(drop=True, inplace=True)

# Sprawdzenie, czy mamy wystarczająco dużo danych
minimalna_liczba_wierszy = HISTORIA_GODZIN + PROGNOZA_GODZIN
if len(df_nowe) < minimalna_liczba_wierszy:
    print(f"Za mało danych! Potrzeba minimum {minimalna_liczba_wierszy} rekordów.")
    exit()

# --- PRZYGOTOWANIE CECH DLA CAŁEGO ZBIORU ---
# Robimy to raz przed pętlą, żeby przyspieszyć obliczenia
df_nowe['hour'] = df_nowe['Time'].dt.hour
df_nowe['month'] = df_nowe['Time'].dt.month
df_nowe['hour_sin'] = np.sin(2 * np.pi * df_nowe['hour'] / 24)
df_nowe['hour_cos'] = np.cos(2 * np.pi * df_nowe['hour'] / 24)
df_nowe['month_sin'] = np.sin(2 * np.pi * df_nowe['month'] / 12)
df_nowe['month_cos'] = np.cos(2 * np.pi * df_nowe['month'] / 12)
df_nowe['dayofweek'] = df_nowe['Time'].dt.dayofweek

# Ważne: Kolejność MUSI być taka sama jak podczas treningu LSTM!
cols_order = ['PM10', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dayofweek']

# Przekształcamy na szybką tablicę numpy
dane_macierz = df_nowe[cols_order].values 
czasy = df_nowe['Time'].values

print("Generowanie rekurencyjnej prognozy LSTM na całych danych...")

przewidywania_na_czas = {}
rzeczywiste_na_czas = {}
wszystkie_rzeczywiste = []
wszystkie_przewidywane = []
liczba_sesji = 0

ostatni_start = len(df_nowe) - PROGNOZA_GODZIN

# 3. Rekurencyjna prognoza
for start_index in range(HISTORIA_GODZIN, ostatni_start + 1):
    # Kopiujemy prawdziwą historię dla danego punktu startowego (48 wierszy, 6 kolumn)
    aktualne_okienko = dane_macierz[start_index - HISTORIA_GODZIN : start_index].copy()
    liczba_sesji += 1

    for krok in range(PROGNOZA_GODZIN):
        aktualny_indeks = start_index + krok
        data_docelowa = czasy[aktualny_indeks]
        rzeczywista_wartosc_pm10 = dane_macierz[aktualny_indeks, 0] # PM10 jest na indeksie 0

        # Skalowanie aktualnego okienka (48 wierszy)
        skalowane_okienko = feature_scaler.transform(aktualne_okienko)
        
        # Formatowanie do 3D dla LSTM (1 próbka, 48 kroków, 6 cech)
        X_input = skalowane_okienko.reshape(1, HISTORIA_GODZIN, len(cols_order))

        # Predykcja (zwraca wartość od 0 do 1)
        pred_scaled = model.predict(X_input, verbose=0)
        
        # Odskalowanie do rzeczywistej wartości PM10
        predykcja = target_scaler.inverse_transform(pred_scaled)[0][0]

        # --- ZAPĘTLENIE (Zjadanie własnego ogona) ---
        # Tworzymy nowy wiersz z cechami z "przyszłości", ale podmieniamy prawdziwe PM10 na naszą predykcję!
        nowy_wiersz = dane_macierz[aktualny_indeks].copy()
        nowy_wiersz[0] = predykcja 
        
        # Przesuwamy okienko: odrzucamy najstarszy pomiar, dodajemy nowo przewidziany
        aktualne_okienko = np.vstack([aktualne_okienko[1:], nowy_wiersz])

        # Zapis statystyk
        wszystkie_przewidywane.append(predykcja)
        wszystkie_rzeczywiste.append(rzeczywista_wartosc_pm10)
        rzeczywiste_na_czas[data_docelowa] = rzeczywista_wartosc_pm10
        przewidywania_na_czas.setdefault(data_docelowa, []).append(predykcja)
        
    if liczba_sesji % 50 == 0:
        print(f"Przetworzono {liczba_sesji} sesji...")

# 4. Obliczenie błędu na całym zbiorze sesji oraz po agregacji po czasie
mae_wszystkie_sesje = mean_absolute_error(wszystkie_rzeczywiste, wszystkie_przewidywane)
daty = sorted(przewidywania_na_czas.keys())
rzeczywiste = [rzeczywiste_na_czas[data] for data in daty]
przewidywane_srednie = [np.mean(przewidywania_na_czas[data]) for data in daty]
mae_agregowane = mean_absolute_error(rzeczywiste, przewidywane_srednie)

print(f"\nGOTOWE!")
print(f"Wykonano {liczba_sesji} sesji prognozy po {PROGNOZA_GODZIN} godzin.")
print(f"Łącznie wygenerowano {len(wszystkie_przewidywane)} predykcji LSTM.")
print(f"MAE dla wszystkich kroków ze wszystkich sesji: {mae_wszystkie_sesje:.2f} µg/m³")
print(f"MAE po uśrednieniu nakładających się prognoz w czasie: {mae_agregowane:.2f} µg/m³")

# 5. Generowanie wykresu
plt.figure(figsize=(12, 6))
plt.plot(daty, rzeczywiste, label='Rzeczywiste PM10', color='blue', linewidth=2, alpha=0.7)
plt.plot(daty, przewidywane_srednie, label=f'Średnia prognoza LSTM {PROGNOZA_GODZIN}h', color='orange', linestyle='--', linewidth=2)
plt.title(f'Rekurencyjna prognoza LSTM PM10 na {PROGNOZA_GODZIN}h')
plt.xlabel('Data')
plt.ylabel('Stężenie PM10 (µg/m³)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('weryfikacja_modelu_lstm.png')
print("Zapisano wykres do pliku 'weryfikacja_modelu_lstm.png'")