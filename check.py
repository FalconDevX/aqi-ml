import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

HISTORIA_GODZIN = 48
PROGNOZA_GODZIN = 10
# 1. Wczytanie wcześniej wytrenowanego modelu
try:
    model = joblib.load('pm10_model.joblib')
    print("Pomyślnie wczytano model z pliku 'pm10_model.joblib'.")
except FileNotFoundError:
    print("Nie znaleziono pliku 'pm10_model.joblib'. Upewnij się, że model został wytrenowany.")
    exit()

# 2. Wczytanie nowych danych testowych
df_nowe = pd.read_csv("Tarnow_Sitko_PM10_interpolated.csv")
df_nowe = df_nowe.dropna(subset=['PM10', 'Time'])  # Usunięcie pustych wierszy
df_nowe['Time'] = pd.to_datetime(df_nowe['Time'])
df_nowe.sort_values('Time', inplace=True)
df_nowe.reset_index(drop=True, inplace=True)

# Sprawdzenie, czy mamy wystarczająco dużo danych do co najmniej jednej sesji 24h
minimalna_liczba_wierszy = HISTORIA_GODZIN + PROGNOZA_GODZIN
if len(df_nowe) < minimalna_liczba_wierszy:
    print(
        "Za mało danych! "
        f"Do prognozy 24h potrzeba minimum {minimalna_liczba_wierszy} rekordów, "
        f"a dostępnych jest tylko {len(df_nowe)}."
    )
    exit()

print("Generowanie rekurencyjnej prognozy 24h na całych danych...")

przewidywania_na_czas = {}
rzeczywiste_na_czas = {}
wszystkie_rzeczywiste = []
wszystkie_przewidywane = []
liczba_sesji = 0

ostatni_start = len(df_nowe) - PROGNOZA_GODZIN

# 3. Rekurencyjna prognoza 24h wykonywana dla każdego możliwego punktu startowego
for start_index in range(HISTORIA_GODZIN, ostatni_start + 1):
    historia_pm10 = df_nowe['PM10'].iloc[start_index - HISTORIA_GODZIN:start_index].tolist()
    liczba_sesji += 1

    for krok in range(PROGNOZA_GODZIN):
        aktualny_indeks = start_index + krok
        data_docelowa = df_nowe['Time'].iloc[aktualny_indeks]

        hour = data_docelowa.hour
        month = data_docelowa.month

        input_data = pd.DataFrame([{
            'hour': hour,
            'dayofweek': data_docelowa.dayofweek,
            'month': month,
            'dayofyear': data_docelowa.dayofyear,
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12),
            'PM10_lag_1': historia_pm10[-1],
            'PM10_lag_2': historia_pm10[-2],
            'PM10_lag_3': historia_pm10[-3],
            'PM10_lag_24': historia_pm10[-24],
            'PM10_lag_48': historia_pm10[-48]
        }])

        predykcja = model.predict(input_data)[0]
        rzeczywista = df_nowe['PM10'].iloc[aktualny_indeks]

        # Doklejamy predykcję do historii, aby kolejny krok korzystał już z prognoz zamiast nowych pomiarów.
        historia_pm10.append(predykcja)

        wszystkie_przewidywane.append(predykcja)
        wszystkie_rzeczywiste.append(rzeczywista)
        rzeczywiste_na_czas[data_docelowa] = rzeczywista
        przewidywania_na_czas.setdefault(data_docelowa, []).append(predykcja)

# 4. Obliczenie błędu na całym zbiorze sesji oraz po agregacji po czasie
mae_wszystkie_sesje = mean_absolute_error(wszystkie_rzeczywiste, wszystkie_przewidywane)
daty = sorted(przewidywania_na_czas.keys())
rzeczywiste = [rzeczywiste_na_czas[data] for data in daty]
przewidywane_srednie = [np.mean(przewidywania_na_czas[data]) for data in daty]
mae_agregowane = mean_absolute_error(rzeczywiste, przewidywane_srednie)

print(f"\nGOTOWE!")
print(f"Wykonano {liczba_sesji} sesji prognozy po {PROGNOZA_GODZIN} godzin.")
print(f"Łącznie wygenerowano {len(wszystkie_przewidywane)} predykcji.")
print(f"MAE dla wszystkich kroków ze wszystkich sesji: {mae_wszystkie_sesje:.2f} µg/m³")
print(f"MAE po uśrednieniu nakładających się prognoz w czasie: {mae_agregowane:.2f} µg/m³")

# 5. Generowanie wykresu zagregowanego dla całego zakresu danych
plt.figure(figsize=(12, 6))
plt.plot(daty, rzeczywiste, label='Rzeczywiste PM10', color='blue', linewidth=2)
plt.plot(daty, przewidywane_srednie, label='Średnia prognoza rekurencyjna 24h', color='orange', linestyle='--', linewidth=2)
plt.title('Rekurencyjna prognoza PM10 24h na całym zakresie danych')
plt.xlabel('Data')
plt.ylabel('Stężenie PM10 (µg/m³)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# Zapis wykresu
plt.savefig('weryfikacja_modelu.png')
print("Zapisano wykres do pliku 'weryfikacja_modelu.png'")
# Jeśli odpalasz to w Jupyter Notebook użyj: plt.show()