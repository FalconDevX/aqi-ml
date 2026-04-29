import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks 
from pandas.plotting import autocorrelation_plot

def prepare_data(df, index_name):
    data = df[['Time', df.columns[1]]].copy()
    data.rename(columns={df.columns[1]: index_name}, inplace=True)
    data['Time'] = pd.to_datetime(data['Time'])
    data.sort_values('Time', inplace=True)
    data.set_index('Time', inplace=True)
    data[index_name] = data[index_name].interpolate(method='linear').bfill().ffill()
    return data

def check_autocorrelation(csv_name, index_name):
    df = pd.read_csv(csv_name)
    data = prepare_data(df, index_name)
    acf_values = acf(data[f"{index_name}"], nlags=200)
    peaks, _ = find_peaks(acf_values, height=0.2)
    valleys, _ = find_peaks(-acf_values)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 7))
    lags = range(len(acf_values))
    ax.plot(lags, acf_values, color="#00d4ff", linewidth=2.5, alpha=0.9, label="ACF")
    ax.fill_between(lags, acf_values, color="#00d4ff", alpha=0.1)
    ax.scatter(
        peaks,
        acf_values[peaks],
        color="#ff007f",
        s=80,
        zorder=5,
        label="Max",
        edgecolors="white",
        linewidth=1,
    )
    ax.scatter(
        valleys,
        acf_values[valleys],
        color="#ccff00",
        s=80,
        zorder=5,
        label="Min",
        edgecolors="white",
        linewidth=1,
    )
    ax.axhline(0, color="white", linewidth=1, alpha=0.5)
    ax.axhline(0.2, linestyle="--", color="#ffcc00", linewidth=1, alpha=0.6, label="Significance Threshold")
    ax.axhline(-0.2, linestyle="--", color="#ffcc00", linewidth=1, alpha=0.6)
    ax.set_xlim(0, 200)
    ax.set_ylim(min(acf_values) - 0.1, 1.05)
    ax.grid(True, linestyle=":", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.3)
    ax.spines["bottom"].set_alpha(0.3)
    ax.set_title(f"Autocorrelation Analysis: {index_name}", fontsize=18, fontweight="bold", pad=20, loc="left")
    ax.set_xlabel("Lag (hours)", fontsize=13, color="#cccccc")
    ax.set_ylabel("Autocorrelation Score", fontsize=13, color="#cccccc")
    ax.legend(frameon=True, facecolor="#1a1a1a", edgecolor="none", fontsize=11)
    plt.tight_layout()
    plt.show()
    print("Detected Max (lags):", peaks)
    print("Detected Min (lags):", valleys)

#check_autocorrelation("PM10_1g_joint_2017-2023.csv", "MpKrakWadow-PM10-1g")

def find_best_stations(csv_name):
    df = pd.read_csv(csv_name)
    df_numeric = df.select_dtypes(include=['number'])
    total = len(df_numeric)
    missing_percent = df_numeric.isna().sum() / total * 100
    coverage_percent = 100 - missing_percent

    ranking = pd.DataFrame({
        'station': coverage_percent.index,
        'coverage_%': coverage_percent.values
    })

    ranking = ranking.sort_values(by='coverage_%', ascending=False)
    top20 = ranking.head(50)

    print(top20.to_string(index=False))

find_best_stations("data/merged_PM10_2017_2023.csv")

#check_autocorrelation("data/merged_O3_2017_2023.csv", "O3")