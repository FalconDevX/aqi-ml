import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks 
from pandas.plotting import autocorrelation_plot

def prepare_data(df, index_name):
    data = df[['Time', index_name]].copy()
    data.rename(columns={index_name: 'PM10'}, inplace=True)
    data['Time'] = pd.to_datetime(data['Time'])
    data.sort_values('Time', inplace=True)
    data.set_index('Time', inplace=True)
    data['PM10'] = data['PM10'].interpolate(method='linear').bfill().ffill()
    return data

def check_autocorrelation(csv_name, index_name):
    df = pd.read_csv(csv_name)
    data = prepare_data(df, index_name)

    #acf - autocorrelation function
    acf_values = acf(data['PM10'], nlags=200)

    # filter only significant peaks and valleys
    peaks, _ = find_peaks(acf_values, height=0.2)   
    valleys, _ = find_peaks(-acf_values)

    # print("Maxima (lags):", peaks)
    # print("Minima (lags):", valleys)

    # #draw plot
    # plt.figure(figsize=(14, 6))

    # lags = range(len(acf_values))
    # plt.plot(lags, acf_values, linewidth=2)

    # plt.scatter(peaks, acf_values[peaks], s=60, label='Maxima')
    # plt.scatter(valleys, acf_values[valleys], s=60, label='Minima')

    # # linie referencyjne
    # plt.axhline(0, linewidth=1)
    # plt.axhline(0.2, linestyle='--', linewidth=1)
    # plt.axhline(-0.2, linestyle='--', linewidth=1)

    # plt.xlim(0, 200)

    # plt.title(f"ACF for {index_name}", fontsize=14)
    # plt.xlabel("Lag (hours)", fontsize=12)
    # plt.ylabel("Autocorrelation", fontsize=12)

    # plt.legend()
    # plt.grid(alpha=0.3)

    # plt.tight_layout()
    # plt.show()

#check_autocorrelation("PM10_1g_joint_2017-2023.csv", "MpTarRoSitko-PM10-1g")