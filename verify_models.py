import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

history_hours = 48
prediction_hours = 10

#take always second column from csv file as a station
def verify_model(model_path, csv_path, index_name):
    """
    Function loading the previously trained model from the given path.
    """
    try:
        model = joblib.load(model_path)
        print(f"Successfully loaded model from '{model_path}'.")
    except FileNotFoundError:
        print(f"File '{model_path}' not found. Make sure the model has been trained.")
        exit()

    # Loading new test data
    df_new = pd.read_csv(csv_path)
    df_new = df_new[['Time', df_new.columns[1]]].copy()
    df_new.rename(columns={df_new.columns[1]: index_name}, inplace=True)
    df_new = df_new.dropna(subset=[index_name, 'Time'])
    df_new['Time'] = pd.to_datetime(df_new['Time'])
    df_new.sort_values('Time', inplace=True)
    df_new.reset_index(drop=True, inplace=True)

    #checking if there is enough data for at least one 24h session
    min_rows = history_hours + prediction_hours
    if len(df_new) < min_rows:
        print(
            "Not enough data! "
            f"To predict 24h, we need at least {min_rows} records, "
            f"but we have only {len(df_new)}."
        )
        exit()

    print("Generating recursive prediction for 24h on all data...")

    predictions_by_time = {}
    real_by_time = {}
    all_real = []
    all_predictions = []
    number_of_sessions = 0

    last_start = len(df_new) - prediction_hours

    # Recursive prediction for 24h for each possible starting point
    for start_index in range(history_hours, last_start + 1):
        history_index = df_new[index_name].iloc[start_index - history_hours:start_index].tolist()
        number_of_sessions += 1

        for step in range(prediction_hours):
            current_index = start_index + step
            target_time = df_new['Time'].iloc[current_index]

            hour = target_time.hour
            month = target_time.month

            input_data = pd.DataFrame([{
                'hour': hour,
                'dayofweek': target_time.dayofweek,
                'month': month,
                'dayofyear': target_time.dayofyear,
                'hour_sin': np.sin(2 * np.pi * hour / 24),
                'hour_cos': np.cos(2 * np.pi * hour / 24),
                'month_sin': np.sin(2 * np.pi * month / 12),
                'month_cos': np.cos(2 * np.pi * month / 12),
                'PM10_lag_1': history_index[-1],
                'PM10_lag_2': history_index[-2],
                'PM10_lag_3': history_index[-3],
                'PM10_lag_24': history_index[-24],
                'PM10_lag_48': history_index[-48]
            }])

            prediction = model.predict(input_data)[0]
            real = df_new[index_name].iloc[current_index]

            # adding the prediction to the history, so the next step can use the prediction instead of the new measurement.
            history_index.append(prediction)

            all_predictions.append(prediction)
            all_real.append(real)
            real_by_time[target_time] = real
            predictions_by_time.setdefault(target_time, []).append(prediction)

    # calculating the error on the whole set of sessions and by aggregation by time
    mae_all_sessions = mean_absolute_error(all_real, all_predictions)
    daty = sorted(predictions_by_time.keys())
    real = [real_by_time[data] for data in daty]
    predictions_average = [np.mean(predictions_by_time[data]) for data in daty]
    mae_aggregated = mean_absolute_error(real, predictions_average)

    print(f"\nDONE!")
    print(f"Executed {number_of_sessions} sessions of prediction for {prediction_hours} hours.")
    print(f"Total {len(all_predictions)} predictions.")
    print(f"MAE for all steps in all sessions: {mae_all_sessions:.2f} µg/m³")
    print(f"MAE by averaging overlapping predictions by time: {mae_aggregated:.2f} µg/m³")

    plt.style.use('dark_background')

    plt.figure(figsize=(12, 6), dpi=120)

    real_color = '#4FC3F7'        
    pred_color = '#FFB74D'       

    plt.plot(daty, real, label=f'Real {index_name}', color=real_color, linewidth=2.2)

    plt.plot(daty, predictions_average, label='Recursive forecast (24h avg)', color=pred_color, linestyle='--', linewidth=2.2)

    plt.title(f'{index_name} — 24h Recursive Forecast', fontsize=14, weight='bold', pad=15)

    plt.xlabel('Date', fontsize=11)
    plt.ylabel(f'{index_name} (µg/m³)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.15)

    plt.legend(frameon=False, fontsize=10)

    plt.xticks(rotation=45, fontsize=9)
    plt.yticks(fontsize=9)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_color('#888')
    ax.spines['bottom'].set_color('#888')

    plt.tight_layout()

    plt.savefig( f'verif_images/verify_{index_name}_model.png', dpi=300, bbox_inches='tight', facecolor='#0E1117' )
    print(f"Saved plot to 'verif_images/verify_{index_name}_model.png'")

    plt.show()