import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

history_hours = 24
prediction_hours = 10
max_lag = 48

def load_station_csv(csv_path, index_name):
    df = pd.read_csv(csv_path)
    time_col = next(
        (col for col in df.columns if col.lower() in ("time", "timestamp")),
        df.columns[0],
    )
    value_col = next(col for col in df.columns if col != time_col)

    df = df[[time_col, value_col]].copy()
    df.rename(columns={time_col: "Time", value_col: index_name}, inplace=True)
    df = df.dropna(subset=[index_name, "Time"])
    df["Time"] = pd.to_datetime(df["Time"])
    df.sort_values("Time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def get_lag(history, series, current_index, lag):
    if len(history) >= lag:
        return history[-lag]
    ref_index = current_index - lag
    if ref_index >= 0:
        return series.iloc[ref_index]
    return series.iloc[0]

def prepend_context(df_test, context_csv_path, index_name):
    if not context_csv_path:
        return df_test

    df_context = load_station_csv(context_csv_path, index_name)
    test_start = df_test["Time"].min()
    df_context = df_context[df_context["Time"] < test_start]

    if df_context.empty:
        return df_test

    df_context = df_context.tail(max_lag)
    combined = pd.concat([df_context, df_test], ignore_index=True)
    combined.drop_duplicates(subset=["Time"], keep="last", inplace=True)
    combined.sort_values("Time", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined

def session_start_indices(total_rows):
    last_start = total_rows - prediction_hours
    if last_start < 0:
        return []

    if total_rows >= max_lag + prediction_hours:
        return list(range(max_lag, last_start + 1))
    if total_rows >= history_hours + prediction_hours:
        return list(range(history_hours, last_start + 1))

    return [last_start]

#take always second column from csv file as a station
def verify_model(model_path, csv_path, index_name, context_csv_path=None):
    """
    Function loading the previously trained model from the given path.
    """
    try:
        model = joblib.load(model_path)
        print(f"Successfully loaded model from '{model_path}'.")
    except FileNotFoundError:
        print(f"File '{model_path}' not found. Make sure the model has been trained.")
        exit()

    df_test = load_station_csv(csv_path, index_name)
    df_new = prepend_context(df_test, context_csv_path, index_name)

    min_rows = history_hours + prediction_hours
    if len(df_new) < prediction_hours:
        print(
            "Not enough data! "
            f"Need at least {prediction_hours} records to verify a "
            f"{prediction_hours}h forecast, but have only {len(df_new)}."
        )
        exit()

    if len(df_new) < min_rows:
        print(
            f"Warning: only {len(df_test)} test records "
            f"(recommended {min_rows} = {history_hours}h history + {prediction_hours}h forecast). "
            "Running a single session with shorter history."
        )

    if len(df_new) > len(df_test):
        print(f"Prepended {len(df_new) - len(df_test)} history rows from context file.")

    print(
        f"Generating recursive prediction for {prediction_hours}h "
        f"with {history_hours}h history on all data..."
    )

    predictions_by_time = {}
    real_by_time = {}
    all_real = []
    all_predictions = []
    number_of_sessions = 0

    for start_index in session_start_indices(len(df_new)):
        history_start = max(0, start_index - history_hours)
        history_index = df_new[index_name].iloc[history_start:start_index].tolist()
        number_of_sessions += 1

        for step in range(prediction_hours):
            current_index = start_index + step
            target_time = df_new['Time'].iloc[current_index]

            hour = target_time.hour
            month = target_time.month

            series = df_new[index_name]
            input_data = pd.DataFrame([{
                'hour': hour,
                'dayofweek': target_time.dayofweek,
                'month': month,
                'dayofyear': target_time.dayofyear,
                'hour_sin': np.sin(2 * np.pi * hour / 24),
                'hour_cos': np.cos(2 * np.pi * hour / 24),
                'month_sin': np.sin(2 * np.pi * month / 12),
                'month_cos': np.cos(2 * np.pi * month / 12),
                f'{index_name}_lag_1': get_lag(history_index, series, current_index, 1),
                f'{index_name}_lag_2': get_lag(history_index, series, current_index, 2),
                f'{index_name}_lag_3': get_lag(history_index, series, current_index, 3),
                f'{index_name}_lag_24': get_lag(history_index, series, current_index, 24),
                f'{index_name}_lag_48': get_lag(history_index, series, current_index, 48),
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

    plt.plot(
        daty, predictions_average,
        label=f'Recursive forecast ({prediction_hours}h avg)',
        color=pred_color, linestyle='--', linewidth=2.2,
    )

    plt.title(
        f'{index_name} — {prediction_hours}h forecast ({history_hours}h history)',
        fontsize=14, weight='bold', pad=15,
    )

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

verify_model("models/PM25_model.joblib", "data/test_PM25.csv", "PM25")