import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle
from sklearn.metrics import mean_absolute_error, r2_score

history_hours = 48
prediction_hours = 10
max_lag = 48


class LegendTextHandle:
    pass


class LegendTextOnly(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        artist = Rectangle(
            (xdescent, ydescent),
            width,
            height,
            linewidth=0,
            facecolor='none',
            edgecolor='none',
            visible=False,
        )
        return [artist]


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

    horizon_results = {
        h: {"real": [], "pred": []}
        for h in range(1, prediction_hours + 1)
    }

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

            horizon = step + 1

            horizon_results[horizon]["real"].append(real)
            horizon_results[horizon]["pred"].append(prediction)

            history_index.append(prediction)

            all_predictions.append(prediction)
            all_real.append(real)
            real_by_time[target_time] = real
            predictions_by_time.setdefault(target_time, []).append(prediction)

    mae_all_sessions = mean_absolute_error(all_real, all_predictions)
    r2_all_sessions = r2_score(all_real, all_predictions)
    daty = sorted(predictions_by_time.keys())
    real = [real_by_time[data] for data in daty]
    predictions_average = [np.mean(predictions_by_time[data]) for data in daty]
    mae_aggregated = mean_absolute_error(real, predictions_average)
    r2 = r2_score(real, predictions_average)

    print(f"\nDONE!")
    print(f"Executed {number_of_sessions} sessions of prediction for {prediction_hours} hours.")
    print(f"Total {len(all_predictions)} predictions.")
    print(f"MAE for all steps in all sessions: {mae_all_sessions:.2f} µg/m³")
    print(f"R2 for all steps in all sessions: {r2_all_sessions:.3f}")
    print(f"MAE by averaging overlapping predictions by time: {mae_aggregated:.2f} µg/m³")
    print(f"R² = {r2:.3f}")

    horizons = []
    r2_values = []
    mae_values = []

    print("\nMetrics by forecast horizon:")

    for horizon in range(1, prediction_hours + 1):
        y_true = horizon_results[horizon]["real"]
        y_pred = horizon_results[horizon]["pred"]

        mae_h = mean_absolute_error(y_true, y_pred)
        r2_h = r2_score(y_true, y_pred)

        horizons.append(horizon)
        r2_values.append(r2_h)
        mae_values.append(mae_h)

        print(f"+{horizon}h -> MAE: {mae_h:.2f} µg/m³, R²: {r2_h:.3f}")

    fig_r2, ax_r2 = plt.subplots(figsize=(10, 5))

    bg = '#070709'
    fig_r2.patch.set_facecolor(bg)
    ax_r2.set_facecolor(bg)

    ax_r2.plot(
        horizons,
        r2_values,
        marker='o',
        linewidth=2.5,
        color='#4ADE80',
    )

    ax_r2.axhline(
        0,
        color='white',
        alpha=0.15,
        linewidth=1,
    )

    ax_r2.grid(
        color='white',
        alpha=0.05,
        linewidth=0.8,
    )

    for spine in ax_r2.spines.values():
        spine.set_visible(False)

    ax_r2.tick_params(
        colors='#9AA4B2',
        labelsize=11,
        length=0,
    )

    ax_r2.set_xticks(horizons)

    ax_r2.set_xlabel(
        'Prediction horizon [h]',
        color='#9AA4B2',
        fontsize=12,
        labelpad=12,
    )

    ax_r2.set_ylabel(
        'R² score',
        color='#9AA4B2',
        fontsize=12,
        labelpad=12,
    )

    ax_r2.set_title(
        f'{index_name} - R² by prediction horizon',
        fontsize=18,
        color='#F0F0F0',
        pad=18,
    )

    for x, y in zip(horizons, r2_values):
        ax_r2.text(
            x,
            y,
            f'{y:.2f}',
            color='#D1D5DB',
            fontsize=10,
            ha='center',
            va='bottom',
        )

    plt.tight_layout()

    r2_output_path = f'verif_images/r2_by_horizon_{index_name}.png'
    plt.savefig(
        r2_output_path,
        dpi=300,
        bbox_inches='tight',
        facecolor=bg,
    )

    print(f"Saved R² horizon plot to '{r2_output_path}'")

    plt.show()

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif']

    plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(15, 7))

    bg = '#070709'
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    color_real = '#FF9F43'
    color_pred = '#4ADE80'

    ax.plot(
        daty,
        real,
        color=color_real,
        linewidth=6,
        alpha=0.1,
    )

    ax.plot(
        daty,
        real,
        color=color_real,
        linewidth=2.5,
        label='Real',
    )

    ax.plot(
        daty,
        predictions_average,
        color=color_pred,
        linewidth=1.5,
        linestyle='--',
        label='Forecast',
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(
        color='#FFFFFF',
        alpha=0.03,
        linewidth=0.8,
    )

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))

    ax.tick_params(
        colors='#7A8490',
        labelsize=11,
        length=0,
    )

    plt.setp(ax.get_xticklabels(), rotation=35, ha='right')

    ax.set_xlabel('Data', color='#9AA4B2', fontsize=12, labelpad=12)
    ax.set_ylabel(f'{index_name} (µg/m³)', color='#9AA4B2', fontsize=12, labelpad=12)

    ax.set_title(
        f'{index_name} - {prediction_hours}h recursive forecast',
        fontsize=22,
        color='#F0F0F0',
        weight='normal',
        pad=25,
    )

    exact_horizon = prediction_hours
    mae_exact_horizon = mae_values[exact_horizon - 1]
    r2_exact_horizon = r2_values[exact_horizon - 1]

    handles, labels = ax.get_legend_handles_labels()

    metric_labels = [
        f'Aggregated MAE = {mae_aggregated:.2f} µg/m³',
        f'Aggregated R² = {r2:.3f}',
        f'MAE +{exact_horizon}h = {mae_exact_horizon:.2f} µg/m³',
        f'R² +{exact_horizon}h = {r2_exact_horizon:.3f}',
    ]

    for metric_label in metric_labels:
        handles.append(LegendTextHandle())
        labels.append(metric_label)

    leg = ax.legend(
        handles=handles,
        labels=labels,
        handler_map={LegendTextHandle: LegendTextOnly()},
        facecolor='#101218',
        edgecolor='#2A3240',
        labelcolor='#E0E0E0',
        fontsize=12,
        loc='upper left',
        bbox_to_anchor=(0.01, 0.93),
        framealpha=0.95,
    )

    for metric_label in leg.get_texts()[-4:]:
        metric_label.set_color('#FFC48A')
        metric_label.set_fontweight('bold')
        metric_label.set_fontsize(13)

    plt.tight_layout()

    output_path = f'verif_images/verify_{index_name}_minimal.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=bg)
    print(f"Saved plot to '{output_path}'")

    plt.show()

if __name__ == '__main__':
    verify_model("models/exp/PM10_model.joblib", "data/old/test_PM10.csv", "PM10")