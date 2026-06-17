import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle
from sklearn.metrics import mean_absolute_error, r2_score

prediction_hours = 10
required_history = 48


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

    df = df.dropna(subset=["Time", index_name])
    df["Time"] = pd.to_datetime(df["Time"])

    df.sort_values("Time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def get_lag_direct(history, lag):
    if len(history) >= lag:
        return history[-lag]

    return history[0]


def create_direct_input(history, target_time, index_name):
    """
    Creates one input row for direct forecasting.

    history contains real values up to the prediction start time.
    target_time is the future time for which the model predicts the value.
    """
    hour = target_time.hour
    dayofweek = target_time.dayofweek
    month = target_time.month
    dayofyear = target_time.dayofyear

    input_data = pd.DataFrame([{
        "hour": hour,
        "dayofweek": dayofweek,
        "month": month,
        "dayofyear": dayofyear,

        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),

        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),

        f"{index_name}_lag_1": get_lag_direct(history, 1),
        f"{index_name}_lag_2": get_lag_direct(history, 2),
        f"{index_name}_lag_3": get_lag_direct(history, 3),
        f"{index_name}_lag_6": get_lag_direct(history, 6),
        f"{index_name}_lag_12": get_lag_direct(history, 12),
        f"{index_name}_lag_24": get_lag_direct(history, 24),
        f"{index_name}_lag_48": get_lag_direct(history, 48),
    }])

    return input_data


def verify_direct_models(models_dir, csv_path, index_name):
    df = load_station_csv(csv_path, index_name)

    horizon_results = {
        h: {"time": [], "real": [], "pred": []}
        for h in range(1, prediction_hours + 1)
    }

    predictions_by_time = {}
    real_by_time = {}

    models = {}

    for horizon in range(1, prediction_hours + 1):
        model_path = os.path.join(
            models_dir,
            f"{index_name}_h{horizon}_model.joblib",
        )

        models[horizon] = joblib.load(model_path)
        print(f"Loaded model: {model_path}")

    last_origin_index = len(df) - prediction_hours - 1

    for origin_index in range(required_history - 1, last_origin_index + 1):
        history = df[index_name].iloc[
            origin_index - required_history + 1 : origin_index + 1
        ].tolist()

        for horizon in range(1, prediction_hours + 1):
            model = models[horizon]

            target_index = origin_index + horizon
            target_time = df["Time"].iloc[target_index]
            real = df[index_name].iloc[target_index]

            input_data = create_direct_input(history, target_time, index_name)
            input_data = input_data[list(model.feature_names_in_)]

            prediction = model.predict(input_data)[0]
            prediction = max(0, prediction)

            horizon_results[horizon]["time"].append(target_time)
            horizon_results[horizon]["real"].append(real)
            horizon_results[horizon]["pred"].append(prediction)

            real_by_time[target_time] = real
            predictions_by_time.setdefault(target_time, []).append(prediction)

    horizons = []
    mae_values = []
    r2_values = []

    print("\nMetrics by forecast horizon:")

    for horizon in range(1, prediction_hours + 1):
        y_true = horizon_results[horizon]["real"]
        y_pred = horizon_results[horizon]["pred"]

        mae_h = mean_absolute_error(y_true, y_pred)
        r2_h = r2_score(y_true, y_pred)

        horizons.append(horizon)
        mae_values.append(mae_h)
        r2_values.append(r2_h)

        print(f"+{horizon}h -> MAE: {mae_h:.2f} µg/m³, R²: {r2_h:.3f}")

    all_real = []
    all_predictions = []

    for horizon in range(1, prediction_hours + 1):
        all_real.extend(horizon_results[horizon]["real"])
        all_predictions.extend(horizon_results[horizon]["pred"])

    r2_all_horizons = r2_score(all_real, all_predictions)
    mae_all_horizons = mean_absolute_error(all_real, all_predictions)

    print(f"MAE for all direct horizons: {mae_all_horizons:.2f} µg/m³")
    print(f"R² for all direct horizons: {r2_all_horizons:.3f}")
    print(f"R² = {r2_all_horizons:.3f}")

    daty = sorted(predictions_by_time.keys())

    real_aggregated = [
        real_by_time[time]
        for time in daty
    ]

    predictions_average = [
        np.mean(predictions_by_time[time])
        for time in daty
    ]

    mae_aggregated = mean_absolute_error(real_aggregated, predictions_average)
    r2_aggregated = r2_score(real_aggregated, predictions_average)

    exact_horizon = prediction_hours
    mae_exact_horizon = mae_values[exact_horizon - 1]
    r2_exact_horizon = r2_values[exact_horizon - 1]

    print(f"MAE by averaging overlapping direct predictions by time: {mae_aggregated:.2f} µg/m³")
    print(f"R² by averaging overlapping direct predictions by time: {r2_aggregated:.3f}")

    plot_r2_by_horizon(index_name, horizons, r2_values)
    plot_aggregated_direct_forecast(
        index_name,
        daty,
        real_aggregated,
        predictions_average,
        mae_aggregated,
        r2_aggregated,
        exact_horizon,
        mae_exact_horizon,
        r2_exact_horizon,
    )


def plot_r2_by_horizon(index_name, horizons, r2_values):
    bg = "#070709"

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    ax.plot(
        horizons,
        r2_values,
        marker="o",
        linewidth=2.5,
        color="#4ADE80",
    )

    ax.axhline(0, color="white", alpha=0.15, linewidth=1)

    ax.grid(color="white", alpha=0.05, linewidth=0.8)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(colors="#9AA4B2", labelsize=11, length=0)

    ax.set_xticks(horizons)

    ax.set_xlabel("Prediction horizon [h]", color="#9AA4B2", fontsize=12)
    ax.set_ylabel("R2 score", color="#9AA4B2", fontsize=12)

    ax.set_title(
        f"{index_name} - direct forecast R2 by horizon",
        fontsize=18,
        color="#F0F0F0",
        pad=18,
    )

    for x, y in zip(horizons, r2_values):
        ax.text(
            x,
            y,
            f"{y:.2f}",
            color="#D1D5DB",
            fontsize=10,
            ha="center",
            va="bottom",
        )

    os.makedirs("verif_images", exist_ok=True)

    output_path = f"verif_images/direct_r2_by_horizon_{index_name}.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=bg)

    print(f"Saved R² plot to '{output_path}'")

    plt.show()


def plot_aggregated_direct_forecast(
    index_name,
    times,
    real,
    predictions_average,
    mae,
    r2,
    exact_horizon,
    mae_exact_horizon,
    r2_exact_horizon,
):
    bg = "#070709"

    fig, ax = plt.subplots(figsize=(15, 7))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    ax.plot(
        times,
        real,
        color="#FF9F43",
        linewidth=2.5,
        label="Real",
    )

    ax.plot(
        times,
        predictions_average,
        color="#4ADE80",
        linewidth=1.8,
        linestyle="--",
        label=f"Average direct forecast 1-{prediction_hours}h",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(color="white", alpha=0.04, linewidth=0.8)

    ax.tick_params(colors="#9AA4B2", labelsize=10, length=0)

    ax.set_xlabel("Date", color="#9AA4B2", fontsize=12)
    ax.set_ylabel(f"{index_name} (µg/m³)", color="#9AA4B2", fontsize=12)

    ax.set_title(
        f"{index_name} - aggregated direct forecast window 1-{prediction_hours}h",
        fontsize=20,
        color="#F0F0F0",
        pad=20,
    )

    handles, labels = ax.get_legend_handles_labels()

    metric_labels = [
        f"Aggregated MAE = {mae:.2f} µg/m³",
        f"Aggregated R² = {r2:.3f}",
        f"MAE +{exact_horizon}h = {mae_exact_horizon:.2f} µg/m³",
        f"R² +{exact_horizon}h = {r2_exact_horizon:.3f}",
    ]

    for metric_label in metric_labels:
        handles.append(LegendTextHandle())
        labels.append(metric_label)

    leg = ax.legend(
        handles=handles,
        labels=labels,
        handler_map={LegendTextHandle: LegendTextOnly()},
        facecolor="#101218",
        edgecolor="#2A3240",
        labelcolor="#E0E0E0",
        fontsize=12,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.93),
        framealpha=0.95,
    )

    for metric_label in leg.get_texts()[-4:]:
        metric_label.set_color("#FFC48A")
        metric_label.set_fontweight("bold")
        metric_label.set_fontsize(13)

    os.makedirs("verif_images", exist_ok=True)

    output_path = f"verif_images/direct_{index_name}_aggregated.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=bg)

    print(f"Saved aggregated direct forecast plot to '{output_path}'")

    plt.show()


if __name__ == "__main__":
    verify_direct_models(
        models_dir="models/direct",
        csv_path="data/old/test_PM10.csv",
        index_name="PM10",
    )
