import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from verify_direct import (
    create_direct_input,
    load_station_csv as load_direct_csv,
    prediction_hours as direct_prediction_hours,
    required_history,
)
from verify_models import (
    get_lag,
    history_hours,
    load_station_csv,
    max_lag,
    prediction_hours,
    prepend_context,
    session_start_indices,
)

BG = '#070709'


def collect_recursive_horizon_metrics(
    model_path,
    csv_path,
    index_name,
    context_csv_path=None,
):
    model = joblib.load(model_path)
    df_test = load_station_csv(csv_path, index_name)
    df_new = prepend_context(df_test, context_csv_path, index_name)

    horizon_results = {
        h: {'real': [], 'pred': []}
        for h in range(1, prediction_hours + 1)
    }

    for start_index in session_start_indices(len(df_new)):
        history_start = max(0, start_index - history_hours)
        history_index = df_new[index_name].iloc[history_start:start_index].tolist()

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

            horizon_results[horizon]['real'].append(real)
            horizon_results[horizon]['pred'].append(prediction)
            history_index.append(prediction)

    r2_values = []
    mae_values = []

    for horizon in range(1, prediction_hours + 1):
        y_true = horizon_results[horizon]['real']
        y_pred = horizon_results[horizon]['pred']
        mae_values.append(mean_absolute_error(y_true, y_pred))
        r2_values.append(r2_score(y_true, y_pred))

    return r2_values, mae_values


def collect_direct_horizon_metrics(models_dir, csv_path, index_name):
    df = load_direct_csv(csv_path, index_name)

    horizon_results = {
        h: {'real': [], 'pred': []}
        for h in range(1, direct_prediction_hours + 1)
    }

    models = {}
    for horizon in range(1, direct_prediction_hours + 1):
        model_path = os.path.join(models_dir, f'{index_name}_h{horizon}_model.joblib')
        models[horizon] = joblib.load(model_path)

    last_origin_index = len(df) - direct_prediction_hours - 1

    for origin_index in range(required_history - 1, last_origin_index + 1):
        history = df[index_name].iloc[
            origin_index - required_history + 1 : origin_index + 1
        ].tolist()

        for horizon in range(1, direct_prediction_hours + 1):
            target_index = origin_index + horizon
            target_time = df['Time'].iloc[target_index]
            real = df[index_name].iloc[target_index]

            input_data = create_direct_input(history, target_time, index_name)
            input_data = input_data[list(models[horizon].feature_names_in_)]

            prediction = max(0, models[horizon].predict(input_data)[0])

            horizon_results[horizon]['real'].append(real)
            horizon_results[horizon]['pred'].append(prediction)

    r2_values = []
    mae_values = []

    for horizon in range(1, direct_prediction_hours + 1):
        y_true = horizon_results[horizon]['real']
        y_pred = horizon_results[horizon]['pred']
        mae_values.append(mean_absolute_error(y_true, y_pred))
        r2_values.append(r2_score(y_true, y_pred))

    return r2_values, mae_values


def print_r2_table(horizon_labels, recursive_r2, direct_r2):
    header = ' ' * 12 + ''.join(f'{label:>8}' for label in horizon_labels)
    print(header)
    print(
        f"{'Recursive':<12}"
        + ''.join(f'{value:>8.2f}' for value in recursive_r2)
    )
    print(
        f"{'Direct':<12}"
        + ''.join(f'{value:>8.2f}' for value in direct_r2)
    )


def plot_r2_comparison(index_name, recursive_r2, direct_r2):
    horizons = list(range(1, len(recursive_r2) + 1))
    horizon_labels = [f'+{h}h' for h in horizons]

    heatmap_data = np.array([recursive_r2, direct_r2])

    fig, (ax_heat, ax_lines) = plt.subplots(
        2,
        1,
        figsize=(14, 10),
        gridspec_kw={'height_ratios': [1, 1.15]},
    )

    fig.patch.set_facecolor(BG)
    ax_heat.set_facecolor(BG)
    ax_lines.set_facecolor(BG)

    im = ax_heat.imshow(
        heatmap_data,
        aspect='auto',
        cmap='RdYlGn',
        vmin=-0.2,
        vmax=1.0,
    )

    ax_heat.set_xticks(range(len(horizons)))
    ax_heat.set_xticklabels(horizon_labels, color='#9AA4B2')
    ax_heat.set_yticks([0, 1])
    ax_heat.set_yticklabels(['Recursive', 'Direct'], color='#E0E0E0')

    for row in range(2):
        for col in range(len(horizons)):
            value = heatmap_data[row, col]
            ax_heat.text(
                col,
                row,
                f'{value:.2f}',
                ha='center',
                va='center',
                color='#000000',
                fontsize=11,
                fontweight='bold',
            )

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color='#9AA4B2')
    cbar.set_label('R2 score', color='#9AA4B2')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#9AA4B2')

    ax_heat.set_title(
        f'{index_name} - R2 heatmap: Recursive vs Direct',
        color='#F0F0F0',
        fontsize=18,
        pad=14,
    )

    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    ax_lines.plot(
        horizons,
        recursive_r2,
        marker='o',
        linewidth=2.5,
        color='#FF9F43',
        label='Recursive',
    )
    ax_lines.plot(
        horizons,
        direct_r2,
        marker='o',
        linewidth=2.5,
        color='#4ADE80',
        label='Direct',
    )

    ax_lines.axhline(0, color='white', alpha=0.15, linewidth=1)
    ax_lines.grid(color='white', alpha=0.05, linewidth=0.8)

    for spine in ax_lines.spines.values():
        spine.set_visible(False)

    ax_lines.tick_params(colors='#9AA4B2', labelsize=11, length=0)
    ax_lines.set_xticks(horizons)
    ax_lines.set_xlabel('Prediction horizon [h]', color='#9AA4B2', fontsize=12)
    ax_lines.set_ylabel('R2 score', color='#9AA4B2', fontsize=12)
    ax_lines.set_title(
        f'{index_name} - R2 by horizon',
        color='#F0F0F0',
        fontsize=18,
        pad=14,
    )

    ax_lines.legend(
        facecolor='#101218',
        edgecolor='#2A3240',
        labelcolor='#E0E0E0',
        fontsize=11,
        loc='upper right',
    )

    for x, y in zip(horizons, recursive_r2):
        ax_lines.text(x, y, f'{y:.2f}', color='#FF9F43', fontsize=9, ha='center', va='bottom')

    for x, y in zip(horizons, direct_r2):
        ax_lines.text(x, y, f'{y:.2f}', color='#4ADE80', fontsize=9, ha='center', va='top')

    plt.tight_layout()

    os.makedirs('verif_images', exist_ok=True)
    output_path = f'verif_images/compare_r2_{index_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=BG)
    print(f"Saved comparison plot to '{output_path}'")
    plt.show()


def compare_forecast_methods(
    recursive_model_path,
    direct_models_dir,
    csv_path,
    index_name,
    context_csv_path=None,
):
    if prediction_hours != direct_prediction_hours:
        raise ValueError(
            f'prediction_hours mismatch: recursive={prediction_hours}, '
            f'direct={direct_prediction_hours}'
        )

    print(f'Collecting recursive metrics for {index_name}...')
    recursive_r2, recursive_mae = collect_recursive_horizon_metrics(
        recursive_model_path,
        csv_path,
        index_name,
        context_csv_path,
    )

    print(f'Collecting direct metrics for {index_name}...')
    direct_r2, direct_mae = collect_direct_horizon_metrics(
        direct_models_dir,
        csv_path,
        index_name,
    )

    horizon_labels = [f'+{h}h' for h in range(1, prediction_hours + 1)]

    print('\nR2 comparison table:')
    print_r2_table(horizon_labels, recursive_r2, direct_r2)

    plot_r2_comparison(index_name, recursive_r2, direct_r2)

    return {
        'recursive_r2': recursive_r2,
        'recursive_mae': recursive_mae,
        'direct_r2': direct_r2,
        'direct_mae': direct_mae,
    }


if __name__ == '__main__':
    compare_forecast_methods(
        recursive_model_path='models/exp/PM10_model.joblib',
        direct_models_dir='models/direct',
        csv_path='data/old/test_PM10.csv',
        index_name='PM10',
    )
