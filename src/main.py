import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from config import dataset_folder, datasets, figure_folder, interpolate_limit, draw_clean
from clean import check_data_quality, align_timestamps, plot_time_stamps, analyze_gap_distribution, plot_four_bars_in_one, plot_minute_around_missing_for_all_stages, remove_outliers_and_plot_box
from analysis import plot_hourly_heatmaps, analyze_appliance_lag, plot_log_aggregate_boxplot_by_appliance_activity, event_triggered_dual_average_plot, plot_cooccurrence_vs_window

def analyze_all_datasets():
    """Processes all datasets in the config"""
    for filename in datasets:
        filepath = os.path.join(dataset_folder, filename)
        figurepath = os.path.join(figure_folder, filename)
        try:
            df = pd.read_csv(filepath, parse_dates=['Time'])
            check_data_quality(df, filename)
            # plot_time_stamps(df, figurepath, "1_raw")

            aligned_df = align_timestamps(df)
            check_data_quality(aligned_df, filename)
            # plot_time_stamps(aligned_df, figurepath, "2_aligned", highlight_missing=True)
            
            analyze_gap_distribution(aligned_df, f"{figurepath}_gap_hist.pdf", filename)

            interpolated_df = aligned_df.interpolate(limit=interpolate_limit)
            check_data_quality(interpolated_df, filename)
            # plot_time_stamps(interpolated_df, figurepath, "3_interpolated", highlight_missing=True)
            
            # analyze_gap_distribution(interpolated_df, f"{figurepath}_gap_hist_interpolated.pdf", filename)
            
            dropna_df = interpolated_df.dropna()
            check_data_quality(dropna_df, filename)
            # plot_time_stamps(dropna_df, figurepath, "4_dropna")
            
            if draw_clean:
                plot_four_bars_in_one(
                    df_list=[dropna_df, interpolated_df, aligned_df, df],
                    labels=[
                        {"text": "4. Dropna"},
                        {"text": "3. Interpolated", "highlight_missing": True},
                        {"text": "2. Aligned", "highlight_missing": True},
                        {"text": "1. Raw"},
                    ],
                    figurepath=f"{figurepath}_compact_bars.png",
                    figurename=filename
                )
                
                if filename == 'house1.csv':
                    df_stages = {
                        "1. Raw": df,
                        "2. Aligned": aligned_df,
                        "3. Interpolated": interpolated_df,
                        "4. Dropna": dropna_df,
                    }
                    plot_minute_around_missing_for_all_stages(
                        df_dict=df_stages,
                        base_df=aligned_df,
                        figurepath=figurepath,
                        filename=f"{figurepath}_missing_minute_4stage.pdf"
                    )
            
            plot_hourly_heatmaps(dropna_df, filename, f"{figurepath}_hourly_heatmap.pdf")
            
            dropna_df['washingmachine_active'] = dropna_df['washingmachine'] > 10
            dropna_df['dishwasher_active'] = dropna_df['dishwasher'] > 10
            
            analyze_appliance_lag(
                dropna_df,
                target='dishwasher_active',
                other='washingmachine_active',
                output_path=f"{figurepath}_lag_dishwasher_to_washer.pdf",
                title_name=filename
            )
            
            plot_log_aggregate_boxplot_by_appliance_activity(
                dropna_df,
                output_path=f"{figurepath}_log_boxplot_by_activity.pdf"
            )
            
            event_triggered_dual_average_plot(
                df=dropna_df,
                trigger_cols=('dishwasher_active', 'washingmachine_active'),
                response_col='Aggregate',
                time_col='Time',
                window_minutes=60,
                step_minutes=1,
                output_path=f"{figurepath}_event_dual_avg.pdf",
                title_text=filename
            )

            plot_cooccurrence_vs_window(
                dropna_df,
                max_window=1440,
                step=60,
                output_path=f"{figurepath}_cooccurrence_vs_window.pdf",
                title_name=filename
            )

        except Exception as e:
            print(f"\033[91mError processing {filename}: {str(e)}\033[0m")
            # print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    analyze_all_datasets()
