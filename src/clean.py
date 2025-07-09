import pandas as pd
from config import dataset_folder, datasets
import matplotlib.pyplot as plt
import os
import numpy as np

def check_data_quality(df, dataset_name):
    print(f"\n=== Data Quality Report: {dataset_name} ===")

    missing = df.isnull().sum().sum()
    print(f"Missing Values: {'YES' if missing > 0 else 'NO'} ({missing} total)")

    duplicates = df['Time'].duplicated().sum()
    print(f"Duplicate Timestamps: {'YES' if duplicates > 0 else 'NO'} ({duplicates} found)")

    numeric_cols = df.select_dtypes(include='number').columns
    negatives = (df[numeric_cols] < 0).sum().sum()
    print(f"Negative Values: {'YES' if negatives > 0 else 'NO'} ({negatives} found)")

    if {'Aggregate', 'dishwasher', 'washingmachine'}.issubset(df.columns):
        bad_rows = (df['Aggregate'] > df['dishwasher'] + df['washingmachine']).sum()
        print(f"Aggregate Inconsistency: {'YES' if bad_rows > 0 else 'NO'} ({bad_rows} rows)")
    else:
        print("Aggregate Inconsistency: SKIPPED (columns missing)")


def align_timestamps(df, target_freq='15S'):
    """Align timestamps to a standard frequency grid"""
    resampled_df = df.copy()
    
    # Ensure Time is datetime type and set as index
    resampled_df['Time'] = pd.to_datetime(resampled_df['Time'])
    resampled_df = resampled_df.set_index('Time')
    
    # Resample using mean aggregation
    resampled_df = resampled_df.resample(target_freq).mean()
    
    # Forward fill missing values
    # resampled_df = resampled_df.interpolate()
    # resampled_df = resampled_df.ffill()
    
    # Reset index to make Time a column again
    resampled_df = resampled_df.reset_index()
    
    return resampled_df
    
def analyze_all_datasets():
    """Processes all datasets in the config"""
    for filename in datasets:
        filepath = os.path.join(dataset_folder, filename)
        try:
            df = pd.read_csv(filepath, parse_dates=['Time'])
            check_data_quality(df, filename)
            print(df)
            
            aligned_df = align_timestamps(df)
            check_data_quality(aligned_df, filename)
            print(aligned_df)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

def plot_time_stamps(df, filename_prefix, output_name, highlight_missing=False):
    """绘制时间戳的一维透明散点图"""
    plt.figure(figsize=(10, 1))

    times = df['Time']
    y = [0] * len(times)

    if highlight_missing:
        is_missing = df.isna().any(axis=1)
        plt.scatter(times[~is_missing], [0]*sum(~is_missing), alpha=0.2, color='black')
        plt.scatter(times[is_missing], [0]*sum(is_missing), alpha=0.6, color='red')
    else:
        plt.scatter(times, y, alpha=0.2, color='black')

    plt.yticks([])
    # plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_{output_name}.png")
    plt.close()
    
def plot_four_bars_in_one(df_list, labels, figurepath, figurename):
    """在一个坐标轴里画多个时间戳条形图"""
    plt.figure(figsize=(10, 2))
    ax = plt.gca()

    for i, (df, label) in enumerate(zip(df_list, labels), 1):
        times = df['Time']
        y = [i] * len(times)

        if 'highlight_missing' in label and label['highlight_missing']:
            is_missing = df.isna().any(axis=1)
            ax.scatter(times[~is_missing], [i]*sum(~is_missing), alpha=0.2, color='#1f77b4')
            ax.scatter(times[is_missing], [i]*sum(is_missing), alpha=0.5, color='#ff7f0e')
        else:
            ax.scatter(times, y, alpha=0.2, color='#1f77b4')
            
        ax.text(-0.01, i, label['text'], transform=ax.get_yaxis_transform(), va='center', ha='right', fontsize=9)

    # ax.set_title(figurename)
    ax.set_yticks([])
    # ax.set_xlabel("Time")
    # ax.set_xlabel(figurename)
    ax.set_ylim(0.5, len(df_list) + 0.5)
    plt.tight_layout()
    plt.savefig(figurepath)
    plt.close()

    
def analyze_gap_distribution(df, filename_prefix, figurename):
    """分析缺失 gap 的长度并绘制直方图，并打印统计信息"""
    is_missing = df.isna().any(axis=1).to_numpy()

    gap_lengths = []
    current_gap = 0

    for missing in is_missing:
        if missing:
            current_gap += 1
        else:
            if current_gap > 0:
                gap_lengths.append(current_gap)
                current_gap = 0
    if current_gap > 0:
        gap_lengths.append(current_gap)

    if gap_lengths:
        gap_array = np.array(gap_lengths)

        # 打印五数概括
        print(f"Gap Stats for {filename_prefix}:")
        print(f"  Count: {len(gap_array)}")
        print(f"  Min  : {gap_array.min()}")
        print(f"  25%  : {np.percentile(gap_array, 25)}")
        print(f"  50%  : {np.percentile(gap_array, 50)}")
        print(f"  75%  : {np.percentile(gap_array, 75)}")
        print(f"  Max  : {gap_array.max()}")

        # 绘制直方图
        plt.figure(figsize=(4, 3))
        plt.hist(gap_array, bins=30, color='#1f77b4', edgecolor='black')
        plt.xlabel("Gap Size (number of timestamps)")
        plt.ylabel("Frequency")
        # plt.title(f"Gap Distribution: {figurename}")
        plt.tight_layout()
        plt.savefig(filename_prefix)
        plt.close()
    else:
        print(f"No gaps found in {filename_prefix}")
    
def plot_minute_around_missing_for_all_stages(df_dict, base_df, figurepath, filename):
    """
    查找 base_df 中第一个缺失值的一分钟窗口，绘制各阶段 df 的三列曲线对比图（1行4列）。
    输出为 PDF。
    """
    import matplotlib.dates as mdates

    missing_rows = base_df[base_df.isna().any(axis=1)]
    if missing_rows.empty:
        print(f"\033[93mNo missing values found in base_df for {filename}\033[0m")
        return

    missing_time = pd.to_datetime(missing_rows['Time'].iloc[0])
    start_time = missing_time - pd.Timedelta(seconds=30)
    end_time = missing_time + pd.Timedelta(seconds=30)

    fig, axs = plt.subplots(1, 4, figsize=(13, 3), sharey=True)
    stage_names = list(df_dict.keys())

    for i, stage_name in enumerate(stage_names):
        stage_df = df_dict[stage_name]
        if 'Time' not in stage_df.columns:
            print(f"\033[93m{stage_name} has no 'Time' column.\033[0m")
            continue

        df_window = stage_df[
            (stage_df['Time'] >= start_time) & (stage_df['Time'] <= end_time)
        ].copy()

        if df_window.empty:
            print(f"\033[93m{stage_name} has no data in {start_time} ~ {end_time}.\033[0m")
            continue

        df_window.set_index('Time', inplace=True)
        ax = axs[i]
        for col in ['Aggregate', 'dishwasher', 'washingmachine']:
            if col in df_window.columns:
                ax.plot(df_window.index, df_window[col], label=col)

        ax.set_title(stage_name)
        ax.tick_params(axis='x', rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        if i == 0:
            ax.set_ylabel("Power")

        # 添加红线标记缺失点时间
        ax.axvline(missing_time, color='red', linestyle='--', alpha=0.7)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', ncol=3, bbox_to_anchor=(0.511, 0.975))

    fig.suptitle(
        f"house1 - Values around missing time @ {missing_time.strftime('%Y-%m-%d %H:%M:%S')}",
        fontsize=16, y=1.1
    )
    fig.subplots_adjust(wspace=0.2, bottom=0.2, top=0.85)

    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def remove_outliers_and_plot_box(df, cols, quantile, output_path, filename=None):
    """剔除异常值并画箱线图，保存到指定路径"""
    df_original = df[cols].copy()
    df_clean = df.copy()

    for col in cols:
        if col in df_clean.columns:
            upper = df_clean[col].quantile(quantile)
            df_clean = df_clean[df_clean[col] <= upper]

    df_after = df_clean[cols]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    df_original.boxplot(column=cols, ax=axes[0])
    axes[0].set_title("Before Outlier Removal")
    df_after.boxplot(column=cols, ax=axes[1])
    axes[1].set_title("After Outlier Removal")

    if filename:
        fig.suptitle(f"{filename} - Outlier Filtering (q={quantile})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return df_clean



# if __name__ == "__main__":
#     analyze_all_datasets()
