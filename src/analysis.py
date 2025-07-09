import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.dates as mdates
from datetime import timedelta
import os
from typing import Optional

def plot_hourly_heatmaps(df, filename, figurepath):
    """绘制三个 value 列的小时热力图（日期 × 小时，平均功率），扁平样式 + 稀疏坐标 + 左上角标题"""
    df = df.copy()
    df.set_index('Time', inplace=True)

    value_cols = ['Aggregate', 'dishwasher', 'washingmachine']
    hourly_mean = df[value_cols].resample('1H').mean()
    hourly_mean['Date'] = hourly_mean.index.date
    hourly_mean['Hour'] = hourly_mean.index.hour

    pivot_tables = {
        col: hourly_mean.pivot_table(index='Hour', columns='Date', values=col)
        for col in value_cols
    }

    fig, axs = plt.subplots(len(value_cols), 1, figsize=(12, 5), sharex=True)

    for i, (ax, col) in enumerate(zip(axs, value_cols)):
        data = pivot_tables[col]
        im = ax.imshow(
            data,
            aspect='auto',
            origin='lower',
            # cmap='YlOrRd'
            # cmap='viridis'
            # cmap='plasma'
            # cmap='Blues'
            cmap='PuBuGn'
            # cmap='turbo'
        )

        # 横轴稀疏：每隔 5 个日期标一个
        xticks = range(0, len(data.columns), max(1, len(data.columns)//10))
        xticklabels = [data.columns[i].strftime('%m-%d') for i in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=45, fontsize=7)

        # 纵轴稀疏：每 3 小时标一个
        yticks = range(0, 24, 3)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontsize=7)

        # 左上角标题
        ax.text(0.01, 0.98, col, transform=ax.transAxes,
                fontsize=9, va='top', ha='left')

        # 色条
        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.01)

    axs[-1].set_xlabel('Date', fontsize=9)
    axs[0].set_ylabel('Hour', fontsize=9)

    # fig.suptitle(f"{filename} - Hourly Power Heatmap", fontsize=11)
    # plt.subplots_adjust(hspace=0.3)
    plt.tight_layout()
    plt.savefig(figurepath, dpi=300)
    plt.close()
    
def analyze_appliance_lag(df, time_col='Time', target='dishwasher_active', other='washingmachine_active', window_minutes=60, step_minutes=1, output_path=None, title_name=None):
    """
    分析 target 激活前后 other 的激活频率（±window_minutes），步长为 step_minutes。
    如果提供 output_path，则保存图像；否则直接展示。
    title_name: 用作左上角标题，适合用于拼图。
    """
    df = df.copy()
    df = df.sort_values(time_col)
    df = df.set_index(time_col)

    target_times = df[df[target]].index
    lags = list(range(-window_minutes, window_minutes + 1, step_minutes))
    activation_counts = []

    for lag in lags:
        shifted_times = target_times + pd.Timedelta(minutes=lag)
        shifted_times = shifted_times[shifted_times.isin(df.index)]
        count = df.loc[shifted_times, other].sum()
        activation_counts.append(count)

    # # 绘图（优化后用于拼图）
    # plt.figure(figsize=(3.5, 2.5))
    # plt.plot(lags, activation_counts, marker='o', markersize=2)
    # plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    # plt.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)

    # # 左上角简洁标题
    # if title_name:
    #     plt.text(0.01, 0.95, title_name, transform=plt.gca().transAxes,
    #              fontsize=9, ha='left', va='top')

    # # 更简洁坐标轴
    # plt.xlabel("Time lag (min) relative to d.w. activation", fontsize=8)
    # # if title_name == 'house1.csv' or title_name == 'house4.csv':
    # plt.ylabel("W.m. activate ount", fontsize=8)
    # plt.xticks(fontsize=7)
    # plt.yticks(fontsize=7)

    # plt.tight_layout()
    # plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # plt.close()
    
    # 绘图：柱状图替代折线图
    plt.figure(figsize=(3.5, 2.5))
    plt.bar(lags, activation_counts, width=step_minutes, color='steelblue', edgecolor='black')
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.grid(True, linestyle='--', linewidth=0.3, alpha=0.5, axis='y')

    # 左上角简洁标题
    if title_name:
        plt.text(0.01, 0.95, title_name, transform=plt.gca().transAxes,
                 fontsize=9, ha='left', va='top')

    # 坐标轴样式
    plt.xlabel("Time lag (min) relative to d.w. activation", fontsize=8)
    plt.ylabel("W.m. activate count", fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return pd.DataFrame({'Lag(min)': lags, 'Activation Count': activation_counts})


def plot_log_aggregate_boxplot_by_appliance_activity(df, output_path=None):
    """
    根据洗衣机和洗碗机的活跃状态，将总功耗分成4类，绘制 log10 总功耗箱线图。
    横轴分类：None Active, Washer Active, Dishwasher Active, Both Active
    纵轴为 Aggregate 的 log10 值。
    """
    df = df.copy()
    # 分类编码
    def activity_category(row):
        wm = row['washingmachine_active']
        dw = row['dishwasher_active']
        if wm and dw:
            return "Both Active"
        elif wm:
            return "Washer Active"
        elif dw:
            return "Dishwasher Active"
        else:
            return "None Active"

    df['activity_category'] = df.apply(activity_category, axis=1)
    df = df[df['Aggregate'] > 0]  # 避免 log(0) 错误
    df['log_aggregate'] = np.log10(df['Aggregate'])

    order = ["None Active", "Washer Active", "Dishwasher Active", "Both Active"]
    fig, ax = plt.subplots(figsize=(5, 4))
    df.boxplot(column='log_aggregate', by='activity_category', ax=ax, grid=False, patch_artist=True,
               boxprops=dict(facecolor='skyblue', color='black'),
               medianprops=dict(color='red'),
               flierprops=dict(marker='o', markersize=3, linestyle='none', markerfacecolor='gray'),
               showfliers=True)

    ax.set_xlabel("Appliance Activity")
    ax.set_ylabel("log10(Power)")
    ax.set_title("Log Power by Appliance Activity")
    plt.suptitle("")  # 移除默认标题
    ax.set_xticklabels(order, rotation=15)

    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_log_aggregate_boxplot_by_appliance_activity(df, output_path=None):
    """
    根据 appliance 激活组合分类绘制 log10(Aggregate) 的箱线图。
    横轴为 activity_category（共 4 类），纵轴为 log10(Power)。
    """
    df = df.copy()

    # 分类
    def activity_category(row):
        wm = row['washingmachine_active']
        dw = row['dishwasher_active']
        if wm and dw:
            return "Both Active"
        elif wm:
            return "Washer Active"
        elif dw:
            return "Dishwasher Active"
        else:
            return "None Active"

    df['activity_category'] = df.apply(activity_category, axis=1)

    # 仅保留正功率值以取 log
    df = df[df['Aggregate'] > 0].copy()
    df['log_aggregate'] = np.log10(df['Aggregate'])

    # 设定分类顺序，并根据实际数据筛选
    order = ["None Active", "Washer Active", "Dishwasher Active", "Both Active"]
    present_categories = [cat for cat in order if cat in df['activity_category'].unique()]

    # 绘图
    fig, ax = plt.subplots(figsize=(4, 3))
    df.boxplot(column='log_aggregate', by='activity_category', ax=ax, grid=False, patch_artist=True,
               boxprops=dict(facecolor='skyblue', color='black'),
               medianprops=dict(color='red'),
               flierprops=dict(marker='o', markersize=3, linestyle='none', markerfacecolor='gray'),
               showfliers=True)

    # 设置轴标签
    ax.set_xlabel("Appliance Activity", fontsize=9)
    ax.set_ylabel("log10(Power)", fontsize=9)
    ax.set_title("Log Power by Appliance Activity", fontsize=11)
    plt.suptitle("")  # 去掉 pandas 自带的标题

    # 设置横轴标签（实际出现的类，保持顺序）
    ax.set_xticklabels(present_categories, rotation=15, fontsize=8)
    ax.tick_params(axis='y', labelsize=8)

    # 保存或展示
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

def event_triggered_average_plot(df: pd.DataFrame,
                                 trigger_col: str = 'dishwasher_active',
                                 response_col: str = 'Aggregate',
                                 time_col: str = 'Time',
                                 window_minutes: int = 60,
                                 step_minutes: int = 1,
                                 output_path: Optional[str] = None,
                                 title_text: Optional[str] = None):
    """
    事件驱动窗口分析：
    围绕每一次 trigger_col 为 True 的时刻，观察 response_col 的平均响应变化。
    """
    df = df.copy()
    df = df.sort_values(time_col).set_index(time_col)
    df = df[[trigger_col, response_col]].dropna()

    triggered_times = df[df[trigger_col]].index

    aligned_curves = []

    for t in triggered_times:
        window_range = pd.date_range(t - timedelta(minutes=window_minutes),
                                     t + timedelta(minutes=window_minutes),
                                     freq=f'{step_minutes}min')
        segment = df.reindex(window_range)[response_col]

        if segment.isnull().any():
            continue
        aligned_curves.append(segment.values)

    if not aligned_curves:
        raise ValueError("No valid segments found for averaging.")

    aligned_curves = np.array(aligned_curves)
    mean_curve = aligned_curves.mean(axis=0)
    lags = np.arange(-window_minutes, window_minutes + 1, step_minutes)

    # 绘图
    plt.figure(figsize=(6, 3))
    plt.plot(lags, mean_curve, color='darkblue')
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel(f'Time relative to {trigger_col} (min)', fontsize=9)
    plt.ylabel(response_col, fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)

    if title_text:
        plt.text(0.01, 0.97, title_text, transform=plt.gca().transAxes,
                 fontsize=9, va='top', ha='left')

    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def event_triggered_dual_average_plot(df, trigger_cols=('dishwasher_active', 'washingmachine_active'), response_col='Aggregate', time_col='Time', window_minutes=60, step_minutes=1, output_path=None, title_text=None):
    """
    比较两个 trigger 对同一 response 的事件驱动平均响应曲线。
    在一张图中画出两条响应曲线。
    """
    df = df.copy()
    df = df.sort_values(time_col).set_index(time_col)

    lags = list(range(-window_minutes, window_minutes + 1, step_minutes))

    def compute_avg_response(trigger_col):
        trigger_times = df[df[trigger_col]].index
        response_curves = []

        for t in trigger_times:
            window_range = df.loc[t - pd.Timedelta(minutes=window_minutes):
                                  t + pd.Timedelta(minutes=window_minutes)]
            # if len(window_range) < len(lags):  # 不完整窗口
            #     continue
            if window_range.empty:
                continue
            response = window_range[response_col].reindex(
                pd.date_range(t - pd.Timedelta(minutes=window_minutes),
                            t + pd.Timedelta(minutes=window_minutes),
                            freq=f'{step_minutes}min'),
                method='nearest'
            ).values
            response_curves.append(response)

            response = window_range[response_col].values
            if len(response) != len(lags):
                continue
            response_curves.append(response)

        if not response_curves:
            return np.full(len(lags), np.nan)
        return np.nanmean(response_curves, axis=0)

    avg1 = compute_avg_response(trigger_cols[0])
    avg2 = compute_avg_response(trigger_cols[1])

    # 绘图
    plt.figure(figsize=(4, 2.8))
    plt.plot(lags, avg1, label=trigger_cols[0], color='tab:blue', marker='o', markersize=2)
    plt.plot(lags, avg2, label=trigger_cols[1], color='tab:orange', marker='o', markersize=2)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)
    if title_text:
        plt.text(0.01, 0.95, title_text, transform=plt.gca().transAxes,
                 fontsize=9, ha='left', va='top')
    plt.xlabel(f"Time lag (min)", fontsize=8)
    plt.ylabel(f"{response_col} (mean)", fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.legend(fontsize=7, loc='upper right')

    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return pd.DataFrame({
        'Lag(min)': lags,
        f'{trigger_cols[0]} avg': avg1,
        f'{trigger_cols[1]} avg': avg2
    })

def plot_cooccurrence_vs_window(df, time_col='Time',
                                washer_col='washingmachine_active',
                                dishwasher_col='dishwasher_active',
                                max_window=60, step=5,
                                output_path=None, title_name=None):
    """
    绘制不同时间窗口（±window）下的洗衣机和洗碗机共现率曲线。
    """
    df = df.copy()
    df = df.sort_values(time_col)
    df.set_index(time_col, inplace=True)

    washer_times = df[df[washer_col]].index
    dishwasher_times = df[df[dishwasher_col]].index

    windows = list(range(0, max_window + 1, step))
    washer_ratios = []
    dishwasher_ratios = []

    for w in windows:
        delta = pd.Timedelta(minutes=w)

        cooccur_washer = sum(
            ((dishwasher_times >= t - delta) & (dishwasher_times <= t + delta)).any()
            for t in washer_times
        )
        cooccur_dishwasher = sum(
            ((washer_times >= t - delta) & (washer_times <= t + delta)).any()
            for t in dishwasher_times
        )

        washer_ratio = cooccur_washer / len(washer_times) if len(washer_times) > 0 else 0
        dishwasher_ratio = cooccur_dishwasher / len(dishwasher_times) if len(dishwasher_times) > 0 else 0

        washer_ratios.append(washer_ratio)
        dishwasher_ratios.append(dishwasher_ratio)

    # 绘图
    plt.figure(figsize=(3.6, 2.6))
    plt.plot(windows, washer_ratios, marker='o', label='Washer→Dishwasher')
    plt.plot(windows, dishwasher_ratios, marker='s', label='Dishwasher→Washer')
    plt.xlabel('Time window ±Δt (minutes)', fontsize=9)
    plt.ylabel('Co-occurrence rate', fontsize=9)
    if title_name:
        plt.text(0.01, 0.95, title_name, transform=plt.gca().transAxes,
                 fontsize=9, ha='left', va='top')

    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', linewidth=0.3, alpha=0.6)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(fontsize=7, loc='upper right')

    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return pd.DataFrame({
        'window_minutes': windows,
        'washer_ratio': washer_ratios,
        'dishwasher_ratio': dishwasher_ratios
    })
