import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from matplotlib.gridspec import GridSpec
import matplotlib
matplotlib.use('TkAgg')


def remove_outliers_percentile(df, column, lower=0.01, upper=0.90):
    lower_bound = df[column].quantile(lower)
    upper_bound = df[column].quantile(upper)
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def plot_delay_distributions(data_path, delay_types=['ARR_DELAY', 'DEP_DELAY'], figsize=(12, 8)):
    """
    绘制航班延误的直方图与正态分布拟合曲线

    参数:
        data_path (str): 数据文件路径
        delay_types (list): 需分析的延误类型列表
        figsize (tuple): 图像尺寸
    """
    # 1. 数据加载与清洗
    try:
        df = pd.read_csv(data_path)
        df = remove_outliers_percentile(df, 'ARR_DELAY')
        df = remove_outliers_percentile(df, 'DEP_DELAY')
        print(f"数据加载成功，记录数: {len(df)}")
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return

    # 2. 创建画布
    plt.figure(figsize=figsize)
    gs = GridSpec(len(delay_types), 1, figure=plt.gcf())

    # 3. 统一可视化样式
    hist_kwargs = {
        'bins': 50,
        'alpha': 0.6,
        'stat': 'density',  # 修改点：将density改为stat='density'
        'edgecolor': 'k',
        'linewidth': 0.5
    }
    plot_kwargs = {
        'linewidth': 2,
        'linestyle': '-'
    }
    colors = ['steelblue', 'forestgreen']

    # 4. 对每种延误类型绘图
    for i, delay_type in enumerate(delay_types):
        ax = plt.subplot(gs[i])

        # 数据清洗
        delay_data = df[delay_type].dropna()
        delay_data = delay_data[(delay_data >= -60) & (delay_data <= 240)]  # 限制在-1~4小时

        if len(delay_data) == 0:
            print(f"警告: {delay_type} 无有效数据")
            continue

        # 绘制直方图
        sns.histplot(delay_data, ** hist_kwargs,
        color=colors[i], label='Empirical Distribution', ax = ax)

        # 正态分布拟合
        mu, std = delay_data.mean(), delay_data.std()
        x = np.linspace(delay_data.min(), delay_data.max(), 100)
        pdf = norm.pdf(x, mu, std)
        ax.plot(x, pdf, color='darkred',** plot_kwargs,
        label = f'Normal Fit\nμ={mu:.1f}, σ={std:.1f}')

        # 样式设置
        ax.set_title(f'{delay_type} Distribution_2020', fontsize=14, pad=10)
        ax.set_xlabel('Delay (minutes)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, linestyle=':', alpha=0.5)

        # 标注数据量
        ax.text(0.98, 0.95, f'N={len(delay_data):,}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.8))

    # 5. 保存和显示
    plt.tight_layout()
    plt.savefig('flight_delays_2020.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print("可视化完成，图像已保存为 flight_delays_2024.pdf")


# 使用示例
plot_delay_distributions(
    data_path='D:\project\Delay_data\Datasets\data\\flight_with_weather_2020.csv',
    delay_types=['ARR_DELAY', 'DEP_DELAY']
)