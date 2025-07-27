import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

def plot_monthly_dep_delay(data_path):
    """
    绘制2020年每月平均出发延误（DEP_DELAY）的折线图

    参数:
        data_path (str): 数据文件路径，需包含 'FL_DATE' 和 'DEP_DELAY' 列
    """
    try:
        df = pd.read_csv(data_path, parse_dates=['FL_DATE'])
        df = df[df['FL_DATE'].dt.year == 2020]  # 只保留2020年的数据
        df = df[['FL_DATE', 'ARR_DELAY']].dropna()
        df = df[(df['ARR_DELAY'] >= -60) & (df['ARR_DELAY'] <= 240)]  # 限制异常值
        df['Month'] = df['FL_DATE'].dt.month
        monthly_delay = df.groupby('Month')['ARR_DELAY'].mean().reset_index()
        print("成功计算月平均延误")
    except Exception as e:
        print(f"数据处理失败: {str(e)}")
        return

    # 绘图
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=monthly_delay, x='Month', y='ARR_DELAY', marker='o', color='steelblue', linewidth=2)
    plt.title('Monthly Average Arrive Delay in 2020', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Average ARR_DELAY (minutes)', fontsize=12)
    plt.xticks(ticks=range(1,13))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('monthly_arr_delay_2020.pdf', dpi=300)
    plt.show()
    print("图像保存为 monthly_arr_delay_2020.pdf")

# 示例调用（注意替换为你的 2020 年数据路径）
plot_monthly_dep_delay(
    data_path='D:/project/Delay_data/Datasets/data/flight_with_weather_2020.csv'
)
