import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

file_name = 'E:\\Delay_data\\flight_with_weather_2024.csv'
df = pd.read_csv(file_name)
df = df.dropna(subset=['ARR_DELAY', 'ARR_TIME'])

# 选择前5大机场并聚合数据
airports = df['ORIGIN'].value_counts().nlargest(5).index.tolist()
radar_data = df.groupby('ORIGIN').agg({
    'ARR_DELAY': 'median',
    'DEP_DELAY': 'median',
    'AIR_TIME': 'median',
    'TAXI_OUT': 'median',
    'O_TEMP': 'median',
    'O_WSPD': 'median',
}).loc[airports]

# 数据标准化
radar_data = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())

# 自定义颜色和标签
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 五色方案
airport_names = radar_data.index.tolist()

# 雷达图参数
labels = ['Arrival Delay','Departure Delay', 'Air Time', 'Taxi Out', 'Temperature',  'Wind Speed']
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()

# 绘图
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, polar=True)

# 绘制每个机场
for idx, (airport, row) in enumerate(radar_data.iterrows()):
    values = row.values.flatten().tolist()
    values += values[:1]  # 闭合雷达图
    ax.plot(angles + angles[:1], values, color=colors[idx], linewidth=2,
            linestyle='-', marker='o', markersize=4, label=airport)
    ax.fill(angles + angles[:1], values, color=colors[idx], alpha=0.1)

# 美化图形
ax.set_theta_offset(np.pi / 2)       # 0度位置在顶部
ax.set_theta_direction(-1)           # 顺时针方向
ax.set_thetagrids(np.degrees(angles), labels)
ax.set_rlabel_position(30)          # 半径标签位置

# 添加清晰图例
legend = ax.legend(
    loc='upper right',
    bbox_to_anchor=(1.25, 1),
    frameon=True,
    edgecolor='black',
    title='Airports'
)

# 高分辨率保存
plt.savefig('radar_airports.png', dpi=300, bbox_inches='tight')
plt.close()