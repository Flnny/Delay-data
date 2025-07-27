import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib

matplotlib.use('TkAgg')


# 1. 数据准备函数（带弧度版）
def prepare_route_delays():
    # 加载2024年航班数据
    file_path = "D:\project\Delay_data\Datasets\data\\flight_with_weather_2024.csv"
    df = pd.read_csv(file_path)

    # 计算每个机场对的航班数和平均延误
    route_stats = df.groupby(['ORIGIN', 'DEST']).agg({
        'DEP_DELAY': ['mean', 'count'],
        'O_LATITUDE': 'first',
        'O_LONGITUDE': 'first',
        'D_LATITUDE': 'first',
        'D_LONGITUDE': 'first'
    }).reset_index()

    # 扁平化多级列名
    route_stats.columns = ['ORIGIN', 'DEST', 'ARR_DELAY_mean', 'FLIGHT_COUNT',
                           'O_LATITUDE', 'O_LONGITUDE', 'D_LATITUDE', 'D_LONGITUDE']

    # 按航班数排序并选择前50个机场对
    top_50_routes = route_stats.sort_values('FLIGHT_COUNT', ascending=False).head(5)

    return top_50_routes


# 2. 绘制带弧度的航线
def plot_curved_routes(route_data):
    plt.figure(figsize=(16, 9))

    # 使用LambertConformal投影
    ax = plt.axes(projection=ccrs.LambertConformal())
    ax.set_extent([-125, -66, 24, 50], ccrs.PlateCarree())

    # 设置地图样式
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='white')
    ax.add_feature(cfeature.STATES.with_scale('50m'),
                   linewidth=0.5, edgecolor='#DDDDDD')
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
                   linewidth=0.5, edgecolor='#DDDDDD')

    # 创建颜色映射
    cmap = plt.get_cmap('RdYlGn_r')
    min_delay = route_data['ARR_DELAY_mean'].min()
    max_delay = route_data['ARR_DELAY_mean'].max()
    norm = Normalize(vmin=min_delay, vmax=max_delay)

    # 为每条航线生成弧线点
    def generate_curve_points(lon1, lat1, lon2, lat2, num_points=50):
        """生成两点之间的弧线坐标点"""
        # 创建大圆弧线
        gc = ccrs.Geodetic()
        points = gc.transform_points(ccrs.PlateCarree(),
                                     np.linspace(lon1, lon2, num_points),
                                     np.linspace(lat1, lat2, num_points))
        return points[:, 0], points[:, 1]

    # 绘制带弧度的航线
    for _, row in route_data.iterrows():
        delay = row['ARR_DELAY_mean']
        lon1, lat1 = row['O_LONGITUDE'], row['O_LATITUDE']
        lon2, lat2 = row['D_LONGITUDE'], row['D_LATITUDE']

        # 生成弧线点
        lons, lats = generate_curve_points(lon1, lat1, lon2, lat2)

        # 绘制弧线
        ax.plot(lons, lats,
                color=cmap(norm(delay)),
                linewidth=1.2,
                alpha=0.7,
                transform=ccrs.PlateCarree())

    # 添加颜色条
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical',
                        pad=0.02, aspect=40)
    cbar.set_label('Average Delay (minutes)',
                   fontsize=12, labelpad=10)

    # 添加标题和标注
    plt.title('Top 50 US airport-Pairs  (DEP_DELAY 2024)',
              fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig('D:\project\Delay_data\\util\\figures\Airport\\top5_curved_routes_delay.png', dpi=300, bbox_inches='tight')
    plt.show()


# 主程序
if __name__ == "__main__":
    # 获取前50个机场对的数据
    top_50_routes = prepare_route_delays()

    # 打印前10个查看
    print("航班数最多的前10个机场对:")
    # print(top_50_routes[['ORIGIN', 'DEST', 'FLIGHT_COUNT', 'ARR_DELAY_mean']].head(10))

    # 绘制带弧度的航线图
    plot_curved_routes(top_50_routes)