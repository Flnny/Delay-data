import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# -------- 1. 构建航班链轨迹数据 --------
def prepare_flight_chain():
    file_path = "D:\\project\\Delay_data\\Datasets\\data\\flight_with_weather_2024.csv"
    df = pd.read_csv(file_path)

    # 排序：按飞机编号和时间顺序构建轨迹
    df = df.sort_values(by=['OP_CARRIER_FL_NUM', 'FL_DATE', 'DEP_TIME'])

    # 筛选字段
    df_chain = df[['OP_CARRIER_FL_NUM', 'FL_DATE', 'ORIGIN', 'DEST', 'DEP_DELAY',
                   'O_LATITUDE', 'O_LONGITUDE', 'D_LATITUDE', 'D_LONGITUDE']].copy()

    # 构建链条
    chains = []
    grouped = df_chain.groupby('OP_CARRIER_FL_NUM')

    for _, group in grouped:
        group = group.reset_index(drop=True)
        for i in range(len(group) - 1):
            chains.append({
                'FROM': group.loc[i, 'DEST'],
                'TO': group.loc[i + 1, 'DEST'],
                'FROM_LAT': group.loc[i, 'D_LATITUDE'],
                'FROM_LON': group.loc[i, 'D_LONGITUDE'],
                'TO_LAT': group.loc[i + 1, 'D_LATITUDE'],
                'TO_LON': group.loc[i + 1, 'D_LONGITUDE'],
                'DELAY': group.loc[i + 1, 'DEP_DELAY']
            })

    chain_df = pd.DataFrame(chains)
    return chain_df


# -------- 2. 绘图函数（支持两种图） --------
def plot_flight_chain(chain_df, title, save_path, core_airport=None):
    plt.figure(figsize=(16, 9))
    ax = plt.axes(projection=ccrs.LambertConformal())
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='white')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5, edgecolor='#DDDDDD')
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5, edgecolor='#DDDDDD')

    # 设置颜色映射（根据延误）
    cmap = plt.get_cmap('RdYlGn_r')
    norm = Normalize(vmin=chain_df['DELAY'].min(), vmax=chain_df['DELAY'].max())

    for _, row in chain_df.iterrows():
        ax.plot([row['FROM_LON'], row['TO_LON']],
                [row['FROM_LAT'], row['TO_LAT']],
                transform=ccrs.Geodetic(),
                color=cmap(norm(row['DELAY'])),
                linewidth=1.5,
                alpha=0.8)

        # 标注机场点
        ax.plot(row['FROM_LON'], row['FROM_LAT'], 'o', transform=ccrs.PlateCarree(), markersize=2, color='blue')
        ax.plot(row['TO_LON'], row['TO_LAT'], 'o', transform=ccrs.PlateCarree(), markersize=2, color='red')

    # 强调核心机场（若有）
    if core_airport:
        core_rows = chain_df[(chain_df['FROM'] == core_airport) | (chain_df['TO'] == core_airport)]
        if not core_rows.empty:
            lat = core_rows.iloc[0]['FROM_LAT']
            lon = core_rows.iloc[0]['FROM_LON']
            ax.plot(lon, lat, 'o', transform=ccrs.PlateCarree(),
                    markersize=10, color='gold', label='Core airport')

    # 添加色条
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, aspect=40)
    cbar.set_label('Departure Delay (minutes)', fontsize=12, labelpad=10)

    # 标题和保存
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# -------- 3. 筛选核心机场链 --------
def extract_core_airport_chains(chain_df, core_airport='ORD'):
    return chain_df[(chain_df['FROM'] == core_airport) | (chain_df['TO'] == core_airport)].copy()


# -------- 4. 主程序 --------
if __name__ == "__main__":
    # 读取并构建链数据
    chain_data = prepare_flight_chain()

    # -------- 图1：完整航班链轨迹 --------
    print("绘制航班链轨迹图...")
    plot_flight_chain(chain_data.head(100),  # 控制数量防止太乱
                      title='Sample Aircraft Flight Chain (2024)',
                      save_path='/util/figures/airport\\flight_chain_map.png')

    # -------- 图2：以 ORD 为核心机场的航班链网络 --------
    print("绘制核心机场航班链网络图 (ORD)...")
    core_chain = extract_core_airport_chains(chain_data, core_airport='ORD')
    plot_flight_chain(core_chain.head(100),
                      title='Flight Chains Through ORD',
                      save_path='/util/figures/airport\\ord_flight_chain_network.png',
                      core_airport='ORD')

