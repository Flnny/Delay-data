import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import matplotlib
matplotlib.use('TkAgg')

def find_longest_chains(df, top_n=3):
    """
    找出同一架飞机连续执飞的最长航班链

    参数:
        df: 航班数据
        top_n: 返回前N条最长链

    返回:
        chains: 最长航班链列表，每个元素是DataFrame
    """
    # 按飞机号和计划起飞时间排序
    df = df.sort_values(['OP_CARRIER_FL_NUM'])

    # 用字典存储各飞机的航班链
    flight_chains = defaultdict(list)
    current_chains = defaultdict(list)

    for _, row in df.iterrows():
        flight_num = row['OP_CARRIER_FL_NUM']

        if not current_chains[flight_num]:
            # 新链开始
            current_chains[flight_num].append(row)
        else:
            last_flight = current_chains[flight_num][-1]
            # 检查是否连续：当前ORIGIN等于上一个DEST
            if row['ORIGIN'] == last_flight['DEST']:
                current_chains[flight_num].append(row)
            else:
                # 保存当前链并开始新链
                if len(current_chains[flight_num]) > 1:
                    flight_chains[flight_num].append(current_chains[flight_num])
                current_chains[flight_num] = [row]

    # 提取所有航班链并排序
    all_chains = []
    for chains in flight_chains.values():
        all_chains.extend(chains)
    all_chains.sort(key=len, reverse=True)

    # 转换为DataFrame列表
    result = []
    for chain in all_chains[:top_n]:
        chain_df = pd.DataFrame(chain)
        chain_df['Chain_Order'] = range(1, len(chain_df) + 1)
        result.append(chain_df)

    return result


def plot_chain_delays(chains):
    """Plot flight delay chains with airport route labels"""
    plt.figure(figsize=(14, 6))
    plt.rcParams.update({'font.size': 12})  # Base size

    # Departure delays
    plt.subplot(1, 2, 1)
    for i, chain in enumerate(chains):
        chain['Route'] = chain['ORIGIN'] + '→' + chain['DEST']
        plt.plot(chain['Chain_Order'], chain['DEP_DELAY'],
                 marker='o', linestyle='-',
                 label=f'Chain {i + 1} (Len: {len(chain)})')
        plt.xticks(chain['Chain_Order'], chain['Route'], rotation=45, ha='right')

    plt.xlabel('Flight Segment', fontsize=14)
    plt.ylabel('Departure Delay (min)', fontsize=14)
    plt.title('Departure Delays along Flight Chain', fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':')
    plt.tick_params(labelsize=10)

    # Arrival delays
    plt.subplot(1, 2, 2)
    for i, chain in enumerate(chains):
        plt.plot(chain['Chain_Order'], chain['ARR_DELAY'],
                 marker='s', linestyle='--',
                 label=f'Chain {i + 1}')
        plt.xticks(chain['Chain_Order'], chain['Route'], rotation=45, ha='right')

    plt.xlabel('Flight Segment', fontsize=14)
    plt.ylabel('Arrival Delay (min)', fontsize=14)
    plt.title('Arrival Delays along Flight Chain', fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':')
    plt.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig('fig_delay_chain_propagation.pdf', bbox_inches='tight')
    plt.savefig('fig_delay_chain_propagation.png', dpi=300, bbox_inches='tight')
    plt.show()



# 使用示例
file_path = 'D:\project\Delay_data\Datasets\data\\flight_with_weather_2024.csv'
df = pd.read_csv(file_path)

# 找出最长的3条航班链
longest_chains = find_longest_chains(df, top_n=3)

# 打印航班链详情
print(f"Found {len(longest_chains)} longest flight chains:")
for i, chain in enumerate(longest_chains, 1):
    print(f"\nChain {i} (Length: {len(chain)} flights, Aircraft: {chain.iloc[0]['OP_CARRIER_FL_NUM']})")
    for _, row in chain.iterrows():
        print(
            f"  {row['ORIGIN']} → {row['DEST']} (DEP_DELAY: {row['DEP_DELAY']:.1f} min, ARR_DELAY: {row['ARR_DELAY']:.1f} min)")

# 绘制延误图
plot_chain_delays(longest_chains)