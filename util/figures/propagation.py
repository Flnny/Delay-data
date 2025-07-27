import pandas as pd
import plotly.graph_objects as go


file_name = 'E:\\Delay_data\\flight_with_weather_2024.csv'
df = pd.read_csv(file_name)
df = df.dropna()

# 构建航线流量矩阵
flow_df = df.groupby(['ORIGIN', 'DEST']).agg(
    total_flights=('FLIGHTS', 'sum'),
    avg_delay=('ARR_DELAY', 'mean')
).reset_index()

# 创建节点列表
nodes = list(set(flow_df.ORIGIN) | set(flow_df.DEST))
node_dict = {node: i for i, node in enumerate(nodes)}

# 桑基图参数
link = dict(
    source = flow_df.ORIGIN.map(node_dict),
    target = flow_df.DEST.map(node_dict),
    value = flow_df.total_flights,
    color = flow_df.avg_delay,
    colorscale = 'RdYlGn_r',
    hovertemplate = 'Origin: %{source.label}<br>Destination: %{target.label}<br>Flights: %{value}<br>Avg Delay: %{color} min<extra></extra>'
)

# 绘制交互式图表
fig = go.Figure(go.Sankey(
    node = dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label = nodes,
        color = 'lightgray'
    ),
    link = link
))

fig.update_layout(
    title_text="airport Traffic Flow with Delay Propagation",
    font_size=10,
    height=800,
    width=1200
)
fig.write_html("sankey_interactive.html")