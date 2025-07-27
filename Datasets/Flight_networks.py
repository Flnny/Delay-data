import time
import warnings
import datetime
from concurrent.futures import ThreadPoolExecutor
import os

import dgl
import pandas as pd
import numpy as np
import torch

# Set global parameters
warnings.filterwarnings("ignore")
pd.options.display.max_columns = 50

# Define processing function
def process_year_file(year_file):
    print(f"Start processing file: {year_file}")
    start_time = time.perf_counter()

    # Read data
    df = pd.read_csv(year_file, low_memory=False)

    # Process datetime columns
    datetime_cols = [
        'FL_DATE', 'CRS_DEP_TIME', 'DEP_TIME', 'WHEELS_OFF',
        'WHEELS_ON', 'CRS_ARR_TIME', 'ARR_TIME'
    ]
    df[datetime_cols] = df[datetime_cols].apply(pd.to_datetime, errors='coerce', format='%Y-%m-%d %H:%M:%S')

    # Convert categorical variables
    category_cols = ['OP_CARRIER', 'ORIGIN', 'DEST']
    df[category_cols] = df[category_cols].astype('category')

    # Optimize numeric column types
    float_cols = df.select_dtypes(include='float64').columns
    df[float_cols] = df[float_cols].astype('float32')

    # Extract hour from scheduled departure and arrival time
    df['CRS_DEP_TIME_HOUR'] = df['CRS_DEP_TIME'].dt.hour.astype('int8')
    df['CRS_ARR_TIME_HOUR'] = df['CRS_ARR_TIME'].dt.hour.astype('int8')

    # Rename index columns
    df.drop(columns=['ORIGIN', 'DEST'], inplace=True)
    df.rename(columns={
        'ORIGIN_INDEX': 'ORIGIN',
        'DEST_INDEX': 'DEST'
    }, inplace=True)

    # Define function to add edges
    def add_edges1(g, root, depth, df1, idx, visit):
        if visit[root] == 1:
            return
        visit[root] = 1

        df_temp = df1.loc[(df1['ORIGIN'] == df1['DEST'][root]) &
                          (df1['CRS_DEP_TIME'] >= df1['CRS_ARR_TIME'][root]) &
                          (df1['CRS_DEP_TIME'] <= df1['CRS_ARR_TIME'][root] + datetime.timedelta(minutes=15))]

        # Node features
        node_features = {
            'CRS_DEP_TIME': (df1['CRS_DEP_TIME'][root] - pd.to_datetime('20160101')).seconds,
            'DEP_DELAY': df1['DEP_DELAY'][root] * 60,
            'CRS_ARR_TIME': (df1['CRS_ARR_TIME'][root] - pd.to_datetime('20160101')).seconds,
            'ARR_DELAY': df1['ARR_DELAY'][root] * 60,
            'DEST': df1['DEST'][root],
            'ORIGIN': df1['ORIGIN'][root],
            'O_LATITUDE': df1['O_LATITUDE'][root],
            'O_LONGITUDE': df1['O_LONGITUDE'][root],
            'D_LATITUDE': df1['D_LATITUDE'][root],
            'D_LONGITUDE': df1['D_LONGITUDE'][root],
            'FLIGHTS': df1['FLIGHTS'][root],
            'O_TEMP': df1['O_TEMP'][root],
            'O_PRCP': df1['O_PRCP'][root],
            'O_WSPD': df1['O_WSPD'][root],
            'D_TEMP': df1['D_TEMP'][root],
            'D_PRCP': df1['D_PRCP'][root],
            'D_WSPD': df1['D_WSPD'][root],
            'WHEELS_OFF': (df1['WHEELS_OFF'][root] - df1['DEP_TIME'][root]).seconds,
            'MONTH': df1['MONTH'][root],
            'DAY_OF_WEEK': df1['DAY_OF_WEEK'][root],
            'CRS_ARR_TIME_HOUR': df1['CRS_ARR_TIME_HOUR'][root],
            'CRS_DEP_TIME_HOUR': df1['CRS_DEP_TIME_HOUR'][root],
        }

        for f in node_features:
            value = torch.tensor(node_features[f], dtype=g.ndata[f].dtype)
            g.ndata[f][idx[root]] = value

        if df_temp.shape[0] == 0:
            return
        else:
            for index, row in df_temp.iterrows():
                if idx[index] == -1:
                    g.add_nodes(1)
                    idx[index] = g.num_nodes() - 1

                g.add_edge(idx[root], idx[index])

                edge_features = {
                    'INTERVAL_TIME': (df1['CRS_DEP_TIME'][index] - df1['CRS_ARR_TIME'][root]).seconds,
                }

                for f in edge_features:
                    g.edata[f][g.num_edges() - 1] = edge_features[f]

                if df1['CRS_DEP_TIME'][index] >= df1['CRS_ARR_TIME'][index]:
                    df1['CRS_ARR_TIME'][index] = df1['CRS_ARR_TIME'][index] + datetime.timedelta(days=1)

                add_edges1(g, index, depth + 1, df1, idx, visit)

    # Define function to create and save graph
    def create_and_save_graph(date, df):
        df1 = df.query(f"FL_DATE == '{date}'")
        df1 = df1.reset_index(drop=True)

        idx = torch.full((len(df1),), -1, dtype=torch.long)
        visit = np.zeros(len(df1))

        g1 = dgl.DGLGraph()

        features = ['CRS_DEP_TIME', 'DEP_DELAY', 'CRS_ARR_TIME', 'ARR_DELAY', 'WHEELS_OFF', 'DEST', 'ORIGIN',
                    'O_LATITUDE', 'O_LONGITUDE', 'D_LATITUDE', 'D_LONGITUDE', 'FLIGHTS', 'O_TEMP', 'O_PRCP',
                    'O_WSPD', 'D_TEMP', 'D_PRCP', 'D_WSPD', 'MONTH', 'DAY_OF_WEEK', 'CRS_ARR_TIME_HOUR',
                    'CRS_DEP_TIME_HOUR']

        for f in features:
            g1.ndata[f] = torch.ones(g1.num_nodes(), 1)

        edge_features = ['INTERVAL_TIME', 'AIRPORT', 'AIRCRAFT_NUM']
        for f in edge_features:
            g1.edata[f] = torch.ones(g1.num_edges(), 1)

        for root in range(0, len(df1) - 1):
            if idx[root] == -1:
                g1.add_nodes(1)
                idx[root] = g1.num_nodes() - 1
            add_edges1(g1, root, 0, df1, idx, visit)

        # Set labels and features
        g1.ndata['label'] = g1.ndata['ARR_DELAY'] / 60
        g1.ndata['feat'] = torch.cat([
            g1.ndata['O_LATITUDE'], g1.ndata['O_LONGITUDE'],
            g1.ndata['D_LATITUDE'], g1.ndata['D_LONGITUDE'],
            g1.ndata['FLIGHTS'], g1.ndata['O_PRCP'],
            g1.ndata['O_WSPD'], g1.ndata['D_PRCP'],
            g1.ndata['D_WSPD'], g1.ndata['DAY_OF_WEEK'],
            g1.ndata['MONTH'], g1.ndata['CRS_ARR_TIME_HOUR'],
            g1.ndata['CRS_DEP_TIME_HOUR'], g1.ndata['ORIGIN'],
            g1.ndata['DEST']
        ], dim=1)

        g1.ndata['feat'] = torch.nan_to_num(g1.ndata['feat'], nan=0)

        # Create output directory
        year = year_file.split('_')[-1].split('.')[0]
        output_dir = f"graph_output_{year}"
        os.makedirs(output_dir, exist_ok=True)

        file_name = os.path.join(output_dir, f"graph{date}.dgl")
        dgl.save_graphs(file_name, g1)
        return file_name

    # Process all dates for the year
    year = year_file.split('_')[-1].split('.')[0]
    start_date = datetime.date(int(year), 1, 1)
    end_date = datetime.date(int(year), 12, 31)
    delta = datetime.timedelta(days=1)

    processed_files = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        try:
            output_file = create_and_save_graph(date_str, df)
            processed_files.append(output_file)
            print(f"Successfully processed {date_str}, saved to {output_file}")
        except Exception as e:
            print(f"Error processing {date_str}: {str(e)}")
        current_date += delta

    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Finished processing file {year_file}, total time: {total_time:.2f} seconds")

    return processed_files

# Main program
if __name__ == "__main__":
    # Get all year files
    year_files = [f"flight_with_weather_{year}.csv" for year in range(2016, 2025)]

    # Use thread pool for parallel processing
    with ThreadPoolExecutor(max_workers=min(4, len(year_files))) as executor:
        results = list(executor.map(process_year_file, year_files))

    print("All year files processed!")
    print("Generated graph file list:")
    for year_result in results:
        for file in year_result:
            print(file)