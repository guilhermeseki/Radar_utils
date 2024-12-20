import os
import xarray as xr
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

os.chdir("/mnt/d/Radar")

#ds_radar = xr.open_dataset("cappi_3000m_lontras_202303.nc")
ds_radar = xr.open_dataset("cappi_3000m_lontras_202303_fixlon.nc")

ds_radar = ds_radar.rename({
    "rainrate_z": "rain_z",
    "rainrate_kdp": "rain_kdp",
    "rainrate_z_kdp": "rain_z_kdp",
    "rainrate_z_zdr_kdp": "rain_z_zdr_kdp"
})

# Transform mm/h to accumulated rain in 10 minutes
ds_radar["rain_z"] = ds_radar["rain_z"] / 6
ds_radar["rain_kdp"] = ds_radar["rain_kdp"] / 6
ds_radar["rain_z_kdp"] = ds_radar["rain_z_kdp"] / 6
ds_radar["rain_z_zdr_kdp"] = ds_radar["rain_z_zdr_kdp"] / 6

def main(ds_radar, regular_time):
    if regular_time:
        ds_radar = ds_radar.resample(time='10T').mean(skipna=True)

    # Step 3: Raise to the power of 5/8 to get the rainfall rate in mm/h
    df_pcd = pd.read_csv("estacoes_2023_03_23_LONTRAS_20-190km.csv", index_col=0)
    df_pcd.iloc[4:, :] = df_pcd.iloc[4:, :].apply(pd.to_numeric, errors='coerce')

    ds_lat, ds_lon = np.meshgrid(ds_radar['lat'].values, ds_radar['lon'].values, indexing='xy')
    dataset_points = np.vstack([ds_lat.ravel(), ds_lon.ravel()]).T

    # Step 2: Build KD-tree for the Dataset points
    tree = cKDTree(dataset_points)

    latitudes = pd.to_numeric(df_pcd.iloc[2,:], errors='coerce')
    longitudes = pd.to_numeric(df_pcd.iloc[3,:], errors='coerce')

    # Step 3: Prepare DataFrame points and query KD-tree for nearest neighbors
    df_points = np.vstack([latitudes, longitudes]).T
    distances, indices = tree.query(df_points, k=1)

    distances = distances*110000

    nearest_data = {}
    for var in ds_radar.data_vars:
        # Reshape to get the time series for each nearest spatial point
        data_var_values = ds_radar[var].values.reshape(-1, ds_radar.dims['time'])  # Flatten spatial, keep time
        nearest_data[var] = data_var_values[indices, :]  # Get time series for each nearest spatial point

    # Step 1: Generate new timestamps for every 5 minutes
    #new_index = pd.date_range(start='2023-03-23 00:05:00', end='2023-03-24 00:00:00', freq='5T')

    df_data = df_pcd.iloc[4:,:]
    df_data.index = pd.to_datetime(df_data.index)  - pd.Timedelta(minutes=10)

    df_data.index = pd.to_datetime(df_data.index)


    # Step 5: Create a DataFrame for each time series and merge with `df`
    time_series_dfs = []
    #rounded_datetimes = ds_radar['time'].dt.round('5min').values

    # you have to sync the dates 

    if regular_time:
        for i, estacao in enumerate(df_data.columns):
            print(f"Estação: {estacao}")
            df_estacao = df_data[estacao]
            csv_name = f"{estacao}_{round(latitudes.loc[estacao], 2):.2f}_{round(longitudes.loc[estacao], 2):.2f}".replace(".", ",")

            df_eq_estacao = pd.DataFrame({
                **{var: nearest_data[var][i, :] for var in reversed(list(ds_radar.data_vars))}
            })

            df_eq_estacao.index = pd.to_datetime(ds_radar['time'].values)

            df_merged = pd.concat([df_eq_estacao, df_estacao], axis=1)
            df_merged['distancia_pcd_ponto'] = distances[i]

            df_merged.index = df_merged.index.floor('s')
            df_merged = df_merged.round(2)
            df_merged.to_excel(f"radar_nearest_data_10min/{csv_name}_nearest_10min.xlsx")
    else:
        for i, estacao in enumerate(df_data.columns):
            print(f"Estação: {estacao}")
            df_estacao = df_data[estacao]
            csv_name = f"{estacao}_{round(latitudes.loc[estacao], 2):.2f}_{round(longitudes.loc[estacao], 2):.2f}".replace(".", ",")

            df_eq_estacao = pd.DataFrame({
                **{var: nearest_data[var][i, :] for var in reversed(list(ds_radar.data_vars))}
            })
            df_eq_estacao.index = pd.to_datetime(ds_radar['time'].values).floor('s')
            df_eq_estacao = df_eq_estacao.round(2)
            df_eq_estacao.to_excel(f"radar_nearest_data/{csv_name}_nearest.xlsx")

regular_time = False
main(ds_radar, regular_time)
regular_time = True
main(ds_radar, regular_time)
