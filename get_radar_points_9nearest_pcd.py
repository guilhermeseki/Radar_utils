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

    df_pcd = pd.read_csv("estacoes_2023_03_23_LONTRAS_20-190km.csv", index_col=0)
    df_pcd.iloc[4:, :] = df_pcd.iloc[4:, :].apply(pd.to_numeric, errors='coerce')

    # Step 1: Create a grid of radar dataset points
    ds_lat, ds_lon = np.meshgrid(ds_radar['lat'].values, ds_radar['lon'].values, indexing='ij')
    dataset_points = np.vstack([ds_lat.ravel(), ds_lon.ravel()]).T

    # Step 2: Build KD-tree for the Dataset points
    tree = cKDTree(dataset_points)

    # Prepare latitude and longitude arrays for the query
    latitudes = pd.to_numeric(df_pcd.iloc[2, :], errors='coerce')
    longitudes = pd.to_numeric(df_pcd.iloc[3, :], errors='coerce')

    # Step 3: Prepare points and query KD-tree for the 9 nearest neighbors
    df_points = np.vstack([latitudes, longitudes]).T
    distances, indices = tree.query(df_points, k=9)  # k=9 to find 9 nearest neighbors
    distances = distances*110000

    # Initialize a dictionary to store mean values
    nearest_data = {}

    for var in ds_radar.data_vars:
        # Reshape the variable to a 2D array (points, time) for easier access
        data_var_values = ds_radar[var].values.reshape(-1, ds_radar.dims['time'])  # Flatten spatial dimensions
        nearest_values = data_var_values[indices, :]  # Retrieve values for all 9 neighbors

        # Calculate the mean across the 9 neighbors for each point
        nearest_data[var] = np.nanmean(nearest_values, axis=1)  # Take the mean along the neighbors' axis

    # Convert the results into a DataFrame for easy access
    #mean_df = pd.DataFrame(mean_data)
    #mean_df.index = df_pcd.index  # Ensure the indices match your query points

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
            values = distances[i]  # Length = 9

            # Create a column of NaN values, then update the first few rows with `values`
            df_merged['distancia_pcd_9pontos'] = np.nan  # Initialize the column with NaNs
            df_merged.iloc[:len(values), df_merged.columns.get_loc('distancia_pcd_9pontos')] = values

            df_merged.index = df_merged.index.floor('s')
            df_merged = df_merged.round(2)
            df_merged.to_excel(f"radar_nearest9_data_10min/{csv_name}_nearest9_10min.xlsx")
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
            df_eq_estacao.to_excel(f"radar_nearest9_data/{csv_name}_nearest9.xlsx")

regular_time = False
main(ds_radar, regular_time)
regular_time = True
main(ds_radar, regular_time)

