# mandatory imports
import os
from xcube.core.store import find_data_store_extensions
from xcube.core.store import get_data_store_params_schema
from xcube.core.store import new_data_store

# Utilities for notebook visualization
from IPython.display import JSON

bbox=[68.137207,24.886436,84.836426,34.379713]
res = (bbox[2]-bbox[0])/512 # ~3629m
date_range = ['2019-01-01', '2023-12-31']
timesteps = '5D'

def download_S5PL2(output_path, bbox, spatial_res, time_range, time_period):

    # Load the data
    JSON({e.name: e.metadata for e in find_data_store_extensions()})

    # Get the schema for the SentinelHub data store
    get_data_store_params_schema("sentinelhub")

    # Create a new SentinelHub data store
    store = new_data_store(
        "sentinelhub",
        api_url="https://creodias.sentinel-hub.com",
        num_retries=400,
        client_id=os.environ["SH_CLIENT_ID"],
        client_secret=os.environ["SH_CLIENT_SECRET"],
    )

    # List available data products in SetinelHub store
    print("Sentinel products available: ", list(store.get_data_ids()))

    # Describe the Sentinel-5P level 2 data products
    print("Available Sentinel-5P products: " store.describe_data('S5PL2'))

    # Define the region of interest coordinates

    # Open the Sentinel-5P level 2 data cube
    cube = store.open_data(
        'S5PL2',
        variable_names=['NO2', 'SO2', 'O3', 'CO', 'CH4', 'HCHO', 'AER_AI_340_380', 'AER_AI_354_388', 'CLOUD_FRACTION'],
        tile_size=[512, 512],
        bbox=bbox,
        spatial_res=spatial_res,
        upsampling='BILINEAR',
        time_range=time_range,
        time_period=time_period
    )

    # Remove the history attribute (causes issues with netcdf export)
    cube.attrs['history'] = str(cube.attrs['history'])

    # Save the data cube to a netcdf file
    cube.to_netcdf(output_path, engine = "netcdf4")


download_S5PL2("../data/S5PL2.nc", bbox, res, date_range, timesteps)