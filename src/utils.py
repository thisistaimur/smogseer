import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.io.img_tiles import GoogleTiles
import cartopy.feature as cf

def timeseries_lineplot(dataset : xr.Dataset, var: str, lat: float, lon: float, output_path: str):
    """
    Visualizes a time series of a variable at a specific latitude and longitude.

    Args:
        dataset (xr.Dataset): The dataset containing the variable.
        var (str): The name of the variable to plot.
        lat (float): The latitude of the location.
        lon (float): The longitude of the location.
        output_path (str): The path to save the plot.

    Returns:
        None
    """
    # Replace 'variable_name' with the name of the variable you want to plot
    variable = dataset[var].sel(lat=lat, lon=lon, method="nearest")

    # Step 3: Plot the data
    # Assuming the variable has a time dimension
    variable.plot.line(x='time', color="green")

    # Customize the plot (optional)
    plt.title(f'Time Series of {var} at lat: {lat} and lon: {lon}')
    plt.xlabel('Time')
    plt.ylabel(f'{var}')
    plt.grid(True)

    # Show the plot
    plt.draw()
    plt.savefig(output_path, dpi=300)

def variable_map(dataset : xr.Dataset, var: str, output_path: str):
    """
    Plots a dataset on a map using Mercator projection.

    Args:
        dataset (xr.Dataset): The dataset to plot.
        var (str): The name of the variable to plot.
        output_path (str): The path to save the plot.

    Returns:
        None
    """
    # First we specify Coordinate Refference System for Map Projection
    # We will use Mercator, which is a cylindrical, conformal projection. 
    # It has bery large distortion at high latitudes, cannot 
    # fully reach the polar regions.
    
    tiler = GoogleTiles(style="satellite")
    mercator = tiler.crs
    projection = ccrs.Mercator()
    # Specify CRS, that will be used to tell the code, where should our data be plotted
    crs = ccrs.PlateCarree()
    # Now we will create axes object having specific projection 
    plt.figure(figsize=(16,9), dpi=150)
    ax = plt.axes(projection=mercator, frameon=True)
    # Draw gridlines in degrees over Mercator map
    gl = ax.gridlines(crs=crs, draw_labels=True,
                    linewidth=.6, color='gray', alpha=0.5, linestyle='-.')
    gl.xlabel_style = {"size" : 7}
    gl.ylabel_style = {"size" : 7}
    # To plot borders and coastlines, we can use cartopy feature
    ax.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.5)
    ax.add_feature(cf.BORDERS.with_scale("50m"), lw=0.3)
    ax.add_feature(cf.LAKES, alpha=0.95)
    ax.add_feature(cf.RIVERS)
   
    
    # Now, we will specify extent of our map in minimum/maximum longitude/latitude
    # Note that these values are specified in degrees of longitude and degrees of latitude
    # However, we can specify them in any crs that we want, but we need to provide appropriate
    # crs argument in ax.set_extent
    # crs is PlateCarree -> we are explicitly telling axes, that we are creating bounds that are in degrees
    lon_min = dataset.attrs["geospatial_lon_min"]
    lon_max = dataset.attrs["geospatial_lon_max"]
    lat_min = dataset.attrs["geospatial_lat_min"]
    lat_max = dataset.attrs["geospatial_lat_max"]
   

    cbar_kwargs = {'orientation':'horizontal', 'shrink':0.6, "pad" : .05, 'aspect':40, 'label':f'{var}'}
    dataset[var].isel(time=2).plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs=cbar_kwargs, levels=21, cmap='Spectral'), 
    ################################
    
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=crs)
    ax.add_image(tiler, 6)
    #ax.add_image(stamen_terrain, 8)
    #ax.stock_img()
    #plt.title(f"NO2 anomaly over study")
    plt.draw()
    plt.savefig(output_path, dpi=300)


# Plot training history
def plot_training_history(history, save_path):
    """
    Plots the training and validation loss and accuracy over epochs.

    Parameters:
    - history: Keras History object
    - save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss over epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Accuracy over epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Plot comparisons and training history

def plot_comparison(y_true, y_pred, index, save_path):
    """
    Plots the ground truth and the predicted output for a given index.
    
    Parameters:
    - y_true: Ground truth data
    - y_pred: Predicted data
    - index: Index of the sample to plot
    - save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot ground truth
    ax = axes[0]
    ax.imshow(y_true[index, 0, :, :, 0], cmap='viridis')
    ax.set_title('Ground Truth')
    ax.axis('off')

    # Plot prediction
    ax = axes[1]
    ax.imshow(y_pred[index, 0, :, :, 0], cmap='viridis')
    ax.set_title('Prediction')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()