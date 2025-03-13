import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc
from scipy.interpolate import griddata
import pandas as pd
from pyproj import Proj, transform

paths = [
    r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\IDCSI4_20130321.txt",
    r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\IDCSI4_20130322.txt",
    r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\IDCSI4_20130323.txt",
    r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\IDCSI4_20130324.txt",
    r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\IDCSI4_20130326.txt",
    r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\IDCSI4_20130425.txt"
]

def get_data(path):
    # Read the CSV file while automatically handling headers and mixed data types
    df = pd.read_csv(path)

    # Extract only the numerical columns we need
    lat = df["lat"].astype(float).values
    lon = df["lon"].astype(float).values
    thickness = df["thickness"].astype(float).values

    mask = (thickness != -99999.) & (thickness != 0.0)
    thickness = np.where(mask, thickness, np.nan)
    return lat, lon, thickness

def latlon_to_polar(lat, lon):
    """Convert lat/lon (degrees) to North Polar Stereographic (meters)."""
    # Define WGS84 (lat/lon) and Polar Stereographic projection
    wgs84 = Proj(proj="latlong", datum="WGS84")
    polar_stereo = Proj(proj="stere", lat_0=90, lon_0=-45, datum="WGS84", k=1, x_0=0, y_0=0)

    # Convert coordinates
    x, y = transform(wgs84, polar_stereo, lon, lat)
    return x, y

def plot_all_data(paths):
    """Plot sea ice thickness from multiple files on the same map."""
    all_x, all_y, all_thickness = [], [], []

    # Load and combine data from all files
    for path in paths:
        lat, lon, thickness = get_data(path)
        x, y = latlon_to_polar(lat, lon)

        all_x.extend(x)
        all_y.extend(y)
        all_thickness.extend(thickness)

    # Create figure
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.NorthPolarStereo()}, figsize=(10, 10))
    
    # Set map extent (meters, not degrees)
    ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())

    # Scatter plot with all data points
    scatter = ax.scatter(all_x, all_y, c=all_thickness, cmap='viridis', vmin=0, vmax=1, zorder=1, transform=ccrs.NorthPolarStereo())

    # Add map features
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=1, zorder=2)
    ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
    ax.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5, zorder=4)

    # Colorbar
    cbar = plt.colorbar(scatter, orientation='vertical')
    cbar.set_label('Sea Ice Thickness (m)')

    plt.title('Sea Ice Thickness (Multiple Paths)')
    plt.show()

# Load data and plot
plot_all_data(paths)