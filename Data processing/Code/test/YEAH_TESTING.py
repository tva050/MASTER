import os
import glob
import numpy as np
from scipy.io import loadmat
from datetime import datetime
from datetime import timedelta, datetime
import h5py
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import matplotlib.cm as cm

path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Framstrait_2015-2016_F11_IPS5_1062.nc"

def print_nc_metadata(file_path):
    """Prints metadata from a NetCDF (.nc) file."""
    # Open the NetCDF file
    with Dataset(file_path, 'r') as nc_file:
        print(nc_file.variables.keys())
        # Print global attributes
        print("Global Attributes:")
        for attr in nc_file.ncattrs():
            print(f"{attr}: {nc_file.getncattr(attr)}")
        
        print("\nVariables:")
        for var_name, var in nc_file.variables.items():
            print(f"{var_name}:")
            print(f"  Dimensions: {var.dimensions}")
            print(f"  Shape: {var.shape}")
            print(f"  Data Type: {var.dtype}")
            
            # Print variable attributes
            for attr in var.ncattrs():
                print(f"  {attr}: {var.getncattr(attr)}")
                
print_nc_metadata(path)