""" 
This script includes an comprehensive comparison of the different products used to estimate sea ice thickness.
The comparison is done by plotting the data on a map and comparing the results visually.

The script includes the products:
- CryoSat-2 L2 Trajectory Data Baseline D
- CS2 ice thickness data from AWI
- CS2 ice thickness data from CPOM
- SMOS ice thickness data
- CryoSat-2 L3 

"""
import os
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc


cpom_oct = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\CPOM\thk_2021_10.map.nc"
cpom_nov = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\CPOM\thk_2021_11.map.nc"
cpom_dec = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\CPOM\thk_2021_12.map.nc"




def get_cpom(path):
    data = nc.Dataset(path)
    
    print(data.variables.keys())
    
    lat = data.variables['latitude'][:]
    lon = data.variables['longitude'][:]
    si_thickness = data.variables['thickness'][:]
    
    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
        print('Reshaped lat and lon')
    
    return lat, lon, si_thickness
    
    

    
    
get_cpom(cpom_oct)