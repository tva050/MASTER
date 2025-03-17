import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc
from  scipy.interpolate import griddata
import pandas as pd
from pyproj import Proj, transform

one_smos_file = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2013\SMOS_Icethickness_v3.3_north_20130321.nc"


def get_data(file_path):
    data = nc.Dataset(file_path)
    print(data.variables.keys())
    
    
get_data(one_smos_file)