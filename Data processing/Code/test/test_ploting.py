import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc
from  scipy.interpolate import griddata
import pandas as pd
from pyproj import Proj, Transformer, transform
import cartopy.crs as ccrs
import seaborn as sns
from mpl_toolkits.basemap import Basemap


import h5py

file_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\BGEP moorings\Data\uls10a_dailyn.mat"

try:

    import scipy.io as sio
    test = sio.loadmat('file_path')

except NotImplementedError:

    import h5py
    with h5py.File('file_path', 'r') as hf:
        data = hf['name-of-dataset'][:]

except:
    ValueError('could not read at all...')