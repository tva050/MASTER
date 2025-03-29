import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc
from  scipy.interpolate import griddata
import scipy.io
import pandas as pd
from pyproj import Proj, Transformer, transform
import cartopy.crs as ccrs
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import os
import glob
import h5py as h5


""" 
Plan 
 - Extract the files for corresponding mooring,
    - these are represtented in the file name as "**a_**.mat", "**b_**.mat" and "**c_**.mat"
 - create an single data set for each mooring (a, b, d), which includes IDS and dates
    - Filter out data which not covers the month 01, 02, 03, 04, 10, 11, and 12
    - These are the only valid months for Cryosat and SMOS
 - IDS: daily ice draft statistics: number, mean, std, minimum, maximum, median
 - Extract, satellite data SMOS and Cryo 
 - Grid mooring and satellite to the same grid (Cryo grid, 25km)
 - Calculate referance monthly mean ice draft thickness from the moorings
 - Identify the satellite grid cells within the search radius of moorings
 - Satellite grid cells within the search radius is averaged 
 - Convert the derived sea ice freeboard to draft (possible earlier)
    - sid = sit - ifb
 - Plot time series
    - Ranging from 0.9 - 1m draft 
    
Note:
Variables in the .mat file: ['BETA', 'ID', 'IDS', 'WLS', 'TILTS', 'OWBETA', 'BTBETA', 'WL', 'T', 'yday', 'dates', 'name']
"""

mooring_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\BGEP moorings\Data"

one_mooring_data = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\BGEP moorings\Data\uls10b_dailyn.mat"


MOORING_A_LAT = 75
MOORING_A_LON = -150
MOORING_B_LAT = 78
MOORING_B_LON = -150
MOORING_D_LAT = 74
MOORING_D_LON = -140

RADIUS_RANGE = 200e3


def get_mooring_data(path):
    """ 
    Import all .matlap files, by reading the data folder.
    """
    mooring_types = ["a_", "b_", "d_"]
    months = {1, 2, 3, 4, 10, 11, 12}
    
    # makes dict for each mooring 
    mooring_arrays = {}
    
    for mooring in mooring_types:
        mooring_files = glob.glob(os.path.join(path, f"{mooring}*.mat"))
        
        ids_list = []
        dates_list = []
        
        for file in mooring_files:
            get_data = scipy.io.loadmat(file)
            
            if "IDS" in get_data and "date" in get_data:
                ids_array = np.array(get_data["IDS"])  # Ice draft stats
                dates = np.array(get_data["dates"]).flatten()  # Flatten in case it's stored in 2D

                # Convert MATLAB serial date to pandas datetime
                date_series = pd.to_datetime(dates - 719529, unit="D", origin="unix")  # MATLAB to pandas

                # Filter out invalid months
                valid_indices = [i for i, d in enumerate(date_series) if d.month in months]
                ids_filtered = ids_array[valid_indices]
                dates_filtered = dates[valid_indices]  # Keep only valid dates

                # Store in the lists
                ids_list.append(ids_filtered)
                dates_list.append(dates_filtered)

        # Combine all extracted arrays for this mooring
        if ids_list:
            mooring_arrays[mooring] = {
                "IDS": np.vstack(ids_list),  # Stack into a single numpy array
                "dates": np.hstack(dates_list)  # Flatten date array
            }
            print(f"Stored data for mooring {mooring}: {mooring_arrays[mooring]['IDS'].shape} entries")
    print(f"Extracted mooring data: {mooring_arrays}")
    return mooring_arrays

#extracted_mooring_data = get_mooring_data(mooring_path)

def mooring_data(mooring_path):
    """ 
    - Extract matlab files from path 
    - The files are so stored in an single data set for each mooring (a, b, d)
        - and with an single data set for each mooring time
        - and with an single data set for each mooring ice draft statistics
            - ice draft statistics: number, mean, std, minimum, maximum, median
    - Filter out data which not covers the month 01, 02, 03, 04, 10, 11, and 12
    """
    mooring_files_A = glob.glob(os.path.join(mooring_path, "*a_*.mat"))
    mooring_files_B = glob.glob(os.path.join(mooring_path, "*b_*.mat"))
    mooring_files_D = glob.glob(os.path.join(mooring_path, "*d_*.mat"))
    
    ids_A, ids_B, ids_D = [], [], []
    dates_A, dates_B, dates_D = [], [], []
    
    for file in mooring_files_A:
        with h5.File(file, 'r') as data:
            ids_A.append(data["IDS"][()])
            dates_A.append(data["dates"][()])
        
    print("ids_A", ids_A)
    print("dates_A", dates_A)


mooring_data(mooring_path)