import os
import glob
import xarray as xr
import numpy as np
from netCDF4 import Dataset


""" 
- Function to extract the all files from the SMOS folder
- Than calculate the mean ice thickness for each month available in the folder
    - This is done by looking at the file names, which are in the format "SMOS_Icethickness_v3.3_north_YYYYMMDD.nc"
- The mean ice thickness is calculated for each month and stored in a dictionary with the corresponding year and month as the key
- The function returns a dictionary with the mean ice thickness for each month
- The function also returns the lat and lon coordinates for the ice thickness data
- The function also returns the uncertainty data for the ice thickness data
- The function also returns the sea ice draft by: draft = thickness * 0.93 (source: https://doi.org/10.1029/2007JC004252)
- It is so stored in an new .nc file with the name "SMOS_Icethickness_north_YYYYMM.nc" in a new folder called "SMOS_monthly"
- The new folder is created in the same directory as the original folder
- The new .nc file contains the mean ice thickness for each month, the mean ice thickness for each month, the lat and lon coordinates, and the uncertainty data
"""

folder_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\All years"

def SMOS_monthly(folder_path):
    """
    Processes daily SMOS sea ice thickness data to compute monthly averages and saves results as NetCDF files.

    This function reads daily SMOS sea ice thickness NetCDF files from the specified folder, groups them by month, 
    computes the monthly mean sea ice thickness, uncertainty, and sea ice draft, and saves the results in a 
    new folder. The original daily files are removed after processing.

    Parameters
    ----------
    folder_path : str
        The path to the folder containing daily SMOS sea ice thickness NetCDF files. The files should follow 
        the naming convention `"SMOS_Icethickness_*_north_*.nc"`.

    Returns
    -------
    dict
        A dictionary where keys are formatted as "YYYY-MM" (year-month), and values are dictionaries containing:
        - **latitude** (numpy.ndarray): 2D array of latitude values.
        - **longitude** (numpy.ndarray): 2D array of longitude values.
        - **mean_sit** (numpy.ndarray): 2D array of monthly mean sea ice thickness values.
        - **uncertainty** (numpy.ndarray): 2D array of monthly mean uncertainty values.
        - **sea_ice_draft** (numpy.ndarray): 2D array of monthly mean sea ice draft values.

    Notes
    -----
    - The function removes the original daily NetCDF files after processing them.
    - The output NetCDF files are saved in a subfolder named `"SMOS_monthly"` within the specified folder.
    - Sea ice draft is computed as `sea_ice_thickness * 0.93`, based on: 
      `https://doi.org/10.1029/2007JC004252`
    - Missing or invalid data points (`NaN`, `-999.0`, or `0.0`) are masked before averaging.
    - If a land mask is present in the input files (`land` variable), land areas are set to `NaN`.

    Examples
    --------
    >>> smos_data = SMOS_monthly("/path/to/smos/data")
    >>> print(smos_data["2023-01"]["mean_sit"])  # Prints monthly mean sea ice thickness for January 2023
    """
    smos_files = glob.glob(os.path.join(folder_path, "SMOS_Icethickness_*_north_*.nc"))
    smos_data = {}
    
    output_folder = os.path.join(folder_path, "SMOS_monthly")
    os.makedirs(output_folder, exist_ok=True)
    
    monthly_files = {}
    for file in smos_files:
        filename = os.path.basename(file)
        date_part = filename.split("_")[-1].split(".")[0]
        year, month = date_part[:4], date_part[4:6]
        date_key = f"{year}-{month}"
        
        if date_key not in monthly_files:
            monthly_files[date_key] = []
        monthly_files[date_key].append(file)
    
    print(len(monthly_files), "months found")
    
    for date_key, files in monthly_files.items():
        sit_list = []
        uncertainty_list = []
        lat, lon = None, None # place holders 
        
        for file in files:
            with xr.open_dataset(file) as ds:
                sit_list.append(ds["sea_ice_thickness"].values)
                uncertainty_list.append(ds["ice_thickness_uncertainty"].values)
                land_mask = ds.get("land", None)
                
                if land_mask is not None:
                    sit_list[-1] = np.where(land_mask == 1, np.nan, sit_list[-1])
                    uncertainty_list[-1] = np.where(land_mask == 1, np.nan, uncertainty_list[-1])
                else:
                    print("land mask not found in all files")
                    
                if lat is None:
                    lat = ds["latitude"].values
                    lon = ds["longitude"].values
            
            os.remove(file)  # Remove the file after processing
                    
        mask_sit = ~np.isnan(sit_list) & (sit_list != -999.0) & (sit_list != 0.0)
        mask_unc = ~np.isnan(uncertainty_list) & (uncertainty_list != -999.0) & (uncertainty_list != 0.0)
        sit = np.where(mask_sit, sit_list, np.nan)
        uncertainty = np.where(mask_unc, uncertainty_list, np.nan)
        
        sit = np.array(sit_list)
        uncertainty = np.array(uncertainty_list)
        mean_sit = np.mean(sit[0,:,:], axis=0)
        mean_unc = np.mean(uncertainty[0,:,:], axis=0)
        
        sea_ice_draft = mean_sit * 0.93 # source: https://doi.org/10.1029/2007JC004252
        
        smos_data[date_key] = {
            "latitude": lat,
            "longitude": lon,
            "mean_sit": mean_sit,
            "uncertainty": mean_unc,
            "sea_ice_draft": sea_ice_draft,
        }

        output_file = os.path.join(output_folder, f"SMOS_monthly_Icethickness_north_{date_key.replace('-', '')}.nc")
        new_ds = xr.Dataset(
            {
                "mean_ice_thickness": (["y", "x"], mean_sit),
                "uncertainty": (["y", "x"], mean_unc),
                "sea_ice_draft": (["y", "x"], sea_ice_draft),
            },
            coords={
                "latitude": (["y", "x"], lat),
                "longitude": (["y", "x"], lon),
            },
            attrs={
                "description": "Monthly mean sea ice thickness from daily SMOS data",
                "date": date_key,
                "source": "SMOS",
                "units": "m",
            },
        )

        new_ds.to_netcdf(output_file)
    
    return smos_data

#smos_data = SMOS_monthly(folder_path)
   
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
                
#print_nc_metadata(month_path)