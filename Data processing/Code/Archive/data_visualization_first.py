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
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.stats import linregress

cpom_oct_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\CPOM\thk_2021_10.map.nc"
cpom_nov_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\CPOM\thk_2021_11.map.nc"
cpom_dec_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\CPOM\thk_2021_12.map.nc"

cryo_oct_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\2021\ubristol_cryosat2_seaicethickness_nh25km_2021_10_v1.nc"
cryo_nov_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\2021\ubristol_cryosat2_seaicethickness_nh25km_2021_11_v1.nc"
cryo_dec_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\2021\ubristol_cryosat2_seaicethickness_nh25km_2021_12_v1.nc"

smos_oct_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2021\oct_mean_thickness.nc"
smos_nov_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2021\nov_mean_thickness.nc"
smos_dec_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2021\dec_mean_thickness.nc"

awi_oct_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\AWI\awi-siral-l3c-sithick-cryosat2-rep-nh_25km_ease2-202110-fv2p6.nc"
awi_nov_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\AWI\awi-siral-l3c-sithick-cryosat2-rep-nh_25km_ease2-202111-fv2p6.nc"
awi_dec_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\AWI\awi-siral-l3c-sithick-cryosat2-rep-nh_25km_ease2-202112-fv2p6.nc"

def get_cpom(path):
    data = nc.Dataset(path)
    #print(data.variables.keys())
    
    lat = data.variables['latitude'][:]
    lon = data.variables['longitude'][:]
    si_thickness = data.variables['thickness'][:]
    grid_spacing = data.variables["grid_spacing"][:]
    si_thickness_stdev = data.variables["thk_stdev"][:]
    
    #print(f"Lat shape: {lat.shape}")
    #print(f"Lon shape: {lon.shape}")
    #print(f"Thickness shape: {si_thickness.shape}")
    
    return lat, lon, si_thickness

def get_cryo(path):
    data = nc.Dataset(path)
    print(data.variables.keys())
    
    lat = data.variables['Latitude'][:]
    lon = data.variables['Longitude'][:]
    si_thickness = data.variables['Sea_Ice_Thickness'][:]
    si_thickness_un = data.variables['Sea_Ice_Thickness_Uncertainty'][:]
    
    # Check if lat and lon are 1D and need reshaping
    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
        print('Reshaped lat and lon')
    # Mask invalid data
    mask = ~np.isnan(si_thickness)
    filtered_si_thickness = np.where(mask, si_thickness, np.nan)
    return lat, lon, filtered_si_thickness, si_thickness_un


def get_smos(path):
    data = nc.Dataset(path)
    # print(data.variables.keys())
    
    lat = data.variables['latitude'][:]
    lon = data.variables['longitude'][:]
    si_thickness = data.variables['sea_ice_thickness'][:]
    
    smos_mask = (si_thickness < 0)
    si_thickness = np.where(smos_mask, np.nan, si_thickness)
    
    return lat, lon, si_thickness
    
def get_awi(path):
    data = nc.Dataset(path)
    #print(data.variables.keys())
    
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    si_thickness = data.variables['sea_ice_thickness'][:]
    # Mask invalid data
    mask = ~np.isnan(si_thickness)
    filtered_si_thickness = np.where(mask, si_thickness, np.nan)

    return lat, lon, filtered_si_thickness
    
lat_cpom_oct, lon_cpom_oct, si_thickness_cpom_oct = get_cpom(cpom_oct_2021)
lat_cpom_nov, lon_cpom_nov, si_thickness_cpom_nov = get_cpom(cpom_nov_2021)
lat_cpom_dec, lon_cpom_dec, si_thickness_cpom_dec = get_cpom(cpom_dec_2021)

lat_cryo_oct, lon_cryo_oct, si_thickness_cryo_oct, si_thickness_un_cryo_oct = get_cryo(cryo_oct_2021)
lat_cryo_nov, lon_cryo_nov, si_thickness_cryo_nov, si_thickness_un_cryo_nov = get_cryo(cryo_nov_2021)
lat_cryo_dec, lon_cryo_dec, si_thickness_cryo_dec, si_thickness_un_cryo_dec = get_cryo(cryo_dec_2021)

lat_smos_oct, lon_smos_oct, si_thickness_smos_oct = get_smos(smos_oct_2021)
lat_smos_nov, lon_smos_nov, si_thickness_smos_nov = get_smos(smos_nov_2021)
lat_smos_dec, lon_smos_dec, si_thickness_smos_dec = get_smos(smos_dec_2021)

lat_awi_oct, lon_awi_oct, si_thickness_awi_oct = get_awi(awi_oct_2021)
lat_awi_nov, lon_awi_nov, si_thickness_awi_nov = get_awi(awi_nov_2021)
lat_awi_dec, lon_awi_dec, si_thickness_awi_dec = get_awi(awi_dec_2021)
    
    
def plot_cpom(lat, lon, si_thickness):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())

    # Plot the sea ice thickness
    sc = ax.scatter(lon, lat, c=si_thickness, s=1, transform=ccrs.PlateCarree(), cmap='jet', vmin=0, vmax=3.3, zorder=1)
    
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=1, zorder=2)
    ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
    ax.add_feature(cfeature.COASTLINE, color = "gray", linewidth=0.5, zorder=4)
    
    
    cbar = plt.colorbar(sc, orientation='vertical')
    cbar.set_label('Sea Ice Thickness (m)')
    plt.title('Sea Ice Thickness')
    plt.show()


def plot_cryo(lat, lon, si_thickness):
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.NorthPolarStereo()})
    ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())
 
    mesh = ax.pcolormesh(lon, lat, si_thickness, transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=1, zorder=1)
    
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=1, zorder=2)
    ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
    ax.add_feature(cfeature.COASTLINE, color = "gray", linewidth=0.5, zorder=4)
   
    cbar = plt.colorbar(mesh, orientation='vertical')
    cbar.set_label('Sea Ice Thickness (m)')
    
    plt.title('Sea Ice Thickness')
    plt.show()
    

def bar_plot():
    cryo_mean_oct = np.nanmean(si_thickness_cryo_oct)
    cryo_mean_nov = np.nanmean(si_thickness_cryo_nov)
    cryo_mean_dec = np.nanmean(si_thickness_cryo_dec)
    cryo_std_oct = np.nanstd(si_thickness_cryo_oct)
    cryo_std_nov = np.nanstd(si_thickness_cryo_nov)
    cryo_std_dec = np.nanstd(si_thickness_cryo_dec)
    
    cpom_mean_oct = np.nanmean(si_thickness_cpom_oct)
    cpom_mean_nov = np.nanmean(si_thickness_cpom_nov)
    cpom_mean_dec = np.nanmean(si_thickness_cpom_dec)
    cpom_std_oct = np.nanstd(si_thickness_cpom_oct)
    cpom_std_nov = np.nanstd(si_thickness_cpom_nov)
    cpom_std_dec = np.nanstd(si_thickness_cpom_dec)
    
    smos_mean_oct = np.nanmean(si_thickness_smos_oct)
    smos_mean_nov = np.nanmean(si_thickness_smos_nov)
    smos_mean_dec = np.nanmean(si_thickness_smos_dec)
    smos_std_oct = np.nanstd(si_thickness_smos_oct)
    smos_std_nov = np.nanstd(si_thickness_smos_nov)
    smos_std_dec = np.nanstd(si_thickness_smos_dec)
    
    awi_mean_oct = np.nanmean(si_thickness_awi_oct)
    awi_mean_nov = np.nanmean(si_thickness_awi_nov)
    awi_mean_dec = np.nanmean(si_thickness_awi_dec)
    awi_std_oct = np.nanstd(si_thickness_awi_oct)
    awi_std_nov = np.nanstd(si_thickness_awi_nov)
    awi_std_dec = np.nanstd(si_thickness_awi_dec)
    
    print(f"CryoSat-2 November: Mean: {cryo_mean_nov}, Std: {cryo_std_nov}")
    print(f"CPOM November: Mean: {cpom_mean_nov}, Std: {cpom_std_nov}")
    print(f"SMOS November: Mean: {smos_mean_nov}, Std: {smos_std_nov}")
    print(f"AWI November: Mean: {awi_mean_nov}, Std: {awi_std_nov}")
    
    months = ['October', 'November', 'December']
    cryo_means = [cryo_mean_oct, cryo_mean_nov, cryo_mean_dec]
    cryo_stds = [cryo_std_oct, cryo_std_nov, cryo_std_dec]
    cpom_means = [cpom_mean_oct, cpom_mean_nov, cpom_mean_dec]
    cpom_stds = [cpom_std_oct, cpom_std_nov, cpom_std_dec]
    smos_means = [smos_mean_oct, smos_mean_nov, smos_mean_dec]
    smos_stds = [smos_std_oct, smos_std_nov, smos_std_dec]
    awi_means = [awi_mean_oct, awi_mean_nov, awi_mean_dec]
    awi_stds = [awi_std_oct, awi_std_nov, awi_std_dec]
    
    # X locations for the bars month
    x = np.arange(len(months))
    width = 0.1
    
    plt.errorbar(x - width, cryo_means, yerr=cryo_stds, fmt='none', capsize=5, color='black', zorder=1)
    plt.errorbar(x, cpom_means, yerr=cpom_stds, fmt='none', capsize=5, color='black', zorder=1)
    plt.errorbar(x + width, smos_means, yerr=smos_stds,fmt='none', capsize=5, color='black', zorder=1)
    plt.errorbar(x + width*2, awi_means, yerr=awi_stds, fmt='none', capsize=5, color='black', zorder=1)
    
    plt.bar(x - width, cryo_means, width, label='CryoSat-2',color="skyblue", edgecolor='black', zorder=2)
    plt.bar(x, cpom_means, width, label='CPOM', color="orange", edgecolor='black', zorder=2)
    plt.bar(x + width, smos_means, width, label='SMOS', color="green", edgecolor='black', zorder=2)
    plt.bar(x + width*2, awi_means, width, label='AWI', color="red", edgecolor='black', zorder=2)
    
    plt.xticks(x, months)
    plt.ylabel('Sea Ice Thickness [m]')
    plt.legend()
    plt.title('Comparison of 2021 Mean Sea Ice Thickness')
    plt.show()

def scatter_pair(pair_1, pair_2):
    # Flatten the data
    pair_1_flat = pair_1.flatten()
    pair_2_flat = pair_2.flatten()

    # Ensure matching dimensions by interpolating the smaller dataset
    if len(pair_2_flat) < len(pair_1_flat):
        interp_func = interp1d(np.linspace(0, 1, len(pair_2_flat)), pair_2_flat, kind='linear', bounds_error=False, fill_value=np.nan)
        pair_2_flat = interp_func(np.linspace(0, 1, len(pair_1_flat)))
    elif len(pair_1_flat) < len(pair_2_flat):
        interp_func = interp1d(np.linspace(0, 1, len(pair_1_flat)), pair_1_flat, kind='linear', bounds_error=False, fill_value=np.nan)
        pair_1_flat = interp_func(np.linspace(0, 1, len(pair_2_flat)))

    # Remove NaN values (only use pairs where both values exist)
    mask = ~np.isnan(pair_1_flat) & ~np.isnan(pair_2_flat)
    pair_1_filtered = pair_1_flat[mask]
    pair_2_filtered = pair_2_flat[mask]

    # Create scatter plot
    plt.figure(figsize=(8, 6))

    # Plot SMOS data in blue
    plt.scatter(pair_1_filtered, pair_2_filtered, alpha=0.5, edgecolors='k', label='SMOS')

    # Add 1:1 line for comparison
    min_val = min(pair_1_filtered.min(), pair_2_filtered.min())
    max_val = max(pair_1_filtered.max(), pair_2_filtered.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
    
    slope, intercept, r_value, p_value, std_err = linregress(pair_1_filtered, pair_2_filtered)
    mymodel = list(map(lambda x: slope*x + intercept, pair_1_filtered))
    plt.plot(pair_1_filtered, mymodel, 'b-', label="Regression Line")
    # Labels and title
    plt.xlabel('SMOS Sea Ice Thickness (m)')
    plt.ylabel('CryoSat-2 Sea Ice Thickness (m)')
    plt.title('2021 Sea Ice Thickness Comparison: SMOS vs. CryoSat-2')
    plt.xlim(0, 1.35)
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

def BoxPlot(cryo_si, smos_si, cpom_si, awi_si):
    pair_2_flat = cryo_si.flatten()
    pair_1_flat = smos_si.flatten()
    cpom_flat = cpom_si.flatten()
    awi_flat = awi_si.flatten()
    
    pair_2_flat = pair_2_flat[~np.isnan(pair_2_flat)]
    pair_1_flat = pair_1_flat[~np.isnan(pair_1_flat)]
    cpom_flat = cpom_flat[~np.isnan(cpom_flat)]
    awi_flat = awi_flat[~np.isnan(awi_flat)]
    
    data = [pair_2_flat, pair_1_flat, cpom_flat, awi_flat]
    vals, names, xs = [], [], []
    for i, d in enumerate(data):
        vals.append(d)
        names.append(i)
        xs.append(np.random.normal(i + 1, 0.04, len(d)))
    
    plt.figure(figsize=(8, 6))
    plt.boxplot(vals, labels=['CryoSat-2', 'SMOS', 'CPOM', 'AWI'])

    palette = ['sienna', 'darkolivegreen', 'skyblue', 'mediumvioletred']
    for x, val, c in zip(xs, vals, palette):
        plt.scatter(x, val, alpha=0.4, color=c, s=10)

    # Add labels and title
    plt.ylabel("Sea Ice Thickness (m)")
    plt.title("2021 Boxplot Comparison of Sea Ice Thickness Estimates")

    # Show the plot
    plt.show()

    


if __name__ == "__main__":
    #bar_plot()
    #scatter_pair(si_thickness_smos_oct, si_thickness_cryo_oct)
    BoxPlot(si_thickness_cryo_oct, si_thickness_smos_oct, si_thickness_cpom_oct, si_thickness_awi_oct)