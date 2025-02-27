import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import numpy as np
size = 206748
side_length = int(np.sqrt(size))
while size % side_length != 0:
    side_length -= 1
grid_shape = (side_length, size // side_length)
print(f"Calculated grid shape: {grid_shape}")