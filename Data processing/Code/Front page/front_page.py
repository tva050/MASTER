import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np


# set resolution=None to skip processing of boundary datasets.
#m = Basemap(width=12000000,height=9000000,projection='lcc',
#            resolution=None,lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
#m.shadedrelief()
#plt.show()


# Set up the figure and axis
#fig = plt.figure(figsize=(8, 8))
#ax = plt.gca()

# Create a Basemap instance with a north polar stereographic projection
m = Basemap(projection='npstere',
            boundinglat=65,  # Latitude of the map boundary
            lon_0=0,         # Central longitude
            resolution="l",  # Resolution: 'c' (crude), 'l' (low), 'i' (intermediate), 'h' (high), 'f' (full)
            round=True)      # Draw a circular map boundary

#
circle = m.drawmapboundary(linewidth=1.5, color='k')
circle.set_clip_on(False)
m.shadedrelief()
parallels = np.arange(0.,81,10.)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[False,False,False,False], color='gray', linewidth=0.5)
meridians = np.arange(10.,351.,20.)
m.drawmeridians(meridians,labels=[False,False,False,False], color='gray', linewidth=0.5)




# Display the plot
plt.show()
