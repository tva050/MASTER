import os
import glob
import numpy as np
import pandas as pd
import h5py
import scipy.io
import datetime
import matplotlib.pyplot as plt

def plot_2x2():
	fig = plt.figure(figsize=(10, 10))
	box_size = 0.4
	gap = (1-2*box_size)/4
	ax31 = fig.add_axes([gap, gap, box_size, box_size])
	ax32 = fig.add_axes([box_size+3*gap, gap, box_size, box_size])
	ax21 = fig.add_axes([gap, box_size+3*gap, box_size, box_size])
	ax22 = fig.add_axes([box_size+3*gap, box_size+3*gap, box_size, box_size])
	
	plt.show()
 
 
def plot_1x2():
	fig = plt.figure(figsize=(10, 10))
	box_size = 0.4  # for bottom plots (width and height)
	main_height = 0.5  # height of main plot
	gap = (1 - 2 * box_size) / 3  # horizontal gap between boxes
	gap_main = (1 - main_height - box_size) / 3  # vertical gap

	# Main plot (top full width)
	ax_main = fig.add_axes([gap, 2 * gap_main + box_size, 1 - 2 * gap, main_height])

	# Bottom left plot
	ax_left = fig.add_axes([gap, gap_main, box_size, box_size])

	# Bottom right plot
	ax_right = fig.add_axes([2 * gap + box_size, gap_main, box_size, box_size])

	# Optional titles
	ax_main.set_title('Main Plot')
	ax_left.set_title('Bottom Left Plot')
	ax_right.set_title('Bottom Right Plot')

	plt.show()

plot_1x2()

