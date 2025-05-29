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
	ax31 = fig.add_axes([gap, gap, box_size, box_size]) # bottom left
	ax32 = fig.add_axes([box_size+3*gap, gap, box_size, box_size]) # bottom right
	ax21 = fig.add_axes([gap, box_size+3*gap, box_size, box_size]) # top left
	ax22 = fig.add_axes([box_size+3*gap, box_size+3*gap, box_size, box_size]) # top right
	
	plt.show()
 
def plot_2():
    fig = plt.figure(figsize=(10, 5))  # 10x10 figure size
    box_width = 0.4  # Width of each subplot
    gap = (1 - 2 * box_width) / 3  # Gap between plots and edges

    ax1 = fig.add_axes([gap, 0.2, box_width, 0.6])
    ax2 = fig.add_axes([2 * gap + box_width, 0.2, box_width, 0.6])

    
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


def plot_2x1():
    fig = plt.figure(figsize=(6.733, 7))  # your fixed size

    # Dimensions
    bar_w, bar_h = 0.38, 0.22
    hist_w, hist_h = 0.15, 0.12
    mid_bar_w, mid_bar_h = 0.5, 0.22

    # Top bar plots (left and right)
    ax1 = fig.add_axes([0.08, 0.75, bar_w, bar_h])
    ax2 = fig.add_axes([0.54, 0.75, bar_w, bar_h])

    # Row of 4 histograms - centered horizontally
    start_x_hist_row = 0.5 - (2 * hist_w + 1.5 * 0.04)  # total width: 4*hist + 3*gap
    ax3 = fig.add_axes([start_x_hist_row + 0*(hist_w + 0.04), 0.57, hist_w, hist_h])
    ax4 = fig.add_axes([start_x_hist_row + 1*(hist_w + 0.04), 0.57, hist_w, hist_h])
    ax5 = fig.add_axes([start_x_hist_row + 2*(hist_w + 0.04), 0.57, hist_w, hist_h])
    ax6 = fig.add_axes([start_x_hist_row + 3*(hist_w + 0.04), 0.57, hist_w, hist_h])

    # Large middle bar plot
    ax7 = fig.add_axes([0.25, 0.3, mid_bar_w, mid_bar_h])

    # Bottom 2 histograms - centered
    total_width_bottom = 2 * hist_w + 0.04
    start_x_bottom = 0.5 - total_width_bottom / 2
    ax8 = fig.add_axes([start_x_bottom, 0.12, hist_w, hist_h])
    ax9 = fig.add_axes([start_x_bottom + hist_w + 0.04, 0.12, hist_w, hist_h])

    # Optional: Titles
    for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9], start=1):
        ax.set_title(f'Ax{i}', fontsize=8)

    plt.show()

#plot_2x2()
#plot_2()
#plot_1x2()
plot_2x1()
