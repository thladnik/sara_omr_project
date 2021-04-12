import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import classify_fish
from classify_fish import filter_fish, load_summary
import seaborn as sns
import sys

from auxfuns import *
import matplotlib as mpl
from scipy.stats import sem
from time import sleep
import scipy.io as scio

from analyse_df import *

# GAIN COLOURMAP: SPAT FREQ, TEMP FREQ, RETINAL SPEED
# alles ANGULAR gain

# load Data
base_folder = './data/'
Df = pd.read_hdf('Summary_final.h5', 'all')

# create Df for heatmaplike figure
grps_heat = Df.groupby(['water_height', 'spat_frequency', 'temp_freq'])
Df_gain_heat = pd.DataFrame()
Df_gain_heat['angular_gain_mean'] = grps_heat.apply(get_angular_gain_mean)
Df_gain_heat['absolute_gain_mean'] = grps_heat.apply(get_absolute_gain_mean)
Df_gain_heat['spat_frequency_mean'] = grps_heat.apply(get_spat_frequency_mean)
Df_gain_heat['temp_freq_mean'] = grps_heat.apply(get_temp_freq_mean)
Df_gain_heat['temp_freq_magnitude_mean'] = grps_heat.apply(get_temp_freq_magnitude_mean)
Df_gain_heat['retinal_speed_mean'] = grps_heat.apply(get_retinal_speed_mean)
Df_gain_heat['retinal_speed_magnitude_mean'] = grps_heat.apply(get_retinal_speed_magnitude_mean)
Df_gain_heat['water_height_mean'] = grps_heat.apply(get_waterheight_mean)
Df_gain_heat['u_lin_velocity'] = grps_heat.apply(get_u_lin_velocity)
# das habe ich damals gemacht, weil ich für Dein Skript mean gebraucht habe

# alle absoluten/angular Velocities
Heat_all_abs_velo = Df_gain_heat[(np.isclose(Df_gain_heat.u_lin_velocity, 28) | np.isclose(Df_gain_heat.u_lin_velocity, 143) | np.isclose(Df_gain_heat.u_lin_velocity, 286) | np.isclose(Df_gain_heat.u_lin_velocity, -28) | np.isclose(Df_gain_heat.u_lin_velocity, -143) | np.isclose(Df_gain_heat.u_lin_velocity, -286))]
Heat_all_an_velo = Df_gain_heat[np.isclose(Df_gain_heat.u_lin_velocity, 13.3) | np.isclose(Df_gain_heat.u_lin_velocity, 26.6) | np.isclose(Df_gain_heat.u_lin_velocity, 53.2) | np.isclose(Df_gain_heat.u_lin_velocity, -13.3) | np.isclose(Df_gain_heat.u_lin_velocity, -26.6) | np.isclose(Df_gain_heat.u_lin_velocity, -53.2) | ((np.isclose(Df_gain_heat.water_height_mean, 30)) & (np.isclose(Df_gain_heat.u_lin_velocity, 28))) | np.isclose(Df_gain_heat.u_lin_velocity, 56) | np.isclose(Df_gain_heat.u_lin_velocity, 111.9) | ((np.isclose(Df_gain_heat.water_height_mean, 30)) & (np.isclose(Df_gain_heat.u_lin_velocity, -28))) | np.isclose(Df_gain_heat.u_lin_velocity, -56) | np.isclose(Df_gain_heat.u_lin_velocity, -111.9) | ((np.isclose(Df_gain_heat.water_height_mean, 120)) & (np.isclose(Df_gain_heat.u_lin_velocity, 286))) | ((np.isclose(Df_gain_heat.water_height_mean, 60)) & (np.isclose(Df_gain_heat.u_lin_velocity, 143))) | np.isclose(Df_gain_heat.u_lin_velocity, 71.5) | ((np.isclose(Df_gain_heat.water_height_mean, 120)) & (np.isclose(Df_gain_heat.u_lin_velocity, -286))) | ((np.isclose(Df_gain_heat.water_height_mean, 60)) & (np.isclose(Df_gain_heat.u_lin_velocity, -143))) | np.isclose(Df_gain_heat.u_lin_velocity, -71.5)]

# die jeweilige Wasserhöhe mit allen Velocites
wh30 = Df_gain_heat[(np.isclose(Df_gain_heat.water_height_mean, 30))]
wh60 = Df_gain_heat[(np.isclose(Df_gain_heat.water_height_mean, 60))]
wh120 = Df_gain_heat[(np.isclose(Df_gain_heat.water_height_mean, 120))]

# wasserhöhe 6cm mit getrennten velocities (nur beim Screenshot für Dich verwendet)
Heat_abs_wh60 = Heat_all_abs_velo[(np.isclose(Heat_all_abs_velo.water_height_mean, 60))]
Heat_an_wh60 = Heat_all_an_velo[(np.isclose(Heat_all_an_velo.water_height_mean, 60))]


# plot figures


# WATER HEIGHT 30mm

sns.set_theme()
heatmap_size = (12.5, 11.5)

cmap_scheme = 'viridis'  # turbo
markersize = 17

fig = custom_fig('absolute Gain for spat & temp freq at waterheight 30mm', heatmap_size)
ax = fig.add_subplot(1, 1, 1)
ax.semilogy([] ,[])
# add colorbar
minval = np.floor(np.min(wh30['angular_gain_mean']) * 10) / 10
maxval = np.ceil(np.max(wh30['angular_gain_mean']) * 10) / 10
zticks = np.arange(minval, maxval + 0.01, 0.1)
cax = ax.imshow(np.concatenate((zticks, zticks)).reshape((2, -1)), interpolation='nearest', cmap=cmap_scheme)
cbar = fig.colorbar(cax, ticks=zticks)
cbar.ax.set_ylabel('angular gain at 30mm water height')
cmap = np.asarray(cbar.cmap.colors)
valmap = np.arange(minval, maxval, (maxval - minval) / cmap.shape[0])
plt.cla()  # clears imshow plot, but keeps the colorbar

for sfreq, tfreq, gain in zip(wh30['spat_frequency_mean'], wh30['temp_freq_mean'], wh30['angular_gain_mean']):
    plot_colorpoint(ax, sfreq, tfreq, gain, cmap, valmap)

xlim = [0.008, 0.1]
ylim = [-20, 40]

ax.set_xlabel('spatial frequency [cyc/deg]')
ax.set_ylabel('temporal frequency [cyc/sec]')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
adjust_spines(ax)

fig.savefig('../absolute_gain_heatmap_wh30.svg', format='svg')

plt.show()





# WATER HEIGHT 60mm

sns.set_theme()
heatmap_size = (12.5, 11.5)

cmap_scheme = 'viridis'  # turbo
markersize = 17

fig = custom_fig('absolute Gain for spat & temp freq at waterheight 60mm', heatmap_size)
ax = fig.add_subplot(1, 1, 1)
ax.semilogy([] ,[])
# add colorbar
minval = np.floor(np.min(wh60['angular_gain_mean']) * 10) / 10
maxval = np.ceil(np.max(wh60['angular_gain_mean']) * 10) / 10
zticks = np.arange(minval, maxval + 0.01, 0.1)
cax = ax.imshow(np.concatenate((zticks, zticks)).reshape((2, -1)), interpolation='nearest', cmap=cmap_scheme)
cbar = fig.colorbar(cax, ticks=zticks)
cbar.ax.set_ylabel('angular gain at 60mm water height')
cmap = np.asarray(cbar.cmap.colors)
valmap = np.arange(minval, maxval, (maxval - minval) / cmap.shape[0])
plt.cla()  # clears imshow plot, but keeps the colorbar

for sfreq, tfreq, gain in zip(wh60['spat_frequency_mean'], wh60['temp_freq_mean'], wh60['angular_gain_mean']):
    plot_colorpoint(ax, sfreq, tfreq, gain, cmap, valmap)

xlim = [0.008, 0.1]
ylim = [-20, 40]

ax.set_xlabel('spatial frequency [cyc/deg]')
ax.set_ylabel('temporal frequency [cyc/sec]')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
adjust_spines(ax)

fig.savefig('../absolute_gain_heatmap_wh60.svg', format='svg')

plt.show()




# WATER HEIGHT 120mm

sns.set_theme()
heatmap_size = (12.5, 11.5)

cmap_scheme = 'viridis'  # turbo
markersize = 17

fig = custom_fig('absolute Gain for spat & temp freq at waterheight 120mm', heatmap_size)
ax = fig.add_subplot(1, 1, 1)
ax.semilogy([] ,[])
# add colorbar
minval = np.floor(np.min(wh120['angular_gain_mean']) * 10) / 10
maxval = np.ceil(np.max(wh120['angular_gain_mean']) * 10) / 10
zticks = np.arange(minval, maxval + 0.01, 0.1)
cax = ax.imshow(np.concatenate((zticks, zticks)).reshape((2, -1)), interpolation='nearest', cmap=cmap_scheme)
cbar = fig.colorbar(cax, ticks=zticks)
cbar.ax.set_ylabel('angular gain at 120mm water height')
cmap = np.asarray(cbar.cmap.colors)
valmap = np.arange(minval, maxval, (maxval - minval) / cmap.shape[0])
plt.cla()  # clears imshow plot, but keeps the colorbar

for sfreq, tfreq, gain in zip(wh120['spat_frequency_mean'], wh120['temp_freq_mean'], wh120['angular_gain_mean']):
    plot_colorpoint(ax, sfreq, tfreq, gain, cmap, valmap)

xlim = [0.008, 0.1]
ylim = [-20, 40]

ax.set_xlabel('spatial frequency [cyc/deg]')
ax.set_ylabel('temporal frequency [cyc/sec]')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
adjust_spines(ax)

fig.savefig('../absolute_gain_heatmap_wh120.svg', format='svg')

plt.show()
