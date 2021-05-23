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

if __name__ == '__main__':

# load data
    base_folder = './data/'
    Df = pd.read_hdf('Summary_final.h5', 'by_subparticle')



# assumption fish swimming height = water height (surface)

# create Df for heatmaplike figure
    grps_heat = Df.groupby(['water_height', 'stim_ang_sf', 'stim_ang_tf'])
    Df_gain_heat = pd.DataFrame()
    Df_gain_heat['ang_gain_mean'] = grps_heat.apply(lambda df: df.x_ang_gain.mean())
    Df_gain_heat['lin_gain_mean'] = grps_heat.apply(lambda df: df.x_lin_gain.mean())
    Df_gain_heat['spat_freq_mean'] = grps_heat.apply((lambda df: df.stim_ang_sf.mean()))
    Df_gain_heat['temp_freq_mean'] = grps_heat.apply((lambda df: df.stim_ang_tf.mean()))
    Df_gain_heat['temp_freq_magnitude_mean'] = Df_gain_heat['temp_freq_mean'].abs()
    Df_gain_heat['retinal_speed_mean'] = grps_heat.apply((lambda df: df.stim_ang_vel_int.mean()))
    Df_gain_heat['retinal_speed_magnitude_mean'] = Df_gain_heat['retinal_speed_mean'].abs()
    Df_gain_heat['water_height_mean'] = grps_heat.apply((lambda df: df.water_height.mean()))
    Df_gain_heat['stim_lin_vel'] = grps_heat.apply((lambda df: df.stim_lin_vel_int.mean()))

# filter data for each plot
    wh30 = Df_gain_heat[(np.isclose(Df_gain_heat.water_height_mean, 30))]
    wh60 = Df_gain_heat[(np.isclose(Df_gain_heat.water_height_mean, 60))]
    wh120 = Df_gain_heat[(np.isclose(Df_gain_heat.water_height_mean, 120))]

# remove overlap at 60mm
    wh60_26 = wh60[(np.not_equal(wh60.stim_lin_vel, -28)) & (np.not_equal(wh60.stim_lin_vel, 28))]
    wh60_28 = wh60[(np.not_equal(wh60.stim_lin_vel, -26)) & (np.not_equal(wh60.stim_lin_vel, 26))]


    import IPython
    IPython.embed()



#todo: überlappung bei 60mm wegmachen?, danach individual analysis falls sinnvoll


# plot figures

# WATER HEIGHT 30mm

    sns.set_theme()
    heatmap_size = (12.5, 11.5)

    cmap_scheme = 'viridis'  # turbo
    markersize = 17

    fig = custom_fig('angular gain for spat & temp freq at water height 30mm', heatmap_size)
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy([] ,[])
    # add colorbar
    minval = np.floor(np.min(wh30['ang_gain_mean']) * 10) / 10
    maxval = np.ceil(np.max(wh30['ang_gain_mean']) * 10) / 10
    zticks = np.arange(minval, maxval + 0.01, 0.1)
    cax = ax.imshow(np.concatenate((zticks, zticks)).reshape((2, -1)), interpolation='nearest', cmap=cmap_scheme)
    # warum ist GAIN bis zur 4 und aber nichts gelbes dabei, also Gain.max bei 3 und grün (bzw bei 60 oder 120mm).
    # Wahrscheinlich weil max bei 3.3 oder so... ---> noch anpassen
    cbar = fig.colorbar(cax, ticks=zticks)
    cbar.ax.set_ylabel('angular gain at 30mm water height')
    cmap = np.asarray(cbar.cmap.colors)
    valmap = np.arange(minval, maxval, (maxval - minval) / cmap.shape[0])
    plt.cla()  # clears imshow plot, but keeps the colorbar

    for sfreq, tfreq, gain in zip(wh30['spat_freq_mean'], wh30['temp_freq_mean'], wh30['ang_gain_mean']):
        plot_colorpoint(ax, sfreq, tfreq, gain, cmap, valmap)

    xlim = [0.008, 0.1]
    ylim = [-20, 10]

    ax.set_xlabel('spatial frequency [cyc/deg]')
    ax.set_ylabel('temporal frequency [cyc/sec]')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    adjust_spines(ax)

    #fig.savefig('../angular_gain_heatmap_wh30.svg', format='svg')

    plt.show()




# WATER HEIGHT 60mm

    sns.set_theme()
    heatmap_size = (12.5, 11.5)

    cmap_scheme = 'viridis'  # turbo
    markersize = 17

    fig = custom_fig('angular gain for spat & temp freq at water height 60mm with 28 deg per sec', heatmap_size)
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy([], [])
    # add colorbar
    minval = np.floor(np.min(wh60_28['ang_gain_mean']) * 10) / 10
    maxval = np.ceil(np.max(wh60_28['ang_gain_mean']) * 10) / 10
    zticks = np.arange(minval, maxval + 0.01, 0.1)
    cax = ax.imshow(np.concatenate((zticks, zticks)).reshape((2, -1)), interpolation='nearest', cmap=cmap_scheme)
    cbar = fig.colorbar(cax, ticks=zticks)
    cbar.ax.set_ylabel('angular gain at 60mm water height')
    cmap = np.asarray(cbar.cmap.colors)
    valmap = np.arange(minval, maxval, (maxval - minval) / cmap.shape[0])
    plt.cla()  # clears imshow plot, but keeps the colorbar

    for sfreq, tfreq, gain in zip(wh60_28['spat_freq_mean'], wh60_28['temp_freq_mean'], wh60_28['ang_gain_mean']):
        plot_colorpoint(ax, sfreq, tfreq, gain, cmap, valmap)

    xlim = [0.008, 0.1]
    ylim = [-20, 10]

    ax.set_xlabel('spatial frequency [cyc/deg]')
    ax.set_ylabel('temporal frequency [cyc/sec]')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    adjust_spines(ax)

    # fig.savefig('../angular_gain_heatmap_wh60.svg', format='svg')

    plt.show()





# WATER HEIGHT 120mm

    sns.set_theme()
    heatmap_size = (12.5, 11.5)

    cmap_scheme = 'viridis'  # turbo
    markersize = 17

    fig = custom_fig('angular gain for spat & temp freq at water height 120mm', heatmap_size)
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy([], [])
    # add colorbar
    minval = np.floor(np.min(wh120['ang_gain_mean']) * 10) / 10
    maxval = np.ceil(np.max(wh120['ang_gain_mean']) * 10) / 10
    zticks = np.arange(minval, maxval, 0.1)
    cax = ax.imshow(np.concatenate((zticks, zticks)).reshape((2, -1)), interpolation='nearest', cmap=cmap_scheme)
    cbar = fig.colorbar(cax, ticks=zticks)
    cbar.ax.set_ylabel('angular gain at 120mm water height')
    cmap = np.asarray(cbar.cmap.colors)
    valmap = np.arange(minval, maxval, (maxval - minval) / cmap.shape[0])
    plt.cla()  # clears imshow plot, but keeps the colorbar

    for sfreq, tfreq, gain in zip(wh120['spat_freq_mean'], wh120['temp_freq_mean'], wh120['ang_gain_mean']):
        plot_colorpoint(ax, sfreq, tfreq, gain, cmap, valmap)

    xlim = [0.008, 0.1]
    ylim = [-20, 10]

    ax.set_xlabel('spatial frequency [cyc/deg]')
    ax.set_ylabel('temporal frequency [cyc/sec]')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    adjust_spines(ax)

    # fig.savefig('../angular_gain_heatmap_wh120.svg', format='svg')

    plt.show()



# linear gain (falls überhaupt interessant)

# WATER HEIGHT 30mm

    sns.set_theme()
    heatmap_size = (12.5, 11.5)

    cmap_scheme = 'viridis'  # turbo
    markersize = 17

    fig = custom_fig('linear gain for spat & temp freq at water height 30mm', heatmap_size)
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy([] ,[])
    # add colorbar
    minval = np.floor(np.min(wh30['lin_gain_mean']) * 10) / 10
    maxval = np.ceil(np.max(wh30['lin_gain_mean']) * 10) / 10
    zticks = np.arange(minval, maxval + 0.01, 0.1)
    cax = ax.imshow(np.concatenate((zticks, zticks)).reshape((2, -1)), interpolation='nearest', cmap=cmap_scheme)
    # warum ist GAIN bis zur 4 und aber nichts gelbes dabei, also Gain.max bei 3 und grün (bzw bei 60 oder 120mm).
    # Wahrscheinlich weil max bei 3.3 oder so... ---> noch anpassen
    cbar = fig.colorbar(cax, ticks=zticks)
    cbar.ax.set_ylabel('linear gain at 30mm water height')
    cmap = np.asarray(cbar.cmap.colors)
    valmap = np.arange(minval, maxval, (maxval - minval) / cmap.shape[0])
    plt.cla()  # clears imshow plot, but keeps the colorbar

    for sfreq, tfreq, gain in zip(wh30['spat_freq_mean'], wh30['temp_freq_mean'], wh30['lin_gain_mean']):
        plot_colorpoint(ax, sfreq, tfreq, gain, cmap, valmap)

    xlim = [0.008, 0.1]
    ylim = [-20, 10]

    ax.set_xlabel('spatial frequency [cyc/deg]')
    ax.set_ylabel('temporal frequency [cyc/sec]')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    adjust_spines(ax)

    #fig.savefig('../linear_gain_heatmap_wh30.svg', format='svg')

    plt.show()





# WATER HEIGHT 60mm

    sns.set_theme()
    heatmap_size = (12.5, 11.5)

    cmap_scheme = 'viridis'  # turbo
    markersize = 17

    fig = custom_fig('linear gain for spat & temp freq at water height 60mm', heatmap_size)
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy([], [])
    # add colorbar
    minval = np.floor(np.min(wh60['lin_gain_mean']) * 10) / 10
    maxval = np.ceil(np.max(wh60['lin_gain_mean']) * 10) / 10
    zticks = np.arange(minval, maxval + 0.01, 0.1)
    cax = ax.imshow(np.concatenate((zticks, zticks)).reshape((2, -1)), interpolation='nearest', cmap=cmap_scheme)
    cbar = fig.colorbar(cax, ticks=zticks)
    cbar.ax.set_ylabel('linear gain at 60mm water height')
    cmap = np.asarray(cbar.cmap.colors)
    valmap = np.arange(minval, maxval, (maxval - minval) / cmap.shape[0])
    plt.cla()  # clears imshow plot, but keeps the colorbar

    for sfreq, tfreq, gain in zip(wh60['spat_freq_mean'], wh60['temp_freq_mean'], wh60['lin_gain_mean']):
        plot_colorpoint(ax, sfreq, tfreq, gain, cmap, valmap)

    xlim = [0.008, 0.1]
    ylim = [-20, 10]

    ax.set_xlabel('spatial frequency [cyc/deg]')
    ax.set_ylabel('temporal frequency [cyc/sec]')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    adjust_spines(ax)

    # fig.savefig('../linear_gain_heatmap_wh60.svg', format='svg')

    plt.show()





# WATER HEIGHT 120mm

    sns.set_theme()
    heatmap_size = (12.5, 11.5)

    cmap_scheme = 'viridis'  # turbo
    markersize = 17

    fig = custom_fig('linear gain for spat & temp freq at water height 120mm', heatmap_size)
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy([], [])
    # add colorbar
    minval = np.floor(np.min(wh120['lin_gain_mean']) * 10) / 10
    maxval = np.ceil(np.max(wh120['lin_gain_mean']) * 10) / 10
    zticks = np.arange(minval, maxval, 0.1)
    cax = ax.imshow(np.concatenate((zticks, zticks)).reshape((2, -1)), interpolation='nearest', cmap=cmap_scheme)
    cbar = fig.colorbar(cax, ticks=zticks)
    cbar.ax.set_ylabel('linear gain at 120mm water height')
    cmap = np.asarray(cbar.cmap.colors)
    valmap = np.arange(minval, maxval, (maxval - minval) / cmap.shape[0])
    plt.cla()  # clears imshow plot, but keeps the colorbar

    for sfreq, tfreq, gain in zip(wh120['spat_freq_mean'], wh120['temp_freq_mean'], wh120['lin_gain_mean']):
        plot_colorpoint(ax, sfreq, tfreq, gain, cmap, valmap)

    xlim = [0.008, 0.1]
    ylim = [-20, 10]

    ax.set_xlabel('spatial frequency [cyc/deg]')
    ax.set_ylabel('temporal frequency [cyc/sec]')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    adjust_spines(ax)

    # fig.savefig('../linear_gain_heatmap_wh120.svg', format='svg')

    plt.show()


quit()

