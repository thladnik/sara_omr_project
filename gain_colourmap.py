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

    sns.set(font_scale=1.3)
    sns.set_style('ticks')

# assumption fish swimming height = water height (surface) bzw jetzt -10mm

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
    wh60 = Df_gain_heat[(np.isclose(Df_gain_heat.water_height_mean, 60))].copy()
    wh120 = Df_gain_heat[(np.isclose(Df_gain_heat.water_height_mean, 120))]

# fix colourbar at 30mm (changes data(!))
    wh30_ = wh30.copy()
    wh30_.loc[np.isclose(wh30_.lin_gain_mean, 0.530369454209402), 'lin_gain_mean'] = 0.400069454209402
    wh30_.loc[np.isclose(wh30_.lin_gain_mean, 0.461839320083067), 'lin_gain_mean'] = 0.400069454209402




# remove overlap at 60mm
#     wh60_26 = wh60[(np.not_equal(wh60.stim_lin_vel, -28)) & (np.not_equal(wh60.stim_lin_vel, 28))]
#     wh60_28 = wh60[(np.not_equal(wh60.stim_lin_vel, -26)) & (np.not_equal(wh60.stim_lin_vel, 26))]

    wh60_bin = wh60.copy()
    # wh60_bin.loc[wh60_bin['temp_freq_mean'] == 0.26, 'temp_freq_mean'] = 0.25
    # wh60_bin.loc[wh60_bin['temp_freq_mean'] == -0.26, 'temp_freq_mean'] = -0.25
    # wh60_bin.loc[np.isclose(wh60_bin.temp_freq_mean, 0.52), 'temp_freq_mean'] = 0.5
    # wh60_bin.loc[np.isclose(wh60_bin.temp_freq_mean, -0.52), 'temp_freq_mean'] = -0.5
    # wh60_bin.loc[np.isclose(wh60_bin.temp_freq_mean, 1.04), 'temp_freq_mean'] = 1
    # wh60_bin.loc[np.isclose(wh60_bin.temp_freq_mean, -1.04), 'temp_freq_mean'] = -1

    wh60_bin.loc[np.isclose(wh60_bin.ang_gain_mean, 0.32395726270521), 'ang_gain_mean'] = 0.278261065430878
    wh60_bin.loc[np.isclose(wh60_bin.ang_gain_mean, 0.187520911556822), 'ang_gain_mean'] = 0.278261065430878
    wh60_bin.loc[np.isclose(wh60_bin.ang_gain_mean, 0.29399312373556), 'ang_gain_mean'] = 0.278261065430878
    wh60_bin.loc[np.isclose(wh60_bin.ang_gain_mean, 0.307572963725922), 'ang_gain_mean'] = 0.278261065430878

    wh60_bin.loc[np.isclose(wh60_bin.ang_gain_mean, 0.222046297666656), 'ang_gain_mean'] = 0.284328899223394
    wh60_bin.loc[np.isclose(wh60_bin.ang_gain_mean, 0.297069287892534), 'ang_gain_mean'] = 0.284328899223394
    wh60_bin.loc[np.isclose(wh60_bin.ang_gain_mean, 0.317888522527042), 'ang_gain_mean'] = 0.284328899223394
    wh60_bin.loc[np.isclose(wh60_bin.ang_gain_mean, 0.300311488807343), 'ang_gain_mean'] = 0.284328899223394

#sf 0.01
    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.136897726681799), 'lin_gain_mean'] = 0.16045882005314
    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.208960869959905), 'lin_gain_mean'] = 0.16045882005314
    wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.154654181552564), 'lin_gain_mean'] = 0.147988341785429
    wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.141322502018293), 'lin_gain_mean'] = 0.147988341785429

    wh60_bin.loc[np.isclose(wh60_bin.temp_freq_mean, 0.271883861488673), 'temp_freq_mean'] = 0.278752888104045
    wh60_bin.loc[np.isclose(wh60_bin.temp_freq_mean, 0.285621914719417), 'temp_freq_mean'] = 0.278752888104045

    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.32395726270521, atol=1e-02), 'lin_gain_mean'] = 0.259742927398106
    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.187520911556822, atol=1e-02), 'lin_gain_mean'] = 0.259742927398106
    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.29399312373556, atol=1e-02), 'lin_gain_mean'] = 0.259742927398106
    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.307572963725922, atol=1e-02), 'lin_gain_mean'] = 0.259742927398106
    wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.300817458226267, atol=1e-02), 'lin_gain_mean'] = 0.290817458226267

#sf 0.02
    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.300817458226267, atol=1e-02), 'lin_gain_mean'] = 0.259742927398106
    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.176241458230096, atol=1e-02), 'lin_gain_mean'] = 0.259742927398106
    wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.276309326819135), 'lin_gain_mean'] = 0.280956396568031
    wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.285603466316927), 'lin_gain_mean'] = 0.280956396568031

    wh60_bin.loc[np.isclose(wh60_bin.temp_freq_mean, 0.543767722977347), 'temp_freq_mean'] = 0.55750577620809
    wh60_bin.loc[np.isclose(wh60_bin.temp_freq_mean, 0.571243829438833), 'temp_freq_mean'] = 0.55750577620809

#sf: 0.04
    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.206185847833324, atol=1e-02), 'lin_gain_mean'] = 0.265753595736816
    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.279200458545615, atol=1e-02), 'lin_gain_mean'] = 0.265753595736816
    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.298767408390077, atol=1e-02), 'lin_gain_mean'] = 0.265753595736816
    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.278860668178247, atol=1e-02), 'lin_gain_mean'] = 0.265753595736816
    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.222046297666656, atol=1e-02), 'lin_gain_mean'] = 0.288814038284162
    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.297069287892534, atol=1e-02), 'lin_gain_mean'] = 0.288814038284162
    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.317888522527042, atol=1e-02), 'lin_gain_mean'] = 0.288814038284162
    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.300311488807343, atol=1e-02), 'lin_gain_mean'] = 0.288814038284162
    wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.298767408390077), 'lin_gain_mean'] = 0.288814038284162
    wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.278860668178247), 'lin_gain_mean'] = 0.288814038284162


    wh60_bin.loc[np.isclose(wh60_bin.temp_freq_mean, 1.08753544595469), 'temp_freq_mean'] = 1.11501155241618
    wh60_bin.loc[np.isclose(wh60_bin.temp_freq_mean, 1.14248765887767), 'temp_freq_mean'] = 1.11501155241618



    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.32395726270521), 'lin_gain_mean'] = 0.259742927398106
    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.187520911556822), 'lin_gain_mean'] = 0.259742927398106
    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.29399312373556), 'lin_gain_mean'] = 0.259742927398106
    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.307572963725922), 'lin_gain_mean'] = 0.259742927398106
    #
    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.222046297666656), 'lin_gain_mean'] = 0.265753595736816
    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.297069287892534), 'lin_gain_mean'] = 0.265753595736816
    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.317888522527042), 'lin_gain_mean'] = 0.265753595736816
    # wh60_bin.loc[np.isclose(wh60_bin.lin_gain_mean, 0.300311488807343), 'lin_gain_mean'] = 0.265753595736816

    import IPython
    IPython.embed()

    quit()

    #wh60_bin.to_excel('wh60_binn.xlsx')

    #wh30.to_excel('wh30.xlsx')

    sns.set_theme(style='whitegrid')
    sns.set(font_scale=1.3)

    sns.despine()




# LINEAR gain

# WATER HEIGHT 30mm

    sns.set_theme(style='whitegrid')
    sns.set(font_scale=1.3)
    #sns.set_style('ticks')
    heatmap_size = (13.5, 14.5)

    cmap_scheme = 'viridis'  # turbo
    markersize = 17

    fig = custom_fig('absolute gain colourmap 30mm', heatmap_size)
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy([] ,[])
    # add colorbar
    minval = np.floor(np.min(wh30_['lin_gain_mean']) * 10) / 10
    maxval = np.ceil(np.max(wh30_['lin_gain_mean']) * 10) / 10
    zticks = np.arange(minval, maxval + 0.01, 0.1)
    cax = ax.imshow(np.concatenate((zticks, zticks)).reshape((2, -1)), interpolation='nearest', cmap=cmap_scheme)
    # warum ist GAIN bis zur 4 und aber nichts gelbes dabei, also Gain.max bei 3 und grün (bzw bei 60 oder 120mm).
    # Wahrscheinlich weil max bei 3.3 oder so... ---> noch anpassen
    cbar = fig.colorbar(cax, ticks=zticks)
    cbar.ax.set_ylabel('absolute gain')
    cmap = np.asarray(cbar.cmap.colors)
    valmap = np.arange(minval, maxval, (maxval - minval) / cmap.shape[0])
    plt.cla()  # clears imshow plot, but keeps the colorbar

    for sfreq, tfreq, gain in zip(wh30_['spat_freq_mean'], wh30_['temp_freq_mean'], wh30_['lin_gain_mean']):
        plot_colorpoint(ax, sfreq, tfreq, gain, cmap, valmap)

    xlim = [0.008, 0.1]
    ylim = [-20, 10]

    ax.set_xlabel('spatial frequency [cyc/deg]')
    ax.set_ylabel('temporal frequency [cyc/sec]')
    ax.set_title('water height = 30 mm')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    adjust_spines(ax)

    #fig.savefig('../linear_gain_heatmap_wh30.svg', format='svg')

    plt.show()





# WATER HEIGHT 60mm

    sns.set_theme(style='whitegrid')
    heatmap_size = (13.5, 14.5)

    sns.set(font_scale=1.3)
    #sns.set_style('ticks')
    # 3 zeilen auch bei 120 und dann jweils plotten und linien wegmachen
    cmap_scheme = 'viridis'  # turbo
    markersize = 17

    fig = custom_fig('absolute gain colourmap 60mm', heatmap_size)
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy([], [])
    # add colorbar
    minval = np.floor(np.min(wh60_bin['lin_gain_mean']) * 10) / 10
    maxval = np.ceil(np.max(wh60_bin['lin_gain_mean']) * 10) / 10
    zticks = np.arange(minval, maxval + 0.01, 0.1)
    cax = ax.imshow(np.concatenate((zticks, zticks)).reshape((2, -1)), interpolation='nearest', cmap=cmap_scheme)
    cbar = fig.colorbar(cax, ticks=zticks)
    cbar.ax.set_ylabel('absolute gain')
    cmap = np.asarray(cbar.cmap.colors)
    valmap = np.arange(minval, maxval, (maxval - minval) / cmap.shape[0])
    plt.cla()  # clears imshow plot, but keeps the colorbar

    for sfreq, tfreq, gain in zip(wh60_bin['spat_freq_mean'], wh60_bin['temp_freq_mean'], wh60_bin['lin_gain_mean']):
        plot_colorpoint(ax, sfreq, tfreq, gain, cmap, valmap)

    xlim = [0.008, 0.1]
    ylim = [-20, 10]

    ax.set_xlabel('spatial frequency [cyc/deg]')
    ax.set_ylabel('temporal frequency [cyc/sec]')
    ax.set_title('water height = 60 mm')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    adjust_spines(ax)

    # fig.savefig('../linear_gain_heatmap_wh60.svg', format='svg')

    plt.show()





# WATER HEIGHT 120mm

    sns.set_theme(style='whitegrid')
    heatmap_size = (13.5, 14.5)
    sns.set(font_scale=1.3)
    cmap_scheme = 'viridis'  # turbo
    markersize = 17

    fig = custom_fig('absolute gain colourmap 120mm', heatmap_size)
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy([], [])
    # add colorbar
    minval = np.floor(np.min(wh120['lin_gain_mean']) * 10) / 10
    maxval = np.ceil(np.max(wh120['lin_gain_mean']) * 10) / 10
    zticks = np.arange(minval, maxval + 0.01, 0.1)
    cax = ax.imshow(np.concatenate((zticks, zticks)).reshape((2, -1)), interpolation='nearest', cmap=cmap_scheme)
    cbar = fig.colorbar(cax, ticks=zticks)
    cbar.ax.set_ylabel('absolute gain')
    cmap = np.asarray(cbar.cmap.colors)
    valmap = np.arange(minval, maxval, (maxval - minval) / cmap.shape[0])
    plt.cla()  # clears imshow plot, but keeps the colorbar

    for sfreq, tfreq, gain in zip(wh120['spat_freq_mean'], wh120['temp_freq_mean'], wh120['lin_gain_mean']):
        plot_colorpoint(ax, sfreq, tfreq, gain, cmap, valmap)

    xlim = [0.008, 0.1]
    ylim = [-20, 10]

    ax.set_xlabel('spatial frequency [cyc/deg]')
    ax.set_ylabel('temporal frequency [cyc/sec]')
    ax.set_title('water height = 120 mm')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    adjust_spines(ax)

    # fig.savefig('../linear_gain_heatmap_wh120.svg', format='svg')

    plt.show()



# ANGULAR Gain


# WATER HEIGHT 30mm

    sns.set_theme(style='whitegrid')
    heatmap_size = (12.5, 14.5)

    cmap_scheme = 'viridis'  # turbo
    markersize = 17

    fig = custom_fig('relative gain for spat & temp freq at water height 30mm', heatmap_size)
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy([] ,[])
    # add colorbar
    # minval = np.floor(np.min(Df_gain_heat['ang_gain_mean']) * 10) / 10
    # maxval = np.ceil(np.max(Df_gain_heat['ang_gain_mean']) * 10) / 10
    minval = np.floor(0.0 * 10) / 10
    maxval = np.ceil(0.41 * 10) / 10
    zticks = np.arange(minval, maxval, 0.1)
    cax = ax.imshow(np.concatenate((zticks, zticks)).reshape((2, -1)), interpolation='nearest', cmap=cmap_scheme)
    # warum ist GAIN bis zur 4 und aber nichts gelbes dabei, also Gain.max bei 3 und grün (bzw bei 60 oder 120mm).
    # Wahrscheinlich weil max bei 3.3 oder so... ---> noch anpassen
    cbar = fig.colorbar(cax, ticks=zticks)
    cbar.ax.set_ylabel('relative gain [mm/deg]')
    cmap = np.asarray(cbar.cmap.colors)
    # valmap = np.arange(minval, maxval, (maxval - minval) / cmap.shape[0])
    valmap = np.arange(0, 0.41, (0.41 - 0) / cmap.shape[0])
    plt.cla()  # clears imshow plot, but keeps the colorbar

    for sfreq, tfreq, gain in zip(wh30['spat_freq_mean'], wh30['temp_freq_mean'], wh30['ang_gain_mean']):
        plot_colorpoint(ax, sfreq, tfreq, gain, cmap, valmap)

    xlim = [0.008, 0.1]
    ylim = [-20, 10]

    ax.set_xlabel('spatial frequency [cyc/deg]')
    ax.set_ylabel('temporal frequency [cyc/sec]')
    ax.set_title('water height = 30 mm')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    adjust_spines(ax)

    #fig.savefig('../angular_gain_heatmap_wh30.svg', format='svg')

    plt.show()




# WATER HEIGHT 60mm

    sns.set_theme(style='whitegrid')
    heatmap_size = (12.5, 14.5)

    cmap_scheme = 'viridis'  # turbo
    markersize = 17

    fig = custom_fig('relative gain for spat & temp freq at water height (bin) 60mm', heatmap_size)
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy([], [])
    # add colorbar
    minval = np.floor(0.0 * 10) / 10
    maxval = np.ceil(0.4 * 10) / 10
    zticks = np.arange(minval, maxval + 0.01, 0.1)
    cax = ax.imshow(np.concatenate((zticks, zticks)).reshape((2, -1)), interpolation='nearest', cmap=cmap_scheme)
    cbar = fig.colorbar(cax, ticks=zticks)
    cbar.ax.set_ylabel('relative gain [mm/deg]')
    cmap = np.asarray(cbar.cmap.colors)
    # valmap = np.arange(minval, maxval, (maxval - minval) / cmap.shape[0])
    valmap = np.arange(0, 0.41, (0.41 - 0) / cmap.shape[0])
    plt.cla()  # clears imshow plot, but keeps the colorbar

    for sfreq, tfreq, gain in zip(wh60_bin['spat_freq_mean'], wh60_bin['temp_freq_mean'], wh60_bin['ang_gain_mean']):
        plot_colorpoint(ax, sfreq, tfreq, gain, cmap, valmap)

    xlim = [0.008, 0.1]
    ylim = [-20, 10]

    ax.set_xlabel('spatial frequency [cyc/deg]')
    ax.set_ylabel('temporal frequency [cyc/sec]')
    ax.set_title('water height = 60 mm')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    adjust_spines(ax)

    # fig.savefig('../angular_gain_heatmap_wh60.svg', format='svg')

    plt.show()





# WATER HEIGHT 120mm

    sns.set_theme(style='whitegrid')
    heatmap_size = (13.5, 14.5)

    cmap_scheme = 'viridis'  # turbo
    markersize = 17

    fig = custom_fig('angular gain for spat & temp freq at water height 120mm', heatmap_size)
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy([], [])
    # add colorbar
    minval = np.floor(0.0 * 10) / 10
    maxval = np.ceil(0.41 * 10) / 10
    zticks = np.arange(minval, maxval, 0.1)
    cax = ax.imshow(np.concatenate((zticks, zticks)).reshape((2, -1)), interpolation='nearest', cmap=cmap_scheme)
    cbar = fig.colorbar(cax, ticks=zticks)
    cbar.ax.set_ylabel('angular gain [mm/deg]]')
    cmap = np.asarray(cbar.cmap.colors)
    # valmap = np.arange(minval, maxval, (maxval - minval) / cmap.shape[0])
    valmap = np.arange(0, 0.41, (0.41 - 0) / cmap.shape[0])
    plt.cla()  # clears imshow plot, but keeps the colorbar

    for sfreq, tfreq, gain in zip(wh120['spat_freq_mean'], wh120['temp_freq_mean'], wh120['ang_gain_mean']):
        plot_colorpoint(ax, sfreq, tfreq, gain, cmap, valmap)

    xlim = [0.008, 0.1]
    ylim = [-20, 10]

    ax.set_xlabel('spatial frequency [cyc/deg]')
    ax.set_ylabel('temporal frequency [cyc/sec]')
    ax.set_title('water height = 120 mm')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    adjust_spines(ax)

    # fig.savefig('../angular_gain_heatmap_wh120.svg', format='svg')

    plt.show()



quit()

