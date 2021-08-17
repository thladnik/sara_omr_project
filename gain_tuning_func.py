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

# load Data
    base_folder = './data/'
    Df = pd.read_hdf('Summary_final.h5', 'by_subparticle')

# bin data
    stim_lin_vel_int_bins = pd.cut(Df.stim_lin_vel_int, 160)
    Df['stim_lin_vel_int_bins'] = [stim_lin_vel_bins_calc_mid.mid for stim_lin_vel_bins_calc_mid in stim_lin_vel_int_bins]

    stim_ang_vel_int_bins = pd.cut(Df.stim_ang_vel_int, 90)
    Df['stim_ang_vel_int_bins'] = [stim_ang_vel_int_bins_calc_mid.mid for stim_ang_vel_int_bins_calc_mid in stim_ang_vel_int_bins]

    stim_ang_tf_bins = pd.cut(Df.stim_ang_tf, 80)
    Df['stim_ang_tf_bins'] = [stim_ang_tf_bins_calc_mid.mid for stim_ang_tf_bins_calc_mid in stim_ang_tf_bins]


# set size for plots
    sns.set(font_scale=1.4)
    sns.set_style('ticks')


    import IPython
    IPython.embed()

# assumption fish swimming height = water height (surface) -10mm (depends on analyse_df!)

# GAIN TUNING FUNCTION

# linear Gain (gegen lin vel)
    ax = sns.relplot(data=Df, x='stim_lin_vel_int_bins', y='x_lin_gain', hue='water_height', palette='dark', col='stim_ang_sf',kind='line')
    ax.set(xlabel='linear stimulus velocity [mm/sec]', ylabel='absolute gain')
    #ax.set_titles('spatial frequency = {col_name} cyc/deg', fontsize=14)
    ax.set_titles('spatial frequency = {col_name} cyc/deg')
    ax._legend.set_title('water height\n[mm]')
    new_labels = ['30', '60', '120']
    for t, l in zip(ax._legend.texts, new_labels): t.set_text(l)
    plt.tight_layout()
    plt.show()
# (water height in col)
    ax = sns.relplot(data=Df, x='stim_lin_vel_int_bins', y='x_lin_gain', hue='stim_ang_sf', palette='dark', col='water_height',kind='line')
    ax.set(xlabel='linear stimulus velocity [mm/sec]', ylabel='absolute gain')
    ax.set_titles('water height = {col_name} mm') #leider mit kommastelle
    ax._legend.set_title('spatial\nfrequency\n[cyc/deg]')
    plt.tight_layout()
    plt.show()

# angular Gain (gegen retinal speed)
    ax = sns.relplot(data=Df, x='stim_ang_vel_int_bins', y='x_ang_gain', hue='water_height', palette='dark', col='stim_ang_sf', kind='line')
    ax.set(xlabel='retinal speed [deg/sec]', ylabel='relative gain [mm/deg]')
    ax.set_titles('spatial frequency = {col_name} cyc/deg')
    ax._legend.set_title('water height\n[mm]')
    new_labels = ['30', '60', '120']
    for t, l in zip(ax._legend.texts, new_labels): t.set_text(l)
    plt.tight_layout()
    plt.show()


# GAIN GEGEN TF
    stim_ang_tf_bins = pd.cut(Df.stim_ang_tf, 80)
    Df['stim_ang_tf_bins'] = [stim_ang_tf_bins_calc_mid.mid for stim_ang_tf_bins_calc_mid in stim_ang_tf_bins]

    ax = sns.relplot(data=Df, x='stim_ang_tf_bins', y='x_lin_gain', hue='water_height', palette='dark', col='stim_ang_sf',kind='line')
    ax.set(xlabel='temporal frequency [cyc/sec]', ylabel='absolute gain')
    ax.set_titles('spatial frequency = {col_name} cyc/deg')
    ax._legend.set_title('water height\n[mm]')
    new_labels = ['30', '60', '120']
    for t, l in zip(ax._legend.texts, new_labels): t.set_text(l)
    plt.tight_layout()
    plt.show()



# OTHERS

# linear Gain (gegen retinal speed)
    ax = sns.relplot(data=Df, x='stim_ang_vel_int_bins', y='x_lin_gain', hue='water_height', palette='dark', col='stim_ang_sf', kind='line')
    ax.set(xlabel='retinal speed [deg/sec]', ylabel='absolute gain')
    ax.set_titles('spatial frequency = {col_name} cyc/deg')
    plt.tight_layout()
    plt.show()

# lin gain (water height in col)
    ax = sns.relplot(data=Df, x='stim_ang_vel_int', y='x_lin_gain', hue='stim_ang_sf', palette='dark', col='water_height',kind='line')
    ax.set(xlabel='retinal speed [deg/sec]', ylabel='absolute gain')
    ax.set_titles('water height = {col_name} mm')
    ax._legend.set_title('spatial\nfrequency\n[cyc/deg]')
    plt.tight_layout()
    plt.show()
    #und gegen LIN VEL geplottet
    sns.set(font_scale=1.25)
    sns.set_style('ticks')

    ax = sns.relplot(data=Df, x='stim_lin_vel_int_bins', y='x_lin_gain', hue='stim_ang_sf', palette='dark', col='water_height',kind='line')
    ax.set(xlabel='stimulus velocity [mm/sec]', ylabel='absolute gain')
    ax.set_titles('water height = {col_name} mm')
    ax._legend.set_title('spatial\nfrequency\n[cyc/deg]')
    plt.tight_layout()
    plt.show()




    import IPython
    IPython.embed()

    exit()

# min und max Gain rausbekommen:
# Df.x_ang_gain.max()
# passende Wasserhöhe dazu finden (falls in tuning func nicht ersichtlich)
# Df[Df.x_ang_gain.max()].water_height #funktioniert so noch nicht

#todo: individual analysis --> binnen (!), lineare parameter verwenden (wo sinnvoll)

# GAIN TUNING FUNCTION

# linear Gain
    ax = sns.relplot(data=Df, x='stim_ang_vel_int', y='x_lin_gain', hue='water_height', palette='dark', col='stim_ang_sf', kind='line')
    ax.set(xlabel='retinal speed [deg/sec]', ylabel='linear gain')
    ax.set_titles('spatial frequency = {col_name} cyc/deg')
    plt.tight_layout()
    plt.show()



# angular Gain
    ax = sns.relplot(data=Df, x='stim_ang_vel_int', y='x_ang_gain', hue='water_height', palette='dark', col='stim_ang_sf', kind='line')
    ax.set(xlabel='retinal speed [deg/sec]', ylabel='relative gain [mm/deg]')
    ax.set_titles('spatial frequency = {col_name} cyc/deg')
    plt.tight_layout()
    plt.show()


# only constant stim vel
    ax = sns.relplot(data=Df[Df['is_constant_lin_vel']], x='stim_ang_vel_int', y='x_ang_gain_individual', hue='stim_lin_sf',
                     palette='dark', col='water_height', kind='line')
    # ax.set(xlabel='retinal speed [deg/sec]', ylabel='relative gain')
    # ax.set_titles('{col_var} = {col_name} cyc/deg')
    plt.tight_layout()
    plt.show()


# only NOT constant stim vel
    ax = sns.relplot(data=Df[Df['not_constant_lin_vel']], x='stim_ang_vel_int', y='x_ang_gain_individual', hue='stim_lin_sf',
                     palette='dark', col='water_height', kind='line')
    # ax.set(xlabel='retinal speed [deg/sec]', ylabel='relative gain')
    # ax.set_titles('{col_var} = {col_name} cyc/deg')
    plt.tight_layout()
    plt.show()


    import IPython
    IPython.embed()





    '''
    # for water height 60mm
        #pd.cut(Df.stim_lin_vel_int, 40)
    
        Df['stim_lin_vel_int_bins'] = pd.cut(Df.stim_lin_vel_int, 40)
        # Df['stim_lin_vel_int_bins'] = pd.cut(Df.stim_lin_vel_int, 10, right=False, labels=labels)
        # Df['stim_lin_vel_int_bins'] = pd.cut(Df.stim_lin_vel_int, 10)
        # Df.stim_lin_vel_int_bins
        Df['stim_lin_vel_int_bins'] = pd.cut(Df.stim_lin_vel_int, bins=40)
        # Df.stim_lin_vel_int_bins.mid
        #Df['stim_lin_vel_int_bins']=[stim_lin_vel_int_bins_mid.mid for stim_lin_vel_int_bins_mid in stim_lin_vel_int_bins]
        #Df
        stim_lin_vel_int_bins = pd.cut(Df.stim_lin_vel_int, 10)
        Df['stim_lin_vel_int_bins'] = [stim_lin_vel_bins_calc_mid.mid for stim_lin_vel_bins_calc_mid in stim_lin_vel_int_bins]
    
        # Df['stim_lin_vel_int_bins'] = pd.cut(Df.stim_lin_vel_int, 10)
        # Df.stim_lin_vel_int_bins
    '''












    quit()


#ALTES SKRIPT:

    #todo: tuning func getrennt für alle angular und für alle absoluten velos?

    # filter data
    all_abs_velo = Df[(np.isclose(Df.u_lin_velocity, 28) | np.isclose(Df.u_lin_velocity, 143) | np.isclose(Df.u_lin_velocity, 286) | np.isclose(Df.u_lin_velocity, -28) | np.isclose(Df.u_lin_velocity, -143) | np.isclose(Df.u_lin_velocity, -286))]
    all_an_velo = Df[np.isclose(Df.u_lin_velocity, 13.3) | np.isclose(Df.u_lin_velocity, 26.6) | np.isclose(Df.u_lin_velocity, 53.2) | np.isclose(Df.u_lin_velocity, -13.3) | np.isclose(Df.u_lin_velocity, -26.6) | np.isclose(Df.u_lin_velocity, -53.2) | ((np.isclose(Df.water_height, 30)) & (np.isclose(Df.u_lin_velocity, 28))) | np.isclose(Df.u_lin_velocity, 56) | np.isclose(Df.u_lin_velocity, 111.9) | ((np.isclose(Df.water_height, 30)) & (np.isclose(Df.u_lin_velocity, -28))) | np.isclose(Df.u_lin_velocity, -56) | np.isclose(Df.u_lin_velocity, -111.9) | ((np.isclose(Df.water_height, 120)) & (np.isclose(Df.u_lin_velocity, 286))) | ((np.isclose(Df.water_height, 60)) & (np.isclose(Df.u_lin_velocity, 143))) | np.isclose(Df.u_lin_velocity, 71.5) | ((np.isclose(Df.water_height, 120)) & (np.isclose(Df.u_lin_velocity, -286))) | ((np.isclose(Df.water_height, 60)) & (np.isclose(Df.u_lin_velocity, -143))) | np.isclose(Df.u_lin_velocity, -71.5)]


    # ABSOLUTE VELOCITES
    # absolute Gain
    ax = sns.relplot(data=all_abs_velo, x='retinal_speed', y='absolute_gain', hue='water_height', palette='dark', col='spat_frequency', kind='line')
    ax.set(xlabel='retinal speed [deg/sec]', ylabel='absolute gain')
    ax.set_titles('Absolute Velocities; {col_var} = {col_name} cyc/deg')
    plt.tight_layout()
    plt.show()

    # relative Gain
    ax = sns.relplot(data=all_abs_velo, x='retinal_speed', y='angular_gain', hue='water_height', palette='dark', col='spat_frequency', kind='line')
    ax.set(xlabel='retinal speed [deg/sec]', ylabel='relative gain')
    ax.set_titles('Absolute Velocities; {col_var} = {col_name} cyc/deg')
    plt.tight_layout()
    plt.show()


    # ANGULAR VELOCITIES
    # absolute Gain
    ax = sns.relplot(data=all_an_velo, x='retinal_speed', y='absolute_gain', hue='water_height', palette='dark', col='spat_frequency', kind='line')
    ax.set(xlabel='retinal speed [deg/sec]', ylabel='absolute gain')
    ax.set_titles('Angular Velocities; {col_var} = {col_name} cyc/deg')
    plt.tight_layout()
    plt.show()

    # relative Gain
    ax = sns.relplot(data=all_an_velo, x='retinal_speed', y='angular_gain', hue='water_height', palette='dark', col='spat_frequency', kind='line')
    ax.set(xlabel='retinal speed [deg/sec]', ylabel='relative gain')
    ax.set_titles('Angular Velocities; {col_var} = {col_name} cyc/deg')
    plt.tight_layout()
    plt.show()


