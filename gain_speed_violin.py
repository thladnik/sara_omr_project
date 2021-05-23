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
import scipy.io as sciopython
from analyse_df import *

if __name__ == '__main__':

# load Data
    base_folder = './data/'
    Df = pd.read_hdf('Summary_final.h5', 'by_subparticle')


# plot gain at different stimulus speeds

# GAIN SPEED
# all velocities at once

# linear Gain
    ax = sns.relplot(data=Df, x='stim_ang_vel_int_abs', y='x_lin_gain', hue='water_height', palette='dark', col='stim_ang_sf', kind='line')
    #ax.set(xlabel='retinal speed [deg/sec]', ylabel='absolute gain')
    #ax.set_titles('{col_var} = {col_name} bzw spatial frequency [cyc/deg]')
    plt.tight_layout()
    plt.show()


# angular Gain
    ax = sns.relplot(data=Df, x='stim_ang_vel_int_abs', y='x_ang_gain', hue='water_height', palette='dark', col='stim_ang_sf', kind='line')
    #ax.set(xlabel='retinal speed [deg/sec]', ylabel='relative gain')
    #ax.set_titles('{col_var} = {col_name} bzw spatial frequency [cyc/deg]')
    plt.tight_layout()
    plt.show()




#todo: komische negative gains beim violin plot verstehen und ggf wegmachen

# VIOLIN plot: ANGULAR Gain
    ax = sns.violinplot(data=Df, x='stim_ang_vel_int_abs', y='x_ang_gain', hue='water_height')
    #ax.set(xlabel='magnitude of retinal speed [deg/sec]', ylabel='angular gain')
    plt.show()

# with constant stim vel
    ax = sns.violinplot(data=Df[Df['is_constant_lin_vel']], x='stim_ang_vel_int_abs', y='x_ang_gain', hue='water_height')
    #ax.set(xlabel='magnitude of retinal speed [deg/sec]', ylabel='angular gain')
    plt.show()

# with not constant stim vel
    ax = sns.violinplot(data=Df[Df['not_constant_lin_vel']], x='stim_ang_vel_int_abs', y='x_ang_gain', hue='water_height')
    #ax.set(xlabel='magnitude of retinal speed [deg/sec]', ylabel='angular gain')
    plt.show()


# violin plot: LINEAR Gain
    ax = sns.violinplot(data=Df[Df['not_constant_lin_vel']], x='stim_ang_vel_int_abs', y='x_lin_gain', hue='water_height')
    #ax.set(xlabel='magnitude of retinal speed [deg/sec]', ylabel='linear gain')
    plt.show()




quit()



# ALTES SKRIPT, falls ich mal iwas nachschauen will und nicht auf github suchen will
# filter data

all_abs_velo = Df[(np.isclose(Df.u_lin_velocity, 28) | np.isclose(Df.u_lin_velocity, 143) | np.isclose(Df.u_lin_velocity, 286) | np.isclose(Df.u_lin_velocity, -28) | np.isclose(Df.u_lin_velocity, -143) | np.isclose(Df.u_lin_velocity, -286))]

#anvel25 = Df[np.isclose(Df.retinal_speed, 25, atol=0.3, rtol=0.3)]

an_velo25 = Df[np.isclose(Df.u_lin_velocity, 13.3) | np.isclose(Df.u_lin_velocity, 26.6) | np.isclose(Df.u_lin_velocity, 53.2) | np.isclose(Df.u_lin_velocity, -13.3) | np.isclose(Df.u_lin_velocity, -26.6) | np.isclose(Df.u_lin_velocity, -53.2)]
an_velo50 = Df[((np.isclose(Df.water_height, 30)) & (np.isclose(Df.u_lin_velocity, 28))) | np.isclose(Df.u_lin_velocity, 56) | np.isclose(Df.u_lin_velocity, 111.9) | ((np.isclose(Df.water_height, 30)) & (np.isclose(Df.u_lin_velocity, -28))) | np.isclose(Df.u_lin_velocity, -56) | np.isclose(Df.u_lin_velocity, -111.9)]
an_velo100 = Df[((np.isclose(Df.water_height, 120)) & (np.isclose(Df.u_lin_velocity, 286))) | ((np.isclose(Df.water_height, 60)) & (np.isclose(Df.u_lin_velocity, 143))) | np.isclose(Df.u_lin_velocity, 71.5) | ((np.isclose(Df.water_height, 120)) & (np.isclose(Df.u_lin_velocity, -286))) | ((np.isclose(Df.water_height, 60)) & (np.isclose(Df.u_lin_velocity, -143))) | np.isclose(Df.u_lin_velocity, -71.5)]
all_an_velo = Df[np.isclose(Df.u_lin_velocity, 13.3) | np.isclose(Df.u_lin_velocity, 26.6) | np.isclose(Df.u_lin_velocity, 53.2) | np.isclose(Df.u_lin_velocity, -13.3) | np.isclose(Df.u_lin_velocity, -26.6) | np.isclose(Df.u_lin_velocity, -53.2) | ((np.isclose(Df.water_height, 30)) & (np.isclose(Df.u_lin_velocity, 28))) | np.isclose(Df.u_lin_velocity, 56) | np.isclose(Df.u_lin_velocity, 111.9) | ((np.isclose(Df.water_height, 30)) & (np.isclose(Df.u_lin_velocity, -28))) | np.isclose(Df.u_lin_velocity, -56) | np.isclose(Df.u_lin_velocity, -111.9) | ((np.isclose(Df.water_height, 120)) & (np.isclose(Df.u_lin_velocity, 286))) | ((np.isclose(Df.water_height, 60)) & (np.isclose(Df.u_lin_velocity, 143))) | np.isclose(Df.u_lin_velocity, 71.5) | ((np.isclose(Df.water_height, 120)) & (np.isclose(Df.u_lin_velocity, -286))) | ((np.isclose(Df.water_height, 60)) & (np.isclose(Df.u_lin_velocity, -143))) | np.isclose(Df.u_lin_velocity, -71.5)]

an_velo25_50 = Df[((np.isclose(Df.water_height, 30)) & (np.isclose(Df.u_lin_velocity, 28))) | np.isclose(Df.u_lin_velocity, 56) | np.isclose(Df.u_lin_velocity, 111.9) | ((np.isclose(Df.water_height, 30)) & (np.isclose(Df.u_lin_velocity, -28))) | np.isclose(Df.u_lin_velocity, -56) | np.isclose(Df.u_lin_velocity, -111.9) | np.isclose(Df.u_lin_velocity, 13.3) | np.isclose(Df.u_lin_velocity, 26.6) | np.isclose(Df.u_lin_velocity, 53.2) | np.isclose(Df.u_lin_velocity, -13.3) | np.isclose(Df.u_lin_velocity, -26.6) | np.isclose(Df.u_lin_velocity, -53.2)]

freq4 = Df[np.isclose(Df.spat_frequency, 0.04)]
freq2 = Df[np.isclose(Df.spat_frequency, 0.02)]
freq1 = Df[np.isclose(Df.spat_frequency, 0.01)]

an_velo_freq2 = all_an_velo[np.isclose(all_an_velo.spat_frequency, 0.02)]
abs_velo_freq2 = all_abs_velo[np.isclose(all_abs_velo.spat_frequency, 0.02)]

an_velo25_freq2 = an_velo25[np.isclose(an_velo25.spat_frequency, 0.02)]



import IPython
IPython.embed()


# irgendwie die water height dazu, damit wirklich nur die angular velocties. also Liste in einer liste oder so?
velocities_for_an_vel = [ [30, 13.3], [30, 28], [30, 71.51],
                          [60, 26.6], [60, 56], [60, 143],
                          [120, 53.2], [120, 111.9], [1120, 286]]

velocities_for_an_vel = [13.3, 28, 71.51, 26.6, 56, 143, 53.2, 111.9, 286]

Df_an_vel = Df[np.logical_or([np.isclose(Df.u_lin_velocity, v) for v in velocities_for_an_vel])]
#ValueError: invalid number of arguments









'''
# GAIN AT DIFFERENT STIMULUS SPEEDS
#(overview)
ax = sns.relplot(data=Df, x='retinal_speed', y='absolute_gain', hue='water_height', palette='dark', col='spat_frequency')
ax.set(xlabel='(all angular velocities) retinal speed [deg/sec]', ylabel='absolute gain')
ax.set_titles('{col_var} = {col_name} cyc/deg')
plt.tight_layout()
plt.show()


# todo hier sind die VIOLIN PLOTS: Gain ist komisch negativ
# violin plot: absolute Gain, angular velocities, freq 0.02
ax = sns.violinplot(data=an_velo_freq2, x='retinal_speed_magnitude', y='absolute_gain', hue='water_height')
ax.set(xlabel='magnitude of retinal speed [deg/sec]', ylabel='absolute gain')
ax.set_titles('Angular Velocities; {col_var} = {col_name} cyc/deg')
plt.show()

# violin plot: absolute Gain, absolute velocities, freq 0.02
ax = sns.violinplot(data=abs_velo_freq2, x='retinal_speed_magnitude', y='absolute_gain', hue='water_height')
ax.set(xlabel='magnitude of retinal speed [deg/sec]', ylabel='absolute gain')
ax.set_titles('Absolute Velocities; {col_var} = {col_name} cyc/deg')
plt.show()

# violin plot: relative Gain, angular velocities, freq 0.02
ax = sns.violinplot(data=an_velo_freq2, x='retinal_speed_magnitude', y='angular_gain', hue='water_height')
ax.set(xlabel='magnitude of retinal speed [deg/sec]', ylabel='relative gain')
ax.set_titles('Angular Velocities; {col_var} = {col_name} cyc/deg')
plt.show()

# violin plot: relative Gain, absolute velocities, freq 0.02
ax = sns.violinplot(data=abs_velo_freq2, x='retinal_speed_magnitude', y='angular_gain', hue='water_height')
ax.set(xlabel='magnitude of retinal speed [deg/sec]', ylabel='relative gain')
ax.set_titles('Absolute Velocities; {col_var} = {col_name} cyc/deg')
plt.show()


#todo hier sind 4x Gain & SPEED (relplot)

# absolute Gain, angular velocities gepoolt (spat_freq = 0.02)
ax = sns.relplot(data=an_velo_freq2, x='retinal_speed_magnitude', y='absolute_gain', hue='water_height', palette='dark', col='spat_frequency', kind='line')
ax.set(xlabel='magnitude of retinal speed [deg/sec]', ylabel='absolute gain')
ax.set_titles('Angular Velocities; {col_var} = {col_name} cyc/deg')
plt.tight_layout()
plt.subplots_adjust(right=0.85)
plt.show()

# relative Gain, angular velocities gepoolt (spat_freq = 0.02)
ax = sns.relplot(data=an_velo_freq2, x='retinal_speed_magnitude', y='angular_gain', hue='water_height', palette='dark', col='spat_frequency', kind='line')
ax.set(xlabel='magnitude of retinal speed [deg/sec]', ylabel='relative gain')
ax.set_titles('Angular Velocities; {col_var} = {col_name} cyc/deg')
plt.tight_layout()
plt.subplots_adjust(right=0.85)
plt.show()

# absolute Gain, absolute velocities gepoolt (spat_freq = 0.02)
ax = sns.relplot(data=abs_velo_freq2, x='retinal_speed_magnitude', y='absolute_gain', hue='water_height', palette='dark', col='spat_frequency', kind='line')
ax.set(xlabel='magnitude of retinal speed [deg/sec]', ylabel='absolute gain')
ax.set_titles('Absolute Velocities; {col_var} = {col_name} cyc/deg')
plt.tight_layout()
plt.subplots_adjust(right=0.85)
plt.show()

# relative Gain, absolute velocities gepoolt (spat_freq = 0.02)
ax = sns.relplot(data=abs_velo_freq2, x='retinal_speed_magnitude', y='angular_gain', hue='water_height', palette='dark', col='spat_frequency', kind='line')
ax.set(xlabel='magnitude of retinal speed [deg/sec]', ylabel='relative gain')
ax.set_titles('Absolute Velocities; {col_var} = {col_name} cyc/deg')
plt.tight_layout()
plt.subplots_adjust(right=0.85)
plt.show()

'''



