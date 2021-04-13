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


# load Data
base_folder = './data/'
Df = pd.read_hdf('Summary_final.h5', 'all')


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


#4 FISH VELOCITY AT DIFFERENT WATER_HEIGHTS (Hypothesis Matlab)

# spat_freq = 0.02, all angular velocities

# ANGULAR Velocities (all) gepoolt & spat freq 0.02
ax = sns.relplot(data=an_velo_freq2, x='water_height', y='x_vel_magnitude', hue='retinal_speed', palette='dark', col='retinal_speed_magnitude', kind='line')
ax.set(xlabel='water height [mm]', ylabel='magnitude of swimming velocity [mm/s]')
ax.set_titles('magnitude of retinal speed = {col_name} deg/sec')
plt.tight_layout()
plt.subplots_adjust(right=0.92)
plt.show()

# ANGULAR Velocity 25 deg/sec & spat freq 0.02
ax = sns.relplot(data=an_velo25_freq2, x='water_height', y='x_vel', hue='spat_frequency', palette='dark', col='retinal_speed', kind='line')
ax.set(xlabel='water height [mm]', ylabel='swimming velocity [mm/s]')
ax.set_titles('retinal speed = {col_name} cyc/deg')
plt.tight_layout()
plt.show()

# spat_freq = 0.02, all ABSOLUTE velocities (sind zu viele verschiedene für dieselbe darstellung)




#todo: y_pos berechnung aus analyse.df:
        # def calc_real_world_y(df):
        #     correction = 0.5 * dict_dimensions[df.folder][1]
        #     y_new = df.y_real_mean + correction
        #     return y_new
#todo vielleicht etwas weniger als 0.5

# Y POSITION (WORLD)

# Df: Schwimmhöhe und retinal speed
ax = sns.relplot(data=Df, x='retinal_speed', y='y_world', hue='water_height', palette='dark', col='spat_frequency', kind='line')
ax.set(xlabel='retinal speed [°/s]', ylabel='swimming height [mm]')
ax.set_titles('spatial frequency = {col_name} cyc/deg')
plt.tight_layout()
plt.show()

#Df: Schwimmhöhe und retinal speed GEPOOLT
ax = sns.relplot(data=Df, x='retinal_speed_magnitude', y='y_world', hue='water_height', palette='dark', col='spat_frequency', kind='line')
ax.set(xlabel='magnitude of retinal speed [deg/sec]', ylabel='swimming height [mm]')
ax.set_titles('spatial frequency = {col_name} cyc/deg')
plt.tight_layout()
plt.show()




#ax = sns.relplot(data=an_velo25_50, x='temp_freq_magnitude', y='x_vel', hue='retinal_speed_magnitude', palette='dark', col='water_height', kind='line')




