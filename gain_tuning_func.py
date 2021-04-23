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


# load Data
base_folder = './data/'
Df = pd.read_hdf('Summary_final.h5', 'all')


# GAIN TUNING FUNCTION

# absolute Gain
ax = sns.relplot(data=Df, x='retinal_speed', y='absolute_gain', hue='water_height', palette='dark', col='spat_frequency', kind='line')
ax.set(xlabel='retinal speed [deg/sec]', ylabel='absolute gain')
ax.set_titles('{col_var} = {col_name} cyc/deg')
plt.tight_layout()
plt.show()

# relative Gain
ax = sns.relplot(data=Df, x='retinal_speed', y='angular_gain', hue='water_height', palette='dark', col='spat_frequency', kind='line')
ax.set(xlabel='retinal speed [deg/sec]', ylabel='relative gain')
ax.set_titles('{col_var} = {col_name} cyc/deg')
plt.tight_layout()
plt.show()



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

