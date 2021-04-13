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


# filter data:

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




# GAIN AT DIFFERENT WATER_HEIGHTS

# Angular Velocity: 25 deg/sec

# absolute Gain,
ax = sns.relplot(data=an_velo25, x='water_height', y='absolute_gain', hue='spat_frequency', palette='dark', col='retinal_speed', kind='line')
ax.set(xlabel='water_height [mm]', ylabel='absolute gain', title='angular velocity 25 deg/sec')
plt.tight_layout()
plt.show()

# relative Gain,
ax = sns.relplot(data=an_velo25, x='water_height', y='angular_gain', hue='spat_frequency', palette='dark', col='retinal_speed', kind='line')
ax.set(xlabel='water_height [mm]', ylabel='angular gain', title='angular velocity 25 deg/sec')
#plt.tight_layout()
plt.show()


# Angular Velocity: 50 deg/sec








