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

    sns.set(font_scale=1.3)
    sns.set_style('ticks')

    import IPython
    IPython.embed()

stim_ang_vel_individual_int_bins = pd.cut(Df.stim_ang_vel_individual_int, 40)
Df['stim_ang_vel_individual_int_bins'] = [stim_ang_vel_individual_int_bins_calc_mid.mid for stim_ang_vel_individual_int_bins_calc_mid in stim_ang_vel_individual_int_bins]

ax = sns.relplot(data=Df, x='stim_ang_vel_individual_int_bins', y='x_ang_gain_individual', hue='water_height', palette='dark', col='stim_ang_sf', kind='line')
#ax.set(xlabel='stimulus velocity [mm/sec]', ylabel='absolute gain')
#ax.set_titles('spatial frequency = {col_name} cyc/deg')
plt.tight_layout()
plt.show()

stim_ang_vel_individual
stim_ang_tf
stim_ang_vel_individual_int
stim_ang_sf_individual
stim_ang_sp_individual

ax = sns.relplot(data=Df, x='stim_ang_tf', y='stim_ang_vel_individual_int', hue='water_height', palette='dark', col='stim_ang_sf', kind='line')
#ax.set(xlabel='stimulus velocity [mm/sec]', ylabel='absolute gain')
#ax.set_titles('spatial frequency = {col_name} cyc/deg')
plt.tight_layout()
plt.show()




