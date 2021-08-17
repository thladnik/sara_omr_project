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

classify_fish.probability_threshold = 0.75
classify_fish.annot_path = 'annotations'

Df = filter_fish(load_summary('//172.25.250.112/arrenberg_data/shared/Sara_Widera/Ausgewertet/Summary.h5'))
# Df = pd.read_hdf(os.path.join(base_folder, 'Summary.h5'), 'all')

# Calculate real positions and add water height
positions = Df.apply(get_real_pos, axis=1)
Df['x_real'] = [p[0] for p in positions]
Df['y_real'] = [p[1] for p in positions]
Df['water_height'] = Df.apply(get_waterlevel, axis=1)

# Group by particle, damit darunter nicht unbekannt
grps = Df.groupby(['folder', 'phase_name', 'particle'])

# import IPython
# IPython.embed()
# Df2 = Df[0:5000]
# Df2.to_excel('SubparticleTest.xlsx')

# Split particles at side lines
Df = grps.apply(make_subparticles).reset_index(drop=True)

# Df2 = Df[0:5000]
# Df2.to_excel('SubparticleTest.xlsx')

# Group by subparticle
grps = Df.groupby(['folder', 'phase_name', 'particle', 'subparticle'])

# Create final DataFrame
Df = pd.DataFrame()
Df = Df.reset_index()
Df['x_vel'] = grps.apply(calc_x_velocity)
Df['y_vel'] = grps.apply(calc_y_velocity)
Df['folder'] = grps.apply(fun_get_folder)
Df['phase_name'] = grps.apply(fun_get_phase_name)
Df['particle'] = grps.apply(fun_get_particle)
Df['u_lin_velocity'] = grps.apply(fun_get_u_lin_velocity)
Df['u_spat_period'] = grps.apply(fun_get_u_spat_period)
Df['frame_count'] = grps.apply(get_row_count)
Df['absolute_gain'] = Df.apply(calc_absolute_gain, axis=1)
Df['x_real_mean'] = grps.apply(get_x_real_mean)
Df['y_real_mean'] = grps.apply(get_y_real_mean)
Df['water_height'] = Df.apply(get_waterlevel, axis=1)
Df['y_world'] = Df.apply(calc_real_world_y, axis=1)
Df['x_world'] = Df.apply(calc_real_world_x, axis=1)
Df['retinal_speed'] = Df.apply(get_retinal_speed, axis=1)
Df['spat_frequency'] = Df.apply(get_spat_freq, axis=1)
Df['angular_gain'] = Df.apply(calc_angular_gain, axis=1)
Df['retinal_speed_magnitude'] = Df.apply(get_retinal_speed_magnitude, axis=1)
Df['u_lin_velocity_magnitude'] = Df.apply(get_u_lin_velocity_magnitude, axis=1)
Df['x_vel_magnitude'] = Df.apply(get_x_vel_magnitude, axis=1)
Df['temp_freq'] = Df.apply(calc_temp_freq, axis=1)
Df['temp_freq_magnitude'] = Df.apply(get_temp_freq_magnitude, axis=1)
Df['actual_retinal_speed'] = Df.apply(calc_actual_retinal_speed, axis=1)
# Df.to_csv('Df.csv')

# Filter out all fish at the arrival side
Df = Df[(Df['u_lin_velocity'] < 0.) & (Df.x_real_mean < 45.) | (Df['u_lin_velocity'] > 0.) & (Df.x_real_mean > -45.)]

# Filter out all blank phases
Df = Df[np.isfinite(Df.u_lin_velocity)]
import IPython
IPython.embed()
# ONLY coralling stimulus phases
Df = Df[(np.isclose(Df.u_lin_velocity, 60.)) | (np.isclose(Df.u_lin_velocity, 75.)) | (np.isclose(Df.u_lin_velocity, 100.))]

# save
Df.to_hdf('coralling_summary.h5', 'all')
Df.to_excel('coralling_summary.xlsx')

Df = pd.read_hdf('coralling_summary.h5', 'all')
