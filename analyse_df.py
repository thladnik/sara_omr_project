import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import classify_fish
from classify_fish import filter_fish, load_summary
import seaborn as sns
import sys


dict_waterlevel = {'rec_2020-12-17-13-53-27_3cm': 30, 'rec_2020-12-18-09-46-26_6cm': 60,
                   'rec_2020-12-17-14-38-13_6cm': 60, 'rec_2020-12-18-10-31-28_3cm': 30,
                   'rec_2020-12-18-11-38-34_12cm': 120, 'rec_2020-12-18-13-33-52_6cm': 60,
                   'rec_2020-12-18-14-31-07_3cm': 30,
                   'rec_2021-02-03-12-30-46_6cm': 60, 'rec_2021-02-03-15-40-19_12cm': 120,
                   'rec_2021-02-03-17-05-28_3cm': 30}


dict_dimensions = {'rec_2020-12-17-13-53-27_3cm': (121, 33), 'rec_2020-12-18-09-46-26_6cm': (115, 58),
                   'rec_2020-12-17-14-38-13_6cm': (121.5, 60), 'rec_2020-12-18-10-31-28_3cm': (122, 33),
                   'rec_2020-12-18-11-38-34_12cm': (120, 120), 'rec_2020-12-18-13-33-52_6cm': (118, 60),
                   'rec_2020-12-18-14-31-07_3cm': (121, 30),
                   'rec_2021-02-03-12-30-46_6cm': (120, 61), 'rec_2021-02-03-15-40-19_12cm': (120, 120),
                   'rec_2021-02-03-17-05-28_3cm': (120, 31)}


#base_folder = '//172.25.250.112/arrenberg_data/shared/Sara_Widera/Ausgewertet'
base_folder = './data/'

def plot_colorpoint(ax, x, y, z, cmap, valmap, fontsize = 9):
    if sum(z < valmap) > len(valmap) / 2:
        fontcolor = 'w'
    else:
        fontcolor = 'k'
    ax.loglog(x, y, marker='o', markersize=17, color=cmap[np.argmin(np.abs(valmap - z)), :])
    ax.text(x, y, str(round(z, 1)), fontsize=fontsize, color=fontcolor,  ha='center', va='center')


def get_waterlevel(series):
    path = series.folder.split(os.sep)
    wl = dict_waterlevel.get(path[-1])
    return wl


def calc_absolute_gain(series):
    fish_vel = series.x_vel
    stim_vel = series.u_lin_velocity
    gain = fish_vel / stim_vel
    return gain


def calc_angular_gain(series):
    fish_vel = series.x_vel
    stim_vel = series.retinal_speed
    gain = fish_vel / stim_vel
    return gain


def make_subparticles(df):
    bool_vec = ((df.x_real > -40) & (df.x_real < 40)).values
    borders = np.where(np.diff(bool_vec))[0]  #gibt array mit den positionen bei einem übergang, also von true (1) zu false (0) bei filter_conditon und umgekehrt mithilfe der differenz der benachbarten einträge
    outer = [0, *borders+1, bool_vec.shape[0]+1]
    df = df.reset_index()
    for id, (i, j) in enumerate(zip(outer[:-1], outer[1:])):
        df.loc[i:j, 'subparticle'] = id
    return df


def fill_in_IDs(df):
    return pd.DataFrame({'folder': df.folder.unique(),
                         'phase_name': df.phase_name.unique(),
                         'particle': df.particle.unique(),
                         'subparticle': df.subparticle.unique()})


def get_angular_gain_mean(df):
    return df.angular_gain.mean()
def get_absolute_gain_mean(df):
    return df.absolute_gain.mean()
def get_spat_frequency_mean(df):
    return df.spat_frequency.mean()
def get_temp_freq_mean(df):
    return df.temp_freq.mean()
def get_temp_freq_magnitude_mean(df):
    return df.temp_freq_magnitude.mean()
def get_retinal_speed_mean(df):
    return df.retinal_speed.mean()
def get_retinal_speed_magnitude_mean(df):
    return df.retinal_speed_magnitude.mean()
def get_waterheight_mean(df):
    return df.water_height.mean()
def get_u_lin_velocity(df):
    return df.u_lin_velocity.unique()[0]

def conv_vel(v):
    if len(v) >= 20:
        return np.convolve(v, np.ones(20) / 20, 'same')
    else:
        return np.convolve(v, np.ones(len(v)) / len(v), 'same')

if __name__ == '__main__':

    # Df = filter_fish(load_summary('data/Summary.h5'))
    # sub = Df[Df.folder == Df.folder.values[0]]
    # g = sub.groupby(['folder', 'phase_id', 'particle'])
    # gsub = pd.DataFrame()
    # gsub['x_vel'] = g.apply(lambda df: df.x.diff() / df.time.diff())
    # gsub['time'] = g.apply(lambda df: df.time - df.time.values[0])
    # gsub['x_pos'] = g.apply(lambda df: df.x)
    # gsub = gsub.reset_index()
    # sns.relplot(data=gsub[gsub.phase_id < 20], x='time', y='x_vel', hue='particle', col='phase_id', kind='line', col_wrap=5, height=3, aspect=1)
    # # Convolve velocity:
    # gsub["x_vel_conv"] = (gsub.groupby(['folder', 'phase_id', 'particle'])['x_vel'].transform(conv_vel))
    # sns.relplot(data=gsub[gsub.phase_id < 20], x='time', y='x_vel_conv', hue='particle', col='phase_id', kind='line', col_wrap=5, height=3, aspect=1)


    if 'recalc' in sys.argv or not(os.path.exists('Summary_final.h5')):
        print('Recalculating final DataFrame')

        classify_fish.probability_threshold = 0.75
        classify_fish.annot_path = 'annotations'

        Df = filter_fish(load_summary('data/Summary.h5'))

        # Calculate real positions and add water heights
        print('Calcule approx. real position')
        p = np.asarray([Df.x.values, Df.y.values, np.ones(Df.shape[0]) * 30.]) * 10.
        d = p / np.linalg.norm(p, axis=0)
        p_new = p + d * 25. * np.dot(np.array([0., 0., 1.]), d)
        Df['x_real'] = p_new[0, :]
        Df['y_real'] = p_new[1, :]

        # Add folder based information
        for folder in Df.folder.unique():
            alter = Df.folder == folder
            Df.loc[alter, 'dim_x'] = dict_dimensions[folder][0]
            Df.loc[alter, 'dim_y'] = dict_dimensions[folder][1]
            Df.loc[alter,'water_height'] = dict_waterlevel.get(folder)

        # Group by particle and split particles at side lines
        print('Split into subparticles')
        grps = Df.groupby(['folder', 'phase_name', 'particle'])
        Df['subparticle'] = 0
        Df = grps.apply(make_subparticles).reset_index(drop=True)
        Df['subparticle'] = Df['subparticle'].astype(int)

        print('Save full DataFrame to file')
        print('HDF5')
        Df.to_hdf('Summary_final.h5', 'full')
        # print('XLSX')
        # for folder in Df.folder.unique():
        #     print(f'Sheet {folder}')
        #     Df[Df.folder == folder].to_excel('Summary_final.xlsx', sheet_name=folder)

        # Group again, this time with subparticles
        print('Group by subparticles')
        grps = Df.groupby(['folder', 'phase_name', 'particle', 'subparticle'])

        # Create final DataFrame
        Df = pd.DataFrame()

        # Fill Df again (from groups)
        # Basic info
        print('Basic info')
        Df['folder'] = grps.apply(lambda df: df.folder.unique()[0])
        Df['phase_name'] = grps.apply(lambda df: df.phase_name.unique()[0])
        Df['phase_id'] = grps.apply(lambda df: df.phase_id.unique()[0])
        Df['particle'] = grps.apply(lambda df: df.particle.unique()[0])
        Df['subparticle'] = grps.apply(lambda df: df.subparticle.unique()[0])
        Df['water_height'] = grps.apply(lambda df: df.water_height.unique()[0])
        Df['dim_x'] = grps.apply(lambda df: df.dim_x.unique()[0])
        Df['dim_y'] = grps.apply(lambda df: df.dim_y.unique()[0])
        Df['frame_count'] = grps.apply(lambda df: df.shape[0])

        # Filter out all particles with less than 2 frames
        Df = Df[Df['frame_count'] > 1]

        # Filter out all coralling/blank stimulus phases
        print('Filter test phases')
        Df = Df[(Df['phase_id'] + 1) % 3 == 0].copy()

        print('Get fish position data')
        # Get projected coordinates
        Df['x_mean'] = grps.apply(lambda df: df.x_real.mean())
        Df['y_mean'] = grps.apply(lambda df: df.y_real.mean())
        # Calculate x/y_world coordinates in [mm] (positive range) and rectify negative/zero values
        Df['x_world_mean'] = Df['x_mean'] + Df['dim_x']/2.
        Df.loc[Df['x_world_mean'] <= 0., 'x_world_mean'] = 10e-10
        Df['y_world_mean'] = Df['y_mean'] + Df['dim_y']/2.
        Df.loc[Df['y_world_mean'] <= 0., 'y_world_mean'] = 10e-10

        # Stimulus
        print('Get stimulus parameters')
        # Linear parameters
        Df['stim_lin_vel'] = grps.apply(lambda df: df.u_lin_velocity.unique()[0])
        Df['stim_lin_sp'] = grps.apply(lambda df: df.u_spat_period.unique()[0])
        Df['stim_lin_sf'] = 1. / Df['stim_lin_sp']
        Df['stim_lin_tf'] = Df['stim_lin_vel'] / Df['stim_lin_sp']
        Df['stim_lin_vel_int'] = Df['stim_lin_vel'].values.astype(np.int32)
        Df['stim_lin_vel_int_abs'] = Df['stim_lin_vel_int'].abs()

        # Fish-perspective based (retinal) parameters
        # Calculate retinal (angular) spatial frequency from stim_lin_sp [mm] and water_height [mm]
        # under assumption that fish swim ~10mm under the water surface
        # (original assumption, but doesn't work with chosen parameters for experiment, so we'll consider fish to be swimming around water level)
        # 1 / (360 * 2 * atan(sp[mm] / (2 * (wh[mm] - 10.0))) / (2 * pi))
        # Df['stim_ang_sp'] = Df.apply(lambda s: 360 * 2 * np.arctan(s['stim_lin_sp'] / (2 * (s['water_height'] - 10.0))) / (2 * np.pi), axis=1)
        Df['stim_ang_sp'] = Df.apply(lambda s: np.round(360 * 2 * np.arctan(s['stim_lin_sp'] / (2 * s['water_height'])) / (2 * np.pi), 0), axis=1)
        Df['stim_ang_sf'] = 1. / Df['stim_ang_sp']
        Df['stim_ang_sp_individual'] = Df.apply(lambda s: 360 * 2 * np.arctan(s['stim_lin_sp'] / (2 * (s['y_world_mean']))) / (2 * np.pi), axis=1)
        Df['stim_ang_sf_individual'] = 1. / Df['stim_ang_sp_individual']
        # Calculate angular velocity
        # 1 / (360 * 2 * atan((vel[mm/s] * 1[s] )/ (2 * wh[mm])) / (2 * pi))
        # This implicitly makes the assumption that the fish is primarily looking directly downwards
        # Df['stim_ang_vel'] = Df.apply(lambda s: 360 * 2 * np.arctan(s['stim_lin_vel'] * 1.0 / (2 * (s['water_height'] - 10.0))) / (2 * np.pi), axis=1)
        Df['stim_ang_vel'] = Df.apply(lambda s: np.round(360 * 2 * np.arctan(s['stim_lin_vel'] * 1.0 / (2 * s['water_height'])) / (2 * np.pi), 0), axis=1)
        # Calculate angular velocity under assumption that mean fish y position is actual fish position
        Df['stim_ang_vel_individual'] = Df.apply(lambda s: 360 * 2 * np.arctan(s['stim_lin_vel'] * 1.0 / (2 * s['y_world_mean'])) / (2 * np.pi), axis=1)

        Df['stim_ang_tf'] = Df['stim_ang_vel'] / Df['stim_ang_sp']

        # Rounded stimulus velocity
        Df['stim_ang_vel_int'] = Df['stim_ang_vel'].values.astype(np.int32)
        Df['stim_ang_vel_int_abs'] = Df['stim_ang_vel_int'].abs()
        Df['stim_ang_vel_individual_int'] = Df['stim_ang_vel_individual'].values.astype(np.int32)
        Df['stim_ang_vel_individual_int_abs'] = Df['stim_ang_vel_individual_int'].abs()

        Df['stim_ang_vel_abs'] = Df['stim_ang_vel'].abs()
        Df['stim_lin_vel_abs'] = Df['stim_lin_vel'].abs()

        print('Set stimulus condition flags')
        Df['is_constant_lin_vel'] = Df.stim_lin_vel_abs.isin([28, 143, 286])
        Df['not_constant_lin_vel'] = ~Df['is_constant_lin_vel']

        # Fish responses
        print('Get fish parameters')
        Df['x_vel_mean'] = grps.apply(lambda df: (df.x_real.diff() / df.time.diff()).mean())
        Df['x_vel_mean_abs'] = Df['x_vel_mean'] * Df['stim_lin_vel'] /  Df['stim_lin_vel'].abs()
        Df['y_vel_mean'] = grps.apply(lambda df: (df.y_real.diff() / df.time.diff()).mean())

        # Calculate gain
        Df['x_ang_gain'] = Df['x_vel_mean'] / Df['stim_ang_vel']
        Df['x_ang_gain_individual'] = Df['x_vel_mean'] / Df['stim_ang_vel_individual']
        Df['x_lin_gain'] = Df['x_vel_mean'] / Df['stim_lin_vel']

        Df['x_vel_abs'] = Df['x_vel_mean'].abs()

        # Filter out all fish at the start side
        Df = Df[((Df['stim_lin_vel'] > 0.) & (Df['x_mean'] < 40.)) | ((Df['stim_lin_vel'] < 0.) & (Df['x_mean'] > -40.))]

        # Save to final summary
        Df.to_hdf('Summary_final.h5', 'by_subparticle')
        Df.to_excel('Summary_final.xlsx', sheet_name='by_subparticle')

    # Load summary grouped by subparticles
    quit()
    Df = pd.read_hdf('Summary_final.h5', 'by_subparticle')

    # grps_heat = Df.groupby(['water_height', 'stim_lin_sf', 'stim_tf'])
    # Df_gain_heat = pd.DataFrame()
    # Df_gain_heat['ang_gain_mean'] = grps_heat['x_ang_gain'].apply(np.mean)
    # Df_gain_heat['ang_gain_std'] = grps_heat['x_ang_gain'].apply(np.std)
    # Df_gain_heat['test'] = Df_gain_heat['ang_gain_mean'] < Df_gain_heat['ang_gain_std']

    # sns.stripplot(data=Df[Df.water_height == 60], x='stim_lin_sf', y='x_lin_gain', hue='stim_ang_vel_int', dodge=True, s=2.)
    # sns.boxplot(data=Df[Df.water_height == 60], x='stim_lin_sf', y='x_lin_gain', hue='stim_ang_vel_int')


    # Data for gain heatmaps
    grps_heat = Df.groupby(['water_height', 'stim_lin_sf', 'stim_tf'])
    Df_gain_heat = pd.DataFrame()
    Df_gain_heat['angular_gain_mean'] = grps_heat.apply(np.mean)
    Df_gain_heat['absolute_gain_mean'] = grps_heat.apply(get_absolute_gain_mean)
    Df_gain_heat['spat_frequency_mean'] = grps_heat.apply(get_spat_frequency_mean)
    Df_gain_heat['temp_freq_mean'] = grps_heat.apply(get_temp_freq_mean)
    Df_gain_heat['temp_freq_magnitude_mean'] = grps_heat.apply(get_temp_freq_magnitude_mean)
    Df_gain_heat['retinal_speed_mean'] = grps_heat.apply(get_retinal_speed_mean)
    Df_gain_heat['retinal_speed_magnitude_mean'] = grps_heat.apply(get_retinal_speed_magnitude_mean)
    Df_gain_heat['water_height_mean'] = grps_heat.apply(get_waterheight_mean)
    Df_gain_heat['u_lin_velocity'] = grps_heat.apply(get_u_lin_velocity)

# FILTERED DATA FOR FIGURES:

    all_abs_velo = Df[(np.isclose(Df.u_lin_velocity, 28) | np.isclose(Df.u_lin_velocity, 143) | np.isclose(Df.u_lin_velocity, 286) | np.isclose(Df.u_lin_velocity, -28) | np.isclose(Df.u_lin_velocity, -143) | np.isclose(Df.u_lin_velocity, -286))]
    Heat_all_abs_velo = Df_gain_heat[(np.isclose(Df_gain_heat.u_lin_velocity, 28) | np.isclose(Df_gain_heat.u_lin_velocity, 143) | np.isclose(Df_gain_heat.u_lin_velocity, 286) | np.isclose(Df_gain_heat.u_lin_velocity, -28) | np.isclose(Df_gain_heat.u_lin_velocity, -143) | np.isclose(Df_gain_heat.u_lin_velocity, -286))]

    an_velo25 = Df[np.isclose(Df.u_lin_velocity, 13.3) | np.isclose(Df.u_lin_velocity, 26.6) | np.isclose(Df.u_lin_velocity, 53.2) | np.isclose(Df.u_lin_velocity, -13.3) | np.isclose(Df.u_lin_velocity, -26.6) | np.isclose(Df.u_lin_velocity, -53.2)]
    an_velo50 = Df[((np.isclose(Df.water_height, 30)) & (np.isclose(Df.u_lin_velocity, 28))) | np.isclose(Df.u_lin_velocity, 56) | np.isclose(Df.u_lin_velocity, 111.9) | ((np.isclose(Df.water_height, 30)) & (np.isclose(Df.u_lin_velocity, -28))) | np.isclose(Df.u_lin_velocity, -56) | np.isclose(Df.u_lin_velocity, -111.9)]
    an_velo100 = Df[((np.isclose(Df.water_height, 120)) & (np.isclose(Df.u_lin_velocity, 286))) | ((np.isclose(Df.water_height, 60)) & (np.isclose(Df.u_lin_velocity, 143))) | np.isclose(Df.u_lin_velocity, 71.5) | ((np.isclose(Df.water_height, 120)) & (np.isclose(Df.u_lin_velocity, -286))) | ((np.isclose(Df.water_height, 60)) & (np.isclose(Df.u_lin_velocity, -143))) | np.isclose(Df.u_lin_velocity, -71.5)]
    all_an_velo = Df[np.isclose(Df.u_lin_velocity, 13.3) | np.isclose(Df.u_lin_velocity, 26.6) | np.isclose(Df.u_lin_velocity, 53.2) | np.isclose(Df.u_lin_velocity, -13.3) | np.isclose(Df.u_lin_velocity, -26.6) | np.isclose(Df.u_lin_velocity, -53.2) | ((np.isclose(Df.water_height, 30)) & (np.isclose(Df.u_lin_velocity, 28))) | np.isclose(Df.u_lin_velocity, 56) | np.isclose(Df.u_lin_velocity, 111.9) | ((np.isclose(Df.water_height, 30)) & (np.isclose(Df.u_lin_velocity, -28))) | np.isclose(Df.u_lin_velocity, -56) | np.isclose(Df.u_lin_velocity, -111.9) | ((np.isclose(Df.water_height, 120)) & (np.isclose(Df.u_lin_velocity, 286))) | ((np.isclose(Df.water_height, 60)) & (np.isclose(Df.u_lin_velocity, 143))) | np.isclose(Df.u_lin_velocity, 71.5) | ((np.isclose(Df.water_height, 120)) & (np.isclose(Df.u_lin_velocity, -286))) | ((np.isclose(Df.water_height, 60)) & (np.isclose(Df.u_lin_velocity, -143))) | np.isclose(Df.u_lin_velocity, -71.5)]
    Heat_all_an_velo = Df_gain_heat[np.isclose(Df_gain_heat.u_lin_velocity, 13.3) | np.isclose(Df_gain_heat.u_lin_velocity, 26.6) | np.isclose(Df_gain_heat.u_lin_velocity, 53.2) | np.isclose(Df_gain_heat.u_lin_velocity, -13.3) | np.isclose(Df_gain_heat.u_lin_velocity, -26.6) | np.isclose(Df_gain_heat.u_lin_velocity, -53.2) | ((np.isclose(Df_gain_heat.water_height_mean, 30)) & (np.isclose(Df_gain_heat.u_lin_velocity, 28))) | np.isclose(Df_gain_heat.u_lin_velocity, 56) | np.isclose(Df_gain_heat.u_lin_velocity, 111.9) | ((np.isclose(Df_gain_heat.water_height_mean, 30)) & (np.isclose(Df_gain_heat.u_lin_velocity, -28))) | np.isclose(Df_gain_heat.u_lin_velocity, -56) | np.isclose(Df_gain_heat.u_lin_velocity, -111.9) | ((np.isclose(Df_gain_heat.water_height_mean, 120)) & (np.isclose(Df_gain_heat.u_lin_velocity, 286))) | ((np.isclose(Df_gain_heat.water_height_mean, 60)) & (np.isclose(Df_gain_heat.u_lin_velocity, 143))) | np.isclose(Df_gain_heat.u_lin_velocity, 71.5) | ((np.isclose(Df_gain_heat.water_height_mean, 120)) & (np.isclose(Df_gain_heat.u_lin_velocity, -286))) | ((np.isclose(Df_gain_heat.water_height_mean, 60)) & (np.isclose(Df_gain_heat.u_lin_velocity, -143))) | np.isclose(Df_gain_heat.u_lin_velocity, -71.5)]

    an_velo25_50 = Df[((np.isclose(Df.water_height, 30)) & (np.isclose(Df.u_lin_velocity, 28))) | np.isclose(Df.u_lin_velocity, 56) | np.isclose(Df.u_lin_velocity, 111.9) | ((np.isclose(Df.water_height, 30)) & (np.isclose(Df.u_lin_velocity, -28))) | np.isclose(Df.u_lin_velocity, -56) | np.isclose(Df.u_lin_velocity, -111.9) | np.isclose(Df.u_lin_velocity, 13.3) | np.isclose(Df.u_lin_velocity, 26.6) | np.isclose(Df.u_lin_velocity, 53.2) | np.isclose(Df.u_lin_velocity, -13.3) | np.isclose(Df.u_lin_velocity, -26.6) | np.isclose(Df.u_lin_velocity, -53.2)]


    freq4 = Df[np.isclose(Df.spat_frequency, 0.04)]
    freq2 = Df[np.isclose(Df.spat_frequency, 0.02)]
    freq1 = Df[np.isclose(Df.spat_frequency, 0.01)]

    an_velo_freq2 = all_an_velo[np.isclose(all_an_velo.spat_frequency, 0.02)]
    abs_velo_freq2 = all_abs_velo[np.isclose(all_abs_velo.spat_frequency, 0.02)]

    an_velo25_freq2 = an_velo25[np.isclose(an_velo25.spat_frequency, 0.02)]

    # wh30 = Df[(np.isclose(Df.water_height, 30))]
    wh30 = Df_gain_heat[(np.isclose(Df_gain_heat.water_height_mean, 30))]
    wh60 = Df_gain_heat[(np.isclose(Df_gain_heat.water_height_mean, 60))]
    wh120 = Df_gain_heat[(np.isclose(Df_gain_heat.water_height_mean, 120))]
    Heat_abs_wh60 = Heat_all_abs_velo[(np.isclose(Heat_all_abs_velo.water_height_mean, 60))]
    Heat_an_wh60 = Heat_all_an_velo[(np.isclose(Heat_all_an_velo.water_height_mean, 60))]

    import IPython
    IPython.embed()

    # Stats = pd.DataFrame(grp_df2, columns = ['folder', 'phase_name', 'particle', 'u_lin_velocity', 'frame_count', 'x_vel', 'x_vel_magnitude', 'absolute_gain', 'angular_gain', 'y_world', 'water_height', 'retinal_speed', 'retinal_speed_magnitude', 'spat_frequency'])
    # grp_df2.to_excel(r'F:\Sara_Widera\statistics.xlsx', index=True, header=True)


# FIGURES FOR THESIS:

#1 GAIN TUNING FUNCTIONS
    # absolute Gain
    ax = sns.relplot(data=Df, x='retinal_speed', y='absolute_gain', hue='water_height', palette='dark', col='spat_frequency', kind='line')
    ax.set(xlabel='retinal speed [deg/sec]', ylabel='absolute gain')
    ax.set_titles('{col_var} = {col_name} cyc/deg')

    # relative Gain
    ax = sns.relplot(data=Df, x='retinal_speed', y='angular_gain', hue='water_height', palette='dark', col='spat_frequency', kind='line')
    ax.set(xlabel='retinal speed [deg/sec]', ylabel='relative gain')
    ax.set_titles('{col_var} = {col_name} cyc/deg')


#tuning func für alle angular und für alle absoluten velos?

#2 GAIN AT DIFFERENT STIMULUS SPEEDS
    #(overview)
    ax = sns.relplot(data=Df, x='retinal_speed', y='absolute_gain', hue='water_height', palette='dark', col='spat_frequency')
    ax.set(xlabel='(all angular velocities) retinal speed [deg/sec]', ylabel='absolute gain')
    ax.set_titles('{col_var} = {col_name} cyc/deg')

    # violin plot: absolute Gain, angular velocities, freq 0.02
    ax = sns.violinplot(data=an_velo_freq2, x='retinal_speed_magnitude', y='absolute_gain', hue='water_height')
    ax.set(xlabel='magnitude of retinal speed [deg/sec]', ylabel='absolute gain', title='all angular velocities at spat freq 0.02 cyc/deg')

    # violin plot: absolute Gain, absolute velocities, freq 0.02
    ax = sns.violinplot(data=abs_velo_freq2, x='retinal_speed_magnitude', y='absolute_gain', hue='water_height')
    ax.set(xlabel='magnitude of retinal speed [deg/sec]', ylabel='absolute gain', title='all absolute velocities at spat freq 0.02 cyc/deg')

    # violin plot: relative Gain, angular velocities, freq 0.02
    ax = sns.violinplot(data=an_velo_freq2, x='retinal_speed_magnitude', y='angular_gain', hue='water_height')
    ax.set(xlabel='magnitude of retinal speed [deg/sec]', ylabel='relative gain', title='all angular velocities at spat freq 0.02 cyc/deg')

    # violin plot: relative Gain, absolute velocities, freq 0.02
    ax = sns.violinplot(data=abs_velo_freq2, x='retinal_speed_magnitude', y='angular_gain', hue='water_height')
    ax.set(xlabel='magnitude of retinal speed [deg/sec]', ylabel='relative gain', title='all absolute velocities at spat freq 0.02 cyc/deg')

    # absolute Gain, angular velocities gepoolt (spat_freq = 0.02)
    ax = sns.relplot(data=an_velo_freq2, x='retinal_speed_magnitude', y='absolute_gain', hue='water_height', palette='dark', col='spat_frequency', kind='line')
    ax.set(xlabel='magnitude of retinal speed [deg/sec]', ylabel='absolute gain') #title='all angular velocities at spatial frequency 0.02 cyc/deg'
    ax.set_titles('{col_var} = {col_name} cyc/deg')


    # relative Gain, angular velocities gepoolt (spat_freq = 0.02)
    ax = sns.relplot(data=an_velo_freq2, x='retinal_speed_magnitude', y='angular_gain', hue='water_height', palette='dark', col='spat_frequency', kind='line')
    ax.set(xlabel='magnitude of retinal speed [deg/sec]', ylabel='relative gain', title='all angular velocities at spatial frequency 0.02 cyc/deg')

    # absolute Gain, absolute velocities gepoolt (spat_freq = 0.02)
    ax = sns.relplot(data=abs_velo_freq2, x='retinal_speed_magnitude', y='absolute_gain', hue='water_height', palette='dark', col='spat_frequency', kind='line')
    ax.set(xlabel='magnitude of retinal speed [deg/sec]', ylabel='absolute gain',  title='all absolute velocities at spatial frequency 0.02 cyc/deg')

    # relative Gain, absolute velocities gepoolt (spat_freq = 0.02)
    ax = sns.relplot(data=abs_velo_freq2, x='retinal_speed_magnitude', y='angular_gain', hue='water_height', palette='dark', col='spat_frequency', kind='line')
    ax.set(xlabel='magnitude of retinal speed [deg/sec]', ylabel='relative gain', title='all absolute velocities at spatial frequency 0.02 cyc/deg')


#3 GAIN AT DIFFERENT WATER_HEIGHTS
    # absolute Gain,
    ax = sns.relplot(data=an_velo25, x='water_height', y='absolute_gain', hue='spat_frequency', palette='dark', col='retinal_speed', kind='line')
    ax.set(xlabel='an_velo25: water_height [mm]', ylabel='absolute gain', title='angular velocity 25 °/s')
    # relative Gain,
    ax = sns.relplot(data=an_velo25, x='water_height', y='angular_gain', hue='spat_frequency', palette='dark', col='retinal_speed', kind='line')
    ax.set(xlabel='an_velo25: water_height [mm]', ylabel='angular gain', title='angular velocity 25 °/s')


#4 FISH VELOCITY AT DIFFERENT WATER_HEIGHTS (Hypothesis)
    # spat_freq = 0.02, all angular velocities
    ax = sns.relplot(data=an_velo25_freq2, x='water_height', y='x_vel', hue='spat_frequency', palette='dark', col='retinal_speed', kind='line')
    ax.set(xlabel='an_velo25_freq2: water_height [mm]', ylabel='swimming velocity [mm/s]', title='angular velocity 25 °/s at spatial frequency 0.02 c/°')

    ax = sns.relplot(data=an_velo_freq2, x='water_height', y='x_vel_magnitude', hue='retinal_speed', palette='dark', col='retinal_speed_magnitude', kind='line')
    ax.set(xlabel='an_velo_freq2: water_height [mm]', ylabel='magnitude of swimming velocity [mm/s]', title='all angular velocities at spatial frequency 0.02 c/°')

    # spat_freq = 0.02, all absolute velocities

    ax = sns.relplot(data=Df, x='retinal_speed_magnitude', y='y_world', hue='water_height', palette='dark', col='spat_frequency', kind='line')
# Fish velocity at different speeds


#5 TUNING FOR SWIMMING SPEED
    ax = sns.relplot(data=Df, x='retinal_speed', y='x_vel', hue='water_height', palette='dark', col='spat_frequency', kind='line')
    ax.set(xlabel='retinal speed [°/s]', ylabel='y-position [mm]')


#6 Y POSITION (WORLD) AT DIFFERENT RETINAL SPEEDS
    ax = sns.relplot(data=Df, x='retinal_speed', y='y_world', hue='water_height', palette='dark', col='spat_frequency', kind='line')
    ax.set(xlabel='retinal speed [°/s]', ylabel='y-position [mm]')
    ax = sns.relplot(data=an_velo25_50, x='temp_freq_magnitude', y='x_vel', hue='retinal_speed_magnitude', palette='dark', col='water_height', kind='line')


#7 GAIN COLOURMAP: SPAT FREQ, TEMP FREQ, RETINAL SPEED
#----> TIMs COLOURMAP :)
    sns.set_theme()
    heatmap_size = (12.5, 11.5)

    cmap_scheme = 'viridis' #turbo
    markersize = 17

    fig = custom_fig('absolute Gain for spat & temp freq at waterheight 30mm', heatmap_size)
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy([],[])
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

''''Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap'
, 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r',
 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu',
'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', '
Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'b
inary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cube
helix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r',
'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r',
 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r',
'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', '
tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 't
wilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'
'''



    # ax = sns.relplot(data=an_velo25_freq2, x='water_height', y='x_vel', hue='spat_frequency', palette='dark', col='retinal_speed', kind='line')
    # ax.set(xlabel='an_velo25_midper2: water_height in mm', ylabel='swimming velocity')
    #
    #
    # for folder in grp_df2.folder.unique():
    #     df = grp_df2[grp_df2.folder == folder]
    #     sns.lineplot(data=df, x='water_height', y='retinal_speed', hue='spat_frequency')
    #     plt.show()
    #
    # for spat_frequency in grp_df2.spat_frequency.unique():
    #     df = grp_df2[grp_df2.spat_frequency == spat_frequency]
    #     sns.lineplot(data=df, x='retinal_speed', y='absolute_gain', hue='water_height')
    #     plt.show()
    #
    # for folder in grp_df2.folder.unique():
    #     df = grp_df2[grp_df2.folder == folder]
    #     sns.lineplot(data=df, x='y_real_mean', y='retinal_speed', hue='spat_frequency')
    #     plt.show()

"""
    #absolute stim velos
    abs_velos = grp_df2[(np.isclose(grp_df2.u_lin_velocity, 28) | np.isclose(grp_df2.u_lin_velocity, 143) | np.isclose(grp_df2.u_lin_velocity, 286))]
    wh60_abs_velos = grp_df2[(np.isclose(grp_df2.water_height, 60)) & ((np.isclose(grp_df2.u_lin_velocity, 28) | np.isclose(grp_df2.u_lin_velocity, 143) | np.isclose(grp_df2.u_lin_velocity, 286)))]

#angular stim velos (an Waterheight angepasst)
    an_velo25 = grp_df2[np.isclose(grp_df2.u_lin_velocity, 13.3) | np.isclose(grp_df2.u_lin_velocity, 26.6) | np.isclose(grp_df2.u_lin_velocity, 53.2) | np.isclose(grp_df2.u_lin_velocity, -13.3) | np.isclose(grp_df2.u_lin_velocity, -26.6) | np.isclose(grp_df2.u_lin_velocity, -53.2)]
    an_velo50 = grp_df2[((np.isclose(grp_df2.water_height, 30)) & (np.isclose(grp_df2.u_lin_velocity, 28))) | np.isclose(grp_df2.u_lin_velocity, 56) | np.isclose(grp_df2.u_lin_velocity, 111.9) | ((np.isclose(grp_df2.water_height, 30)) & (np.isclose(grp_df2.u_lin_velocity, -28))) | np.isclose(grp_df2.u_lin_velocity, -56) | np.isclose(grp_df2.u_lin_velocity, -111.9)]
    an_velo100 = grp_df2[((np.isclose(grp_df2.water_height, 120)) & (np.isclose(grp_df2.u_lin_velocity, 286))) | ((np.isclose(grp_df2.water_height, 60)) & (np.isclose(grp_df2.u_lin_velocity, 143))) | np.isclose(grp_df2.u_lin_velocity, 71.5) | ((np.isclose(grp_df2.water_height, 120)) & (np.isclose(grp_df2.u_lin_velocity, -286))) | ((np.isclose(grp_df2.water_height, 60)) & (np.isclose(grp_df2.u_lin_velocity, -143))) | np.isclose(grp_df2.u_lin_velocity, -71.5)]
    #all_pos_an_velo = grp_df2[np.isclose(grp_df2.u_lin_velocity, 13.3) | np.isclose(grp_df2.u_lin_velocity, 26.6) | np.isclose(grp_df2.u_lin_velocity, 53.2) | ((np.isclose(grp_df2.water_height, 30)) & (np.isclose(grp_df2.u_lin_velocity, 28))) | np.isclose(grp_df2.u_lin_velocity, 56) | np.isclose(grp_df2.u_lin_velocity, 111.9) | ((np.isclose(grp_df2.water_height, 120)) & (np.isclose(grp_df2.u_lin_velocity, 286))) | ((np.isclose(grp_df2.water_height, 60)) & (np.isclose(grp_df2.u_lin_velocity, 143))) | np.isclose(grp_df2.u_lin_velocity, 71.5)]
    all_an_velo = grp_df2[np.isclose(grp_df2.u_lin_velocity, 13.3) | np.isclose(grp_df2.u_lin_velocity, 26.6) | np.isclose(grp_df2.u_lin_velocity, 53.2) | np.isclose(grp_df2.u_lin_velocity, -13.3) | np.isclose(grp_df2.u_lin_velocity, -26.6) | np.isclose(grp_df2.u_lin_velocity, -53.2) | ((np.isclose(grp_df2.water_height, 30)) & (np.isclose(grp_df2.u_lin_velocity, 28))) | np.isclose(grp_df2.u_lin_velocity, 56) | np.isclose(grp_df2.u_lin_velocity, 111.9) | ((np.isclose(grp_df2.water_height, 30)) & (np.isclose(grp_df2.u_lin_velocity, -28))) | np.isclose(grp_df2.u_lin_velocity, -56) | np.isclose(grp_df2.u_lin_velocity, -111.9) | ((np.isclose(grp_df2.water_height, 120)) & (np.isclose(grp_df2.u_lin_velocity, 286))) | ((np.isclose(grp_df2.water_height, 60)) & (np.isclose(grp_df2.u_lin_velocity, 143))) | np.isclose(grp_df2.u_lin_velocity, 71.5) | ((np.isclose(grp_df2.water_height, 120)) & (np.isclose(grp_df2.u_lin_velocity, -286))) | ((np.isclose(grp_df2.water_height, 60)) & (np.isclose(grp_df2.u_lin_velocity, -143))) | np.isclose(grp_df2.u_lin_velocity, -71.5)]

# Water height: stim velo und gain
    wh30 = all_an_velo[(np.isclose(all_an_velo.water_height, 30))]
    wh60 = all_an_velo[(np.isclose(all_an_velo.water_height, 60))]
    wh120 = all_an_velo[(np.isclose(all_an_velo.water_height, 120))]
    plt.plot(wh30.u_lin_velocity, wh30.gain, color='green', marker='.', linestyle='none', markersize=5)
    plt.plot(wh60.u_lin_velocity, wh60.gain, color='red', marker='.', linestyle='none', markersize=5)
    plt.plot(wh120.u_lin_velocity, wh120.gain, color='blue', marker='.', linestyle='none', markersize=5)
    sns.lineplot(wh30['u_lin_velocity'], wh30['gain'], err_style=None)
    sns.lineplot(wh60['u_lin_velocity'], wh60['gain'], err_style=None)
    sns.lineplot(wh120['u_lin_velocity'], wh120['gain'], err_style=None)


#die drei spatial periods: 0.01 0.02 0.04
    lowper1 = grp_df2[((np.isclose(grp_df2.u_spat_period, 13.30)) | (np.isclose(grp_df2.u_spat_period, 26.60)) | (np.isclose(grp_df2.u_spat_period, 53.21)))]
    midper2 = grp_df2[((np.isclose(grp_df2.u_spat_period, 27.98)) | (np.isclose(grp_df2.u_spat_period, 55.96)) | (np.isclose(grp_df2.u_spat_period, 111.91)))]
    highper4 = grp_df2[((np.isclose(grp_df2.u_spat_period, 71.51)) | (np.isclose(grp_df2.u_spat_period, 143.01)) | (np.isclose(grp_df2.u_spat_period, 286.02)))]
    wh30 = all_an_velo[(np.isclose(all_an_velo.water_height, 30))]
    wh60 = all_an_velo[(np.isclose(all_an_velo.water_height, 60))]
    wh120 = all_an_velo[(np.isclose(all_an_velo.water_height, 120))]
    sns.lineplot(lowper1['u_lin_velocity'], lowper1['gain'], err_style=None)
    sns.lineplot(midper2['u_lin_velocity'], midper2['gain'], err_style=None)
    sns.lineplot(highper4['u_lin_velocity'], highper4['gain'], err_style=None)


    per1 = all_an_velo[((np.isclose(all_an_velo.u_spat_period, 13.30)) | (np.isclose(all_an_velo.u_spat_period, 26.60)) | (np.isclose(all_an_velo.u_spat_period, 53.21)))]
    per2 = all_an_velo[((np.isclose(all_an_velo.u_spat_period, 27.98)) | (np.isclose(all_an_velo.u_spat_period, 55.96)) | (np.isclose(all_an_velo.u_spat_period, 111.91)))]
    per4 = all_an_velo[((np.isclose(all_an_velo.u_spat_period, 71.51)) | (np.isclose(all_an_velo.u_spat_period, 143.01)) | (np.isclose(all_an_velo.u_spat_period, 286.02)))]

    sns.lineplot(per1['u_lin_velocity'], per1['gain'], err_style=None)
    sns.lineplot(per2['u_lin_velocity'], per2['gain'], err_style=None)
    sns.lineplot(per4['u_lin_velocity'], per4['gain'], err_style=None)


    #plt.plot(an_velo25.water_height, an_velo25.gain, color='green', marker='.', linestyle='none', linewidth=0.2, markersize=5)
    #plt.hist(grp_df2.x_vel)
    #plt.plot(grp_df2.x_vel, grp_df2.frame_count, color='green', marker='.', linestyle='dashed', linewidth=0.2, markersize=5)
    #plt.plot(grp_df2.u_lin_velocity, grp_df2.x_vel, color='green', marker='.', linestyle='none', linewidth=0.2, markersize=5)
    #plt.plot(grp_df2.x_vel, grp_df2.x_real_mean, color='green', marker='.', linestyle='none', linewidth=0.2, markersize=5)
    #plt.scatter(grp_df2.x_vel, grp_df2.x_real_mean, color='green', marker='.', linewidth=0.2)


# nur bestimmte Ordner (mit allen Phasen)
    r27_3cmp2 = grp_df2.loc[(grp_df2['folder'] == 'rec_2020-12-17-13-53-27_3cm') & (grp_df2['phase_name'] == 'phase_2')]
    plt.plot(r27_3cmp2.u_lin_velocity, r27_3cmp2.gain, color='green', marker='.', linestyle='none', markersize=5)

    r28_3cm = grp_df2.loc[grp_df2['folder'] == 'rec_2021-02-03-17-05-28_3cm']
    sns.lineplot(r28_3cm['u_lin_velocity'], r28_3cm['gain'], err_style=None)

    r25_6cm = grp_df2.loc[grp_df2['folder'] == 'rec_2020-12-18-13-33-52_6cm']
    plt.plot(r25_6cm.u_lin_velocity, r25_6cm.x_vel, color='green', marker='.', linestyle='none', markersize=5)
    r46_6cm = grp_df.loc[grp_df['folder'] == 'rec_2021-02-03-12-30-46_6cm']
    sns.lineplot(r46_6cm['u_lin_velocity'], r46_6cm['gain'], err_style=None)

    r19_12cm = grp_df2.loc[grp_df2['folder'] == 'rec_2021-02-03-15-40-19_12cm']
    plt.plot(r19_12cm.u_lin_velocity, r19_12cm.x_vel, color='green', marker='.', linestyle='none', markersize=5)
    sns.lineplot(r19_12cm['u_lin_velocity'], r19_12cm['gain'], err_style=None)

    r34_12cm = grp_df2.loc[grp_df2['folder'] == 'rec_2020-12-18-11-38-34_12cm']
    plt.plot(r34_12cm.u_lin_velocity, r34_12cm.x_vel, color='green', marker='.', linestyle='none', markersize=5)




    x_vel_min = grp_df2.x_vel.min()
    x_vel_max = grp_df2.x_vel.max()
    bins = np.linspace(x_vel_min, x_vel_max, 100)
    #plt.hist(grp_df.x_vel.values, bins=bins, width=8, fc=(0., 1., 0., 0.3))
    #plt.hist(grp_df[grp_df.frame_count > 2].x_vel.values, bins=bins, width = 8, fc=(1., 0., 0., 0.3))


    Df_71velo_27per = grp_df2[(np.isclose(grp_df2.u_lin_velocity, 71.5) | np.isclose(grp_df2.u_lin_velocity, -71.5)) & np.isclose(grp_df2.u_spat_period, 27.98)]


    Import IPython
    IPython.embed()
    
    """
