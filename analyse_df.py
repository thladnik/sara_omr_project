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

# drei_28 = grp_df2.loc['rec_2021-02-03-17-05-28_3cm']

#base_folder = '//172.25.250.112/arrenberg_data/shared/Sara_Widera/Ausgewertet'
base_folder = './data/'

def plot_colorpoint(ax, x, y, z, cmap, valmap, fontsize = 9):
    if sum(z < valmap) > len(valmap) / 2:
        fontcolor = 'w'
    else:
        fontcolor = 'k'
    ax.loglog(x, y, marker='o', markersize=17, color=cmap[np.argmin(np.abs(valmap - z)), :])
    ax.text(x, y, str(round(z, 1)), fontsize=fontsize, color=fontcolor,  ha='center', va='center')



def calc_x_velocity(df):
    # Calculate frame-by-frame time and x position differences

    #Startzeitpunkt von allen weiteren Werten subtrahieren (time = Sek seit dem Jahre 1970)
    # df['time'] -= df['time'].values[0]
    # start_time = 1
    # end_time = 30
    # df = df[(df['time'] >= start_time) & (df['time'] < end_time)]


    dt = df.time.diff()
    dx = df.x_real.diff()

    # Calculate velocity form position delta and time delta
    vx = dx/dt

    # Visualization (does it for each particle (MANY particles in Df) -> not for analysis
    # plt.plot(df.time.values - df.time.values[0], vx)
    # plt.show()

    # Inspect:
    # import IPython
    # IPython.embed()

    # Return mean x velocity for this particular particle
    return vx.mean()

def calc_y_velocity(df):
    # Calculate frame-by-frame time and y position differences

    # df['time'] -= df['time'].values[0]
    # start_time = 1
    # end_time = 30
    # df = df[(df['time'] >= start_time) & (df['time'] < end_time)]

    dt = df.time.diff()
    dy = df.y_real.diff()
    vy = dy/dt
    # Return mean y velocity for this particular particle
    return vy.mean()

def get_real_pos(series):
    p = np.array([series.x, series.y, 30.]) * 10. #30cm cameradistance, habe einfach x und y in mm umgerechnet
    d = p/np.linalg.norm(p)
    m = np.array([0., 0., 1.])
    p_new = p + d * 25 * np.dot(m, d) #25 ist hälfte vom becken (annahme fishpos: mitte)
    # x_new = p_new[0]
    # y_new = p_new[1]
    return p_new[:2]

def calc_real_x(series):
    return get_real_pos(series)[0]

def calc_real_y(series):
    return get_real_pos(series)[1]

def fun_get_folder(df):
    return df.folder.unique()[0]

def fun_get_phase_name(df):
    return df.phase_name.unique()[0]

def fun_get_particle(df):
    return df.particle.unique()[0]

def fun_get_u_lin_velocity(df):
    return df.u_lin_velocity.unique()[0]

def fun_get_u_spat_period(df):
    return df.u_spat_period.unique()[0]

def get_row_count(df):
    return df.shape[0]

def get_x_real_mean(df):
    return df.x_real.mean()

def get_y_real_mean(df):
    return df.y_real.mean()

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
    bool_vec = ((df.x_real > -45) & (df.x_real < 45)).values
    borders = np.where(np.diff(bool_vec))[0]  #gibt array mit den positionen bei einem übergang, also von true (1) zu false (0) bei filter_conditon und umgekehrt mithilfe der differenz der benachbarten einträge
    outer = [0, *borders+1, bool_vec.shape[0]+1]
    df = df.reset_index()
    for id, (i, j) in enumerate(zip(outer[:-1], outer[1:])):
        df.loc[i:j, 'subparticle'] = id
    new_df = df
    return new_df

def fill_in_IDs(df):
    return pd.DataFrame({'folder': df.folder.unique(),
                         'phase_name': df.phase_name.unique(),
                         'particle': df.particle.unique(),
                         'subparticle': df.subparticle.unique()})

def get_retinal_speed(series):
    wl = series.water_height #mm
    stim_vel = series.u_lin_velocity #mm/s
    rad_retinal_speed = 2*np.arctan(stim_vel/(2*wl))
    retinal_speed = 360/(2*np.pi)*rad_retinal_speed
    round_retinal_speed = round(retinal_speed, 1)
    return round_retinal_speed

#TODO: tatsächliche Schwimmhöhe nehmen, also richtig machen (?)
def calc_actual_retinal_speed(df):
    ypos = df.y_real_mean
    stim_vel = df.u_lin_velocity
    rad_retinal_speed = 2 * np.arctan(stim_vel / (2 * ypos))
    speed = 360 / (2 * np.pi) * rad_retinal_speed
    return speed


def get_spat_freq(series):
    wl = series.water_height
    spat_period = series.u_spat_period
    spat_freq = (2*np.pi) / (360*2*np.arctan(spat_period/(2*wl)))
    round_spat_freq = round(spat_freq, 4)
    return round_spat_freq


def calc_temp_freq(df):
    raeuml_per_mm = df.u_spat_period # Umlaute und Sonderzeichen (ausser "_") sind beim Programmieren keine gute Idee
    v_mm_s = df.u_lin_velocity
    periodendauerT_s = raeuml_per_mm / v_mm_s
    temp_freq_1_s = 1 / periodendauerT_s
    return temp_freq_1_s

def get_temp_freq_magnitude(series):
    magnitude = np.linalg.norm(series.temp_freq)
    return magnitude

def get_retinal_speed_magnitude(series):
    magnitude = np.linalg.norm(series.retinal_speed)
    return magnitude

def get_u_lin_velocity_magnitude(series):
    magnitude = np.linalg.norm(series.u_lin_velocity)
    return magnitude

def get_x_vel_magnitude(series):
    magnitude = np.linalg.norm(series.x_vel)
    return magnitude

# hier habe ich das mit dem dictionary jetzt so gemacht:
def calc_real_world_y(df):
    correction = 0.5 * dict_dimensions[df.folder][1]
    y_new = df.y_real_mean + correction
    return y_new

def calc_real_world_x(df):
    correction = 0.5 * dict_dimensions[df.folder][0]
    x_new = df.x_real_mean + correction
    return x_new

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


if __name__ == '__main__':

    if 'recalc' in sys.argv or not(os.path.exists('Summary_final.h5')):
        print('Recalculating final DataFrame')

        classify_fish.probability_threshold = 0.75
        classify_fish.annot_path = 'annotations'

        Df = filter_fish(load_summary('//172.25.250.112/arrenberg_data/shared/Sara_Widera/Ausgewertet/Summary.h5'))
        #Df = pd.read_hdf(os.path.join(base_folder, 'Summary.h5'), 'all')

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

        # Filter out all coralling stimulus phases
        Df = Df[np.logical_not((np.isclose(Df.u_lin_velocity, 60.)) | (np.isclose(Df.u_lin_velocity, 75.)) | (np.isclose(Df.u_lin_velocity, 100.)))]


        # save
        Df.to_hdf('Summary_final.h5', 'all')
        Df.to_excel('Summary_final.xlsx')

    Df = pd.read_hdf('Summary_final.h5', 'all')
    # Df_gain_heat = pd.read_hdf('Summary_final.h5', 'all')

    # create Df for Tims figure (GAIN heatmaplike): gruppieren vom Ausgangsdatensatz und in neues df speichern
    grps_heat = Df.groupby(['water_height', 'spat_frequency', 'temp_freq'])
    Df_gain_heat = pd.DataFrame()
    Df_gain_heat['angular_gain_mean'] = grps_heat.apply(get_angular_gain_mean)
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

    anvel25 = Df[np.isclose(Df.retinal_speed, 25, atol=0.3, rtol=0.3)]

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
# alle in einzelnen Skripten bis auf diese:

#5 TUNING FOR SWIMMING SPEED
    ax = sns.relplot(data=Df, x='retinal_speed', y='x_vel', hue='water_height', palette='dark', col='spat_frequency', kind='line')
    ax.set(xlabel='retinal speed [°/s]', ylabel='y-position [mm]')
# todo: wenn, dann poolen...



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
