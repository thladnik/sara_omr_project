import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from classify_fish import filter_fish, load_summary
import seaborn as sns
from scipy import misc

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

#drei_27 = grp_df2.loc['rec_2020-12-17-13-53-27_3cm']
# zwölf_19 = grp_df2.loc['rec_2021-02-03-15-40-19_12cm']
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


def test_fun(df):
    import IPython
    IPython.embed()


def get_real_pos(series):
    p = np.array([series.x*10, series.y*10, 300.]) #300 cameradistance, habe einfach x und y in mm umgerechnet
    magnitude = np.linalg.norm(p)
    d = p/magnitude
    m = np.array([0, 0, 1])
    p_new = p + d * 25 * np.dot(m, d) #25 ist hälfte vom becken (annahme fishpos: mitte)
    x_new = p_new[0]
    y_new = p_new[1]
    return (x_new, y_new)

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
    bool_vec = ((df.x_real < -40) & (df.x_real > 40)).values
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

# hier kommt die Funktion, die ich Dir gezeigt hatte
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
    räuml_per_mm = df.u_spat_period
    v_mm_s = df.u_lin_velocity
    periodendauerT_s = räuml_per_mm / v_mm_s
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

# hier habe ich das mit dem dictionary gemacht, funktioniert aber noch nicht ganz
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


if __name__ == '__main__':
    Df1 = filter_fish(load_summary('//172.25.250.112/arrenberg_data/shared/Sara_Widera/Ausgewertet/Summary.h5'))
    #Df1 = pd.read_hdf(os.path.join(base_folder, 'Summary.h5'), 'all')
    #Df1 = Df1[:100000]
    # import IPython
    # IPython.embed()
    Df1['x_real'] = Df1.apply(calc_real_x, axis=1)
    Df1['y_real'] = Df1.apply(calc_real_y, axis=1)
    Df1['water_height'] = Df1.apply(get_waterlevel, axis=1)

    grps1 = Df1.groupby(['folder', 'phase_name', 'particle'])

    Df2 = grps1.apply(make_subparticles).reset_index(drop=True)

    grps2 = Df2.groupby(['folder', 'phase_name', 'particle', 'subparticle'])


    # grp_df = grps2.apply(fill_in_IDs)
    grp_df = pd.DataFrame()
    grp_df = grp_df.reset_index()
    grp_df['x_vel'] = grps2.apply(calc_x_velocity)
    grp_df['y_vel'] = grps2.apply(calc_y_velocity)
    grp_df['folder'] = grps2.apply(fun_get_folder)
    grp_df['phase_name'] = grps2.apply(fun_get_phase_name)
    grp_df['particle'] = grps2.apply(fun_get_particle)
    grp_df['u_lin_velocity'] = grps2.apply(fun_get_u_lin_velocity)
    grp_df['u_spat_period'] = grps2.apply(fun_get_u_spat_period)
    grp_df['frame_count'] = grps2.apply(get_row_count)
    grp_df['absolute_gain'] = grp_df.apply(calc_absolute_gain, axis=1)
    grp_df['x_real_mean'] = grps2.apply(get_x_real_mean)
    grp_df['y_real_mean'] = grps2.apply(get_y_real_mean)
    grp_df['water_height'] = grp_df.apply(get_waterlevel, axis=1)
    grp_df['y_world'] = grp_df.apply(calc_real_world_y, axis=1)
    grp_df['x_world'] = grp_df.apply(calc_real_world_x,axis=1)
    grp_df['retinal_speed'] = grp_df.apply(get_retinal_speed, axis=1)
    grp_df['spat_frequency'] = grp_df.apply(get_spat_freq, axis=1)
    grp_df['angular_gain'] = grp_df.apply(calc_angular_gain, axis=1)
    grp_df['retinal_speed_magnitude'] = grp_df.apply(get_retinal_speed_magnitude, axis=1)
    grp_df['u_lin_velocity_magnitude'] = grp_df.apply(get_u_lin_velocity_magnitude, axis=1)
    grp_df['x_vel_magnitude'] = grp_df.apply(get_x_vel_magnitude, axis=1)
    grp_df['temp_freq'] = grp_df.apply(calc_temp_freq, axis=1)
    grp_df['temp_freq_magnitude'] = grp_df.apply(get_temp_freq_magnitude, axis=1)
    grp_df['actual_retinal_speed'] = grp_df.apply(calc_actual_retinal_speed, axis=1)


    #calc gain_mean for tims figure: gruppieren vom Ausgangsdatensatz und in neues df speichern
    grps3 = grp_df.groupby(['water_height', 'spat_frequency', 'temp_freq'])
    grp_df3 = pd.DataFrame()
    grp_df3['angular_gain_mean'] = grps3.apply(get_angular_gain_mean)
    grp_df3['absolute_gain_mean'] = grps3.apply(get_absolute_gain_mean)
    grp_df3['spat_frequency_mean'] = grps3.apply(get_spat_frequency_mean)
    grp_df3['temp_freq_mean'] = grps3.apply(get_temp_freq_mean)
    grp_df3['temp_freq_magnitude_mean'] = grps3.apply(get_temp_freq_magnitude_mean)
    grp_df3['retinal_speed_mean'] = grps3.apply(get_retinal_speed_mean)
    grp_df3['retinal_speed_magnitude_mean'] = grps3.apply(get_retinal_speed_magnitude_mean)
    grp_df3['water_height_mean'] = grps3.apply(get_waterheight_mean)




    # import IPython
    # IPython.embed()

    grp_df2 = grp_df[(grp_df.x_real_mean > -50) & (grp_df.x_real_mean < 50)]
    grp_df2 = grp_df2[np.isfinite(grp_df2.u_lin_velocity)]
    #shape vergleichen: grp_df2.u_lin_velocity.values dann coralling rausfiltern, shape, dann Corall=... dann shape Diff
    grp_df2 = grp_df2[np.logical_not((np.isclose(grp_df2.u_lin_velocity, 60.)) | (np.isclose(grp_df2.u_lin_velocity, 75.)) | (np.isclose(grp_df2.u_lin_velocity, 100.)))]

    # negatives y_world.min Problem lösen:
    grp_df2 = grp_df2[grp_df2.y_world >= 0]

    # grp_df2.groupby(['u_lin_velocity', 'u_spat_period'])

    grp_df2.to_hdf('temp.h5', 'all')
    #grp_df2.to_excel('data_analysis', 'all')
    grp_df2 = pd.read_hdf('temp.h5', 'all')
    # grp_df2.to_csv('temp.csv')



    # import IPython
    # IPython.embed()



# FILTERED DATA FOR FIGURES:

    all_abs_velo = grp_df2[(np.isclose(grp_df2.u_lin_velocity, 28) | np.isclose(grp_df2.u_lin_velocity, 143) | np.isclose(grp_df2.u_lin_velocity, 286) | np.isclose(grp_df2.u_lin_velocity, -28) | np.isclose(grp_df2.u_lin_velocity, -143) | np.isclose(grp_df2.u_lin_velocity, -286))]

    an_velo25 = grp_df2[np.isclose(grp_df2.u_lin_velocity, 13.3) | np.isclose(grp_df2.u_lin_velocity, 26.6) | np.isclose(grp_df2.u_lin_velocity, 53.2) | np.isclose(grp_df2.u_lin_velocity, -13.3) | np.isclose(grp_df2.u_lin_velocity, -26.6) | np.isclose(grp_df2.u_lin_velocity, -53.2)]
    an_velo50 = grp_df2[((np.isclose(grp_df2.water_height, 30)) & (np.isclose(grp_df2.u_lin_velocity, 28))) | np.isclose(grp_df2.u_lin_velocity, 56) | np.isclose(grp_df2.u_lin_velocity, 111.9) | ((np.isclose(grp_df2.water_height, 30)) & (np.isclose(grp_df2.u_lin_velocity, -28))) | np.isclose(grp_df2.u_lin_velocity, -56) | np.isclose(grp_df2.u_lin_velocity, -111.9)]
    an_velo100 = grp_df2[((np.isclose(grp_df2.water_height, 120)) & (np.isclose(grp_df2.u_lin_velocity, 286))) | ((np.isclose(grp_df2.water_height, 60)) & (np.isclose(grp_df2.u_lin_velocity, 143))) | np.isclose(grp_df2.u_lin_velocity, 71.5) | ((np.isclose(grp_df2.water_height, 120)) & (np.isclose(grp_df2.u_lin_velocity, -286))) | ((np.isclose(grp_df2.water_height, 60)) & (np.isclose(grp_df2.u_lin_velocity, -143))) | np.isclose(grp_df2.u_lin_velocity, -71.5)]
    all_an_velo = grp_df2[np.isclose(grp_df2.u_lin_velocity, 13.3) | np.isclose(grp_df2.u_lin_velocity, 26.6) | np.isclose(grp_df2.u_lin_velocity, 53.2) | np.isclose(grp_df2.u_lin_velocity, -13.3) | np.isclose(grp_df2.u_lin_velocity, -26.6) | np.isclose(grp_df2.u_lin_velocity, -53.2) | ((np.isclose(grp_df2.water_height, 30)) & (np.isclose(grp_df2.u_lin_velocity, 28))) | np.isclose(grp_df2.u_lin_velocity, 56) | np.isclose(grp_df2.u_lin_velocity, 111.9) | ((np.isclose(grp_df2.water_height, 30)) & (np.isclose(grp_df2.u_lin_velocity, -28))) | np.isclose(grp_df2.u_lin_velocity, -56) | np.isclose(grp_df2.u_lin_velocity, -111.9) | ((np.isclose(grp_df2.water_height, 120)) & (np.isclose(grp_df2.u_lin_velocity, 286))) | ((np.isclose(grp_df2.water_height, 60)) & (np.isclose(grp_df2.u_lin_velocity, 143))) | np.isclose(grp_df2.u_lin_velocity, 71.5) | ((np.isclose(grp_df2.water_height, 120)) & (np.isclose(grp_df2.u_lin_velocity, -286))) | ((np.isclose(grp_df2.water_height, 60)) & (np.isclose(grp_df2.u_lin_velocity, -143))) | np.isclose(grp_df2.u_lin_velocity, -71.5)]

    an_velo25_50 = grp_df2[((np.isclose(grp_df2.water_height, 30)) & (np.isclose(grp_df2.u_lin_velocity, 28))) | np.isclose(grp_df2.u_lin_velocity, 56) | np.isclose(grp_df2.u_lin_velocity, 111.9) | ((np.isclose(grp_df2.water_height, 30)) & (np.isclose(grp_df2.u_lin_velocity, -28))) | np.isclose(grp_df2.u_lin_velocity, -56) | np.isclose(grp_df2.u_lin_velocity, -111.9) | np.isclose(grp_df2.u_lin_velocity, 13.3) | np.isclose(grp_df2.u_lin_velocity, 26.6) | np.isclose(grp_df2.u_lin_velocity, 53.2) | np.isclose(grp_df2.u_lin_velocity, -13.3) | np.isclose(grp_df2.u_lin_velocity, -26.6) | np.isclose(grp_df2.u_lin_velocity, -53.2)]


    freq4 = grp_df2[np.isclose(grp_df2.spat_frequency, 0.04)]
    freq2 = grp_df2[np.isclose(grp_df2.spat_frequency, 0.02)]
    freq1 = grp_df2[np.isclose(grp_df2.spat_frequency, 0.01)]

    an_velo_freq2 = all_an_velo[np.isclose(all_an_velo.spat_frequency, 0.02)]
    abs_velo_freq2 = all_abs_velo[np.isclose(all_abs_velo.spat_frequency, 0.02)]

    an_velo25_freq2 = an_velo25[np.isclose(an_velo25.spat_frequency, 0.02)]


    wh30 = grp_df3[(np.isclose(grp_df3.water_height_mean, 30))]
    wh60 = grp_df3[(np.isclose(grp_df3.water_height_mean, 60))]
    wh120 = grp_df3[(np.isclose(grp_df3.water_height_mean, 120))]


    import IPython
    IPython.embed()

    # Stats = pd.DataFrame(grp_df2, columns = ['folder', 'phase_name', 'particle', 'u_lin_velocity', 'frame_count', 'x_vel', 'x_vel_magnitude', 'absolute_gain', 'angular_gain', 'y_world', 'water_height', 'retinal_speed', 'retinal_speed_magnitude', 'spat_frequency'])
    grp_df2.to_excel(r'F:\Sara_Widera\statistics.xlsx', index=True, header=True)

# FIGURES FOR THESIS:

#1 GAIN TUNING FUNCTIONS
    # absolute Gain
    ax = sns.relplot(data=grp_df2, x='retinal_speed', y='absolute_gain', hue='water_height', palette='dark', col='spat_frequency', kind='line')
    ax.set(xlabel='retinal speed [°/s]', ylabel='absolute gain', title='tuning function (all data)')

    # relative Gain
    ax = sns.relplot(data=grp_df2, x='retinal_speed', y='angular_gain', hue='water_height', palette='dark', col='spat_frequency', kind='line')
    ax.set(xlabel='retinal speed [°/s]', ylabel='relative gain', title='tuning function (all data)')

#tuning func für alle angular und für alle absoluten velos?

#2 GAIN AT DIFFERENT STIMULUS SPEEDS
    #(overview)
    ax = sns.relplot(data=grp_df2, x='retinal_speed', y='absolute_gain', hue='water_height', palette='dark', col='spat_frequency')
    ax.set(xlabel='(all angular velocities) retinal speed [°/s]', ylabel='absolute gain')

    # violin plot: absolute Gain, angular velocities, freq 0.02
    ax = sns.violinplot(data=an_velo_freq2, x='retinal_speed', y='absolute_gain', hue='water_height')
    ax.set(xlabel='retinal speed [°/s]', ylabel='absolute gain')
    ax.set_title('all angular velocities at spat freq 0.02 c/°')

    # violin plot: absolute Gain, absolute velocities, freq 0.02
    ax = sns.violinplot(data=abs_velo_freq2, x='retinal_speed', y='absolute_gain', hue='water_height')
    ax.set(xlabel='retinal speed [°/s]', ylabel='absolute gain', title='all absolute velocities at spat freq 0.02 c/°')


    # absolute Gain, angular velocities gepoolt (spat_freq = 0.02)
    ax = sns.relplot(data=an_velo_freq2, x='retinal_speed_magnitude', y='absolute_gain', hue='water_height', palette='dark', col='spat_frequency', kind='line')
    ax.set(xlabel='magnitude of retinal speed [°/s]', ylabel='absolute gain', title='all angular velocities at spatial frequency 0.02 c/°')

    # relative Gain, angular velocities gepoolt (spat_freq = 0.02)
    ax = sns.relplot(data=an_velo_freq2, x='retinal_speed_magnitude', y='angular_gain', hue='water_height', palette='dark', col='spat_frequency', kind='line')
    ax.set(xlabel='magnitude of retinal speed [°/s]', ylabel='relative gain', title='all angular velocities at spatial frequency 0.02 c/°')

    # absolute Gain, absolute velocities gepoolt (spat_freq = 0.02)
    ax = sns.relplot(data=abs_velo_freq2, x='retinal_speed_magnitude', y='absolute_gain', hue='water_height', palette='dark', col='spat_frequency', kind='line')
    ax.set(xlabel='magnitude of retinal speed [°/s]', ylabel='absolute gain',  title='all absolute velocities at spatial frequency 0.02 c/°')

    # relative Gain, absolute velocities gepoolt (spat_freq = 0.02)
    ax = sns.relplot(data=abs_velo_freq2, x='retinal_speed_magnitude', y='angular_gain', hue='water_height', palette='dark', col='spat_frequency', kind='line')
    ax.set(xlabel='magnitude of retinal speed [°/s]', ylabel='relative gain', title='all absolute velocities at spatial frequency 0.02 c/°')


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

    ax = sns.relplot(data=grp_df2, x='retinal_speed_magnitude', y='y_world', hue='water_height', palette='dark', col='spat_frequency', kind='line')
# Fish velocity at different speeds


#5 TUNING FOR SWIMMING SPEED
    ax = sns.relplot(data=grp_df2, x='retinal_speed', y='x_vel', hue='water_height', palette='dark', col='spat_frequency', kind='line')
    ax.set(xlabel='retinal speed [°/s]', ylabel='y-position [mm]')


#6 Y POSITION (WORLD) AT DIFFERENT RETINAL SPEEDS
    ax = sns.relplot(data=grp_df2, x='retinal_speed', y='y_world', hue='water_height', palette='dark', col='spat_frequency', kind='line')
    ax.set(xlabel='retinal speed [°/s]', ylabel='y-position [mm]')
    ax = sns.relplot(data=an_velo25_50, x='temp_freq_magnitude', y='x_vel', hue='retinal_speed_magnitude', palette='dark', col='water_height', kind='line')


#7 GAIN COLOURMAP: SPAT FREQ, TEMP FREQ, RETINAL SPEED
    # sns.set()
    # ax = sns.scatterplot(data=grp_df2, x="retinal_speed_magnitude", y="temp_freq_magnitude", hue="angular_gain", palette='gist_rainbow')
    # norm = plt.Normalize(grp_df2['angular_gain'].min(), grp_df2['angular_gain'].max())
    # sm = plt.cm.ScalarMappable(cmap="gist_rainbow", norm=norm)
    # sm.set_array([])
    # # Remove the legend and add a colorbar
    # ax.get_legend().remove()
    # ax.figure.colorbar(sm)
    # plt.show()

#----> TIMs COLOURMAP :)
    heatmap_size = (12.5, 11.5)

    cmap_scheme = 'viridis'
    markersize = 17

    fig = custom_fig('angular Gain for spat & temp freq at waterheight 30mm', heatmap_size)
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogy([],[])
    # add colorbar
    minval = np.floor(np.min(wh30['angular_gain_mean']) * 10) / 10
    maxval = np.ceil(np.max(wh30['angular_gain_mean']) * 10) / 10
    zticks = np.arange(minval, maxval + 0.01, 0.1)
    cax = ax.imshow(np.concatenate((zticks, zticks)).reshape((2, -1)), interpolation='nearest', cmap=cmap_scheme)
    cbar = fig.colorbar(cax, ticks=zticks)
    cbar.ax.set_ylabel('angular gain')
    cmap = np.asarray(cbar.cmap.colors)
    valmap = np.arange(minval, maxval, (maxval - minval) / cmap.shape[0])
    plt.cla()  # clears imshow plot, but keeps the colorbar


    for sfreq, tfreq, gain in zip(wh30['spat_frequency_mean'], wh30['temp_freq_mean'], wh30['angular_gain_mean']):
        plot_colorpoint(ax, sfreq, tfreq, gain, cmap, valmap)

    xlim = [0.008, 0.1]
    ylim = [0.08, 6.5]

    ax.set_xlabel('spatial frequency [cyc/deg]')
    ax.set_ylabel('temporal frequency [cyc/s]')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    adjust_spines(ax)

    fig.savefig('../angular_gain_heatmap_wh30.svg', format='svg')

    plt.show()




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
