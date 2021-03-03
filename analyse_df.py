import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


dict_waterlevel = {'rec_2020-12-17-13-53-27_3cm': 30, 'rec_2020-12-18-09-46-26_6cm': 60,
                   'rec_2020-12-17-14-38-13_6cm': 60, 'rec_2020-12-18-10-31-28_3cm': 30,
                   'rec_2020-12-18-11-38-34_12cm': 120, 'rec_2020-12-18-13-33-52_6cm': 60,
                   'rec_2020-12-18-14-31-07_3cm': 30}

#base_folder = '//172.25.250.112/arrenberg_data/shared/Sara_Widera/Ausgewertet'
base_folder = './data/'

def calc_x_velocity(df):
    # Calculate frame-by-frame time and x position differences
    dt = df.time.diff()
    dx = df.x.diff()

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
    dt = df.time.diff()
    dy = df.y.diff()

    # Calculate velocity form position delta and time delta
    vy = dy/dt

    # Visualization (does it for each particle (MANY particles in Df) -> not for analysis
    # plt.plot(df.time.values - df.time.values[0], vy)
    # plt.show()

    # Inspect:
    # import IPython
    # IPython.embed()

    # Return mean y velocity for this particular particle
    return vy.mean()


def test_fun(df):
    import IPython
    IPython.embed()


def get_real_pos(series):
    p = np.array([series.x, series.y, 300.])
    magnitude = np.linalg.norm(p)
    d = p/magnitude
    m = np.array([0, 0, 1])
    p_new = p + d * 25 * np.dot(m, d)
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

def calc_gain(series):
    fish_vel = series.x_vel
    stim_vel = series.u_lin_velocity
    gain = fish_vel / stim_vel
    return gain

def get_waterlevel(series):
    path = series.folder.split(os.sep)
    wl = dict_waterlevel.get(path[-1])
    return wl

def make_subparticles(df):
    bool_vec = ((df.x_real < -50) & (df.x_real > 50)).values
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


if __name__ == '__main__':
    Df1 = pd.read_hdf(os.path.join(base_folder, 'Summary.h5'), 'all')
    #Df1 = Df1[:100000]
    import IPython
    IPython.embed()
    Df1['x_real'] = Df1.apply(calc_real_x, axis=1)
    Df1['y_real'] = Df1.apply(calc_real_y, axis=1)
    Df1['water_height'] = Df1.apply(get_waterlevel, axis=1)

    grps1 = Df1.groupby(['folder', 'phase_name', 'particle'])

    Df2 = grps1.apply(make_subparticles).reset_index(drop=True)

    grps2 = Df2.groupby(['folder', 'phase_name', 'particle', 'subparticle'])


    grp_df = grps2.apply(fill_in_IDs)
    grp_df['x_vel'] = grps2.apply(calc_x_velocity)
    grp_df['y_vel'] = grps2.apply(calc_y_velocity)
    grp_df['folder'] = grps2.apply(fun_get_folder)
    grp_df['phase_name'] = grps2.apply(fun_get_phase_name)
    grp_df['particle'] = grps2.apply(fun_get_particle)
    grp_df['u_lin_velocity'] = grps2.apply(fun_get_u_lin_velocity)
    grp_df['u_spat_period'] = grps2.apply(fun_get_u_spat_period)
    grp_df['frame_count'] = grps2.apply(get_row_count)
    grp_df['gain'] = grp_df.apply(calc_gain, axis=1)
    grp_df['x_real_mean'] = grps2.apply(get_x_real_mean)
    import IPython
    IPython.embed()

    grp_df2 = grp_df[(grp_df.x_real_mean < -50) | (grp_df.x_real_mean > 50)]

    grp_df2 = grp_df2[np.isfinite(grp_df2.u_lin_velocity)]

    #shape vergleichen: grp_df2.u_lin_velocity.values dann coralling rausfiltern, shape, dann Corall=... dann shape Diff

    grp_df2 = grp_df2[np.logical_not((np.isclose(grp_df2.u_lin_velocity, 60.)) | (np.isclose(grp_df2.u_lin_velocity, 75.)) | (np.isclose(grp_df2.u_lin_velocity, 100.)))]


    grp_df2.groupby(['u_lin_velocity', 'u_spat_period'])

    #plt.hist(grp_df2.x_vel)
    #plt.plot(grp_df2.x_vel, grp_df2.frame_count, color='green', marker='.', linestyle='dashed', linewidth=0.2, markersize=5)
    #plt.plot(grp_df2.x_vel, grp_df2.x_real_mean, color='green', marker='.', linestyle='none', linewidth=0.2, markersize=5)
    #plt.scatter(grp_df2.x_vel, grp_df2.x_real_mean, color='green', marker='.', linewidth=0.2)


    x_vel_min = grp_df2.x_vel.min()
    x_vel_max = grp_df2.x_vel.max()
    bins = np.linspace(x_vel_min, x_vel_max, 100)
    #plt.hist(grp_df.x_vel.values, bins=bins, width=8, fc=(0., 1., 0., 0.3))
    #plt.hist(grp_df[grp_df.frame_count > 2].x_vel.values, bins=bins, width = 8, fc=(1., 0., 0., 0.3))


    Df_71velo_27per = grp_df[(np.isclose(grp_df.u_lin_velocity, 71.5) | np.isclose(grp_df.u_lin_velocity, -71.5)) & np.isclose(grp_df.u_spat_period, 27.98)]
