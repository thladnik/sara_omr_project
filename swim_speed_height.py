from analyse_df import *


# THREE figures in ONE skript:

if __name__ == '__main__':
# load Data
    base_folder = './data/'
    #Df = pd.read_hdf('Summary_final.h5', 'all')
    Df = pd.read_hdf('Summary_final.h5', 'by_subparticle')

    stim_ang_vel_int_bins = pd.cut(Df.stim_ang_vel_int, 90)
    Df['stim_ang_vel_int_bins'] = [stim_ang_vel_int_bins_calc_mid.mid for stim_ang_vel_int_bins_calc_mid in stim_ang_vel_int_bins]

    sns.set(font_scale=1.4)
    sns.set_style('ticks')

    import IPython
    IPython.embed()



# Swimming height:
# Y WORLD FISH POSITION at different RETINAL SPEEDS

    ax = sns.relplot(data=Df, x='stim_ang_vel_int', y='y_world_mean', hue='water_height', palette='dark', col='stim_ang_sf', kind='line')
    ax.set(xlabel='retinal speed [deg/sec]', ylabel='swimming height [mm]')
    ax.set_titles('spatial frequency = {col_name} cyc/deg')
    ax._legend.set_title('water height\n[mm]')
    new_labels = ['30', '60', '120']
    for t, l in zip(ax._legend.texts, new_labels): t.set_text(l)
    plt.tight_layout()
    plt.show()


# TUNING SWIMMING SPEED

# retinal speed
    ax = sns.relplot(data=Df, x='stim_ang_vel_int_bins', y='x_vel_mean', hue='water_height', palette='dark', col='stim_ang_sf', kind='line')
    ax.set(xlabel='retinal speed [deg/sec]', ylabel='swimming speed [mm/sec]')
    ax.set_titles('spatial frequency = {col_name} cyc/deg')
    ax._legend.set_title('water height\n[mm]')
    new_labels = ['30', '60', '120']
    for t, l in zip(ax._legend.texts, new_labels): t.set_text(l)
    plt.tight_layout()
    plt.show()
# linear velocity
    ax = sns.relplot(data=Df, x='stim_lin_vel_int', y='x_vel_mean', hue='water_height', palette='dark', col='stim_ang_sf', kind='line')
    ax.set(xlabel='linear stimulus velocity [mm/sec]', ylabel='swimming speed [mm/sec]')
    ax.set_titles('spatial frequency = {col_name} cyc/deg')
    ax._legend.set_title('water height\n[mm]')
    new_labels = ['30', '60', '120']
    for t, l in zip(ax._legend.texts, new_labels): t.set_text(l)
    plt.tight_layout()
    plt.show()

# beide Achsen gepoolt
    ax = sns.relplot(data=Df, x='stim_ang_vel_int_abs', y='x_vel_mean_abs', hue='water_height', palette='dark', col='stim_ang_sf', kind='line')
    ax.set(xlabel='magnitude of retinal speed [deg/sec]', ylabel='swimming speed [mm/sec]')
    ax.set_titles('spatial frequency = {col_name} cyc/deg')
    plt.tight_layout()
    plt.show()

# nochmal nur col und hue getauscht, machts aber nicht besser
    ax = sns.relplot(data=Df, x='stim_ang_vel_int', y='x_vel_mean', hue='stim_ang_sf', palette='dark', col='water_height', kind='line')
    # ax.set(xlabel='retinal speed [deg/sec]', ylabel='absolute gain')
    # ax.set_titles('{col_var} = {col_name} bzw spatial frequency [cyc/deg]')
    plt.tight_layout()
    plt.show()
    ax = sns.relplot(data=Df, x='stim_ang_vel_int_abs', y='x_vel_mean_abs', hue='stim_ang_sf', palette='dark', col='water_height', kind='line')
    # ax.set(xlabel='retinal speed [deg/sec]', ylabel='absolute gain')
    # ax.set_titles('{col_var} = {col_name} bzw spatial frequency [cyc/deg]')
    plt.tight_layout()
    plt.show()




# SWIMMING SPEED tf

# gegen TEMPORAL FREQUENCY

    #Df['stim_ang_tf_abs'] = Df['stim_ang_tf'].abs()
    stim_ang_tf_abs_bins = pd.cut(Df.stim_ang_tf_abs, 60)
    Df['stim_ang_tf_abs_bins'] = [stim_ang_tf_abs_bins_calc_mid.mid for stim_ang_tf_abs_bins_calc_mid in stim_ang_tf_abs_bins]

    ax = sns.relplot(data=Df, x='stim_ang_tf_abs_bins', y='x_vel_mean_abs', hue='water_height', palette='dark', col='stim_ang_sf',kind='line')
    ax.set(xlabel='|temporal frequency| [cyc/sec]', ylabel='|swimming speed| [mm/sec]')
    ax.set_titles('spatial frequency = {col_name} cyc/deg')
    ax._legend.set_title('water height\n[mm]')
    new_labels = ['30', '60', '120']
    for t, l in zip(ax._legend.texts, new_labels): t.set_text(l)
    plt.tight_layout()
    plt.show()

# nur swim speed gepoolt:
    stim_ang_tf_bins = pd.cut(Df.stim_ang_tf, 10)
    Df['stim_ang_tf_bins'] = [stim_ang_tf_bins_calc_mid.mid for stim_ang_tf_bins_calc_mid in stim_ang_tf_bins]

    ax = sns.relplot(data=Df, x='stim_ang_tf_bins', y='x_vel_mean_abs', hue='water_height', palette='dark',col='stim_ang_sf', kind='line')
    ax.set(xlabel='temporal frequency [cyc/sec]', ylabel='swimming speed magnitude [mm/sec]')
    ax.set_titles('spatial frequency = {col_name} cyc/deg')
    ax._legend.set_title('water height\n[mm]')
    plt.tight_layout()
    plt.show()



# absolute GAIN GEGEN TF (eigl in gain_tuning_func.py)
    ax = sns.relplot(data=Df, x='stim_ang_tf', y='x_lin_gain', hue='water_height', palette='dark', col='stim_ang_sf',kind='line')
    ax.set(xlabel='temporal frequency [cyc/sec]', ylabel='absolute gain')
    ax.set_titles('spatial frequency = {col_name} cyc/deg')
    plt.tight_layout()
    plt.show()





    quit()


# FISH SWIMMING VELOCITY at different WATER HEIGHTS (for matlab hypothesis)

# alle stim ang velos (gepoolt und auch schwimmgeschwk gepoolt)

    all_an_velo = Df[np.isclose(Df.stim_lin_vel, 13.3) | np.isclose(Df.stim_lin_vel, 26.6) | np.isclose(Df.stim_lin_vel, 53.2) | np.isclose(Df.stim_lin_vel, -13.3) | np.isclose(Df.stim_lin_vel, -26.6) | np.isclose(Df.stim_lin_vel, -53.2) | ((np.isclose(Df.water_height, 30)) & (np.isclose(Df.stim_lin_vel, 28))) | np.isclose(Df.stim_lin_vel, 56) | np.isclose(Df.stim_lin_vel, 111.9) | ((np.isclose(Df.water_height, 30)) & (np.isclose(Df.stim_lin_vel, -28))) | np.isclose(Df.stim_lin_vel, -56) | np.isclose(Df.stim_lin_vel, -111.9) | ((np.isclose(Df.water_height, 120)) & (np.isclose(Df.stim_lin_vel, 286))) | ((np.isclose(Df.water_height, 60)) & (np.isclose(Df.stim_lin_vel, 143))) | np.isclose(Df.stim_lin_vel, 71.5) | ((np.isclose(Df.water_height, 120)) & (np.isclose(Df.stim_lin_vel, -286))) | ((np.isclose(Df.water_height, 60)) & (np.isclose(Df.stim_lin_vel, -143))) | np.isclose(Df.stim_lin_vel, -71.5)]

    # all_stim_ang_vel = Df[np.isclose(Df.stim_ang_vel, 25.0) | np.isclose(Df.stim_ang_vel, -25.0) | np.isclose(Df.stim_ang_vel, 50.0) | np.isclose(Df.stim_ang_vel, -50.0) | np.isclose(Df.stim_ang_vel, 100.0) | np.isclose(Df.stim_ang_vel, -100.0)]

    ax = sns.relplot(data=all_an_velo, x='water_height', y='x_vel_mean_abs', hue='stim_ang_sf', palette='dark', col='stim_ang_vel_int_abs', kind='line')
    ax.set(xlabel='water height [mm]', ylabel='swimming speed magnitude [mm/sec]')
    ax.set_titles('retinal speed magnitude = {col_name} deg/sec')
    ax._legend.set_title('spatial\nfrequency\n[cyc/deg]')
    plt.tight_layout()
    plt.show()

    # müsste derselbe plot sein:
    ax = sns.relplot(data=Df[Df['not_constant_lin_vel']], x='water_height', y='x_vel_mean_abs', hue='stim_ang_sf', palette='dark', col='stim_ang_vel_int_abs', kind='line')
    ax.set(xlabel='water height [mm]', ylabel='swimming speed magnitude [mm/sec]')
    ax.set_titles('retinal speed magnitude = {col_name} deg/sec')
    ax._legend.set_title('spatial\nfrequency\n[cyc/deg]')
    plt.tight_layout()
    plt.show()



# all velocities
    ax = sns.relplot(data=Df, x='water_height', y='x_vel_mean', hue='stim_ang_vel_int', palette='dark', col='stim_ang_sf', kind='line')
    #ax.set(xlabel='retinal speed [deg/sec]', ylabel='absolute gain')
    #ax.set_titles('{col_var} = {col_name} bzw spatial frequency [cyc/deg]')
    plt.tight_layout()
    plt.show()

# only NOT constant stimuli
    ax = sns.relplot(data=Df[Df['not_constant_lin_vel']], x='water_height', y='x_vel_mean', hue='stim_ang_vel_int', palette='dark', col='stim_ang_sf', kind='line')
    # ax.set(xlabel='retinal speed [deg/sec]', ylabel='absolute gain')
    # ax.set_titles('{col_var} = {col_name} bzw spatial frequency [cyc/deg]')
    plt.tight_layout()
    plt.show()

# only NOT constant stimuli (gepoolt)
    ax = sns.relplot(data=Df[Df['not_constant_lin_vel']], x='water_height', y='x_vel_mean_abs', hue='stim_ang_vel_int_abs', palette='dark', col='stim_ang_sf', kind='line')
    # ax.set(xlabel='retinal speed [deg/sec]', ylabel='absolute gain')
    # ax.set_titles('{col_var} = {col_name} bzw spatial frequency [cyc/deg]')
    plt.tight_layout()
    plt.show()




quit()

wh30 = Df[(np.isclose(Df.water_height, 30))]
wh60 = Df[(np.isclose(Df.water_height, 60))]
wh120 = Df[(np.isclose(Df.water_height, 120))]

# an_velo25 = Df[np.isclose(Df.stim_lin_vel_int, 13.3) | np.isclose(Df.stim_lin_vel_int, 26.6) | np.isclose(Df.stim_lin_vel_int, 53.2) | np.isclose(Df.stim_lin_vel_int, -13.3) | np.isclose(Df.stim_lin_vel_int, -26.6) | np.isclose(Df.stim_lin_vel_int, -53.2)]
an_velo50 = Df[((np.isclose(Df.water_height, 30)) & (np.isclose(Df.stim_lin_vel_int, 28))) | np.isclose(Df.stim_lin_vel_int, 56) | np.isclose(Df.stim_lin_vel_int, 111.9) | ((np.isclose(Df.water_height, 30)) & (np.isclose(Df.stim_lin_vel_int, -28))) | np.isclose(Df.stim_lin_vel_int, -56) | np.isclose(Df.stim_lin_vel_int, -111.9)]
an_velo100 = Df[((np.isclose(Df.water_height, 120)) & (np.isclose(Df.stim_lin_vel_int, 286))) | ((np.isclose(Df.water_height, 60)) & (np.isclose(Df.stim_lin_vel_int, 143))) | np.isclose(Df.stim_lin_vel_int, 71.5) | ((np.isclose(Df.water_height, 120)) & (np.isclose(Df.stim_lin_vel_int, -286))) | ((np.isclose(Df.water_height, 60)) & (np.isclose(Df.stim_lin_vel_int, -143))) | np.isclose(Df.stim_lin_vel_int, -71.5)]

an_velo100.to_excel('an_velo100.xlsx')
an_velo25 = Df[np.isclose(Df.stim_lin_vel, 13.3) | np.isclose(Df.stim_lin_vel, 26.6) | np.isclose(Df.stim_lin_vel, 53.2) | np.isclose(Df.stim_lin_vel, -13.3) | np.isclose(Df.stim_lin_vel, -26.6) | np.isclose(Df.stim_lin_vel, -53.2)]
