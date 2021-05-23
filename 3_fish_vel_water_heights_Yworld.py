from analyse_df import *


# THREE figures in ONE skript:

if __name__ == '__main__':
# load Data
    base_folder = './data/'
    #Df = pd.read_hdf('Summary_final.h5', 'all')
    Df = pd.read_hdf('Summary_final.h5', 'by_subparticle')



# FISH SWIMMING VELOCITY at different WATER HEIGHTS

# all velocities
    ax = sns.relplot(data=Df, x='water_height', y='x_vel_mean', hue='stim_ang_vel_int', palette='dark', col='stim_ang_sf', kind='line')
    #ax.set(xlabel='retinal speed [deg/sec]', ylabel='absolute gain')
    #ax.set_titles('{col_var} = {col_name} bzw spatial frequency [cyc/deg]')
    plt.tight_layout()
    plt.show()

# only NOT constant stimuli
    ax = sns.relplot(data=Df[Df['not_constant_lin_vel']], x='water_height', y='x_vel_mean', hue='stim_ang_vel_int', palette='dark', col='stim_ang_sf',
                     kind='line')
    # ax.set(xlabel='retinal speed [deg/sec]', ylabel='absolute gain')
    # ax.set_titles('{col_var} = {col_name} bzw spatial frequency [cyc/deg]')
    plt.tight_layout()
    plt.show()



# TUNING SWIMMING SPEED

    ax = sns.relplot(data=Df, x='stim_ang_vel_int', y='x_vel_mean', hue='water_height', palette='dark', col='stim_ang_sf', kind='line')
    #ax.set(xlabel='retinal speed [deg/sec]', ylabel='absolute gain')
    #ax.set_titles('{col_var} = {col_name} bzw spatial frequency [cyc/deg]')
    plt.tight_layout()
    plt.show()

# beide Achsen gepoolt
    ax = sns.relplot(data=Df, x='stim_ang_vel_int_abs', y='x_vel_mean_abs', hue='water_height', palette='dark', col='stim_ang_sf', kind='line')
    # ax.set(xlabel='retinal speed [deg/sec]', ylabel='absolute gain')
    # ax.set_titles('{col_var} = {col_name} bzw spatial frequency [cyc/deg]')
    plt.tight_layout()
    plt.show()




# Y WORLD FISH POSITION at different RETINAL SPEEDS

    ax = sns.relplot(data=Df, x='stim_ang_vel_int', y='y_world_mean', hue='water_height', palette='dark', col='stim_ang_sf', kind='line')
    #ax.set(xlabel='retinal speed [deg/sec]', ylabel='absolute gain')
    #ax.set_titles('{col_var} = {col_name} bzw spatial frequency [cyc/deg]')
    plt.tight_layout()
    plt.show()



quit()

