from analyse_df import *

if __name__ == '__main__':

# load data
    base_folder = './data/'
    Df = pd.read_hdf('Summary_final.h5', 'by_subparticle')




# create various Df in Excel to use these filtered / sorted Dataframes for statistical analysis


# only Spatial Frequency 0.02

    Df_02_sf = Df[(np.isclose(Df.stim_ang_sf, 0.02))]

    Df_02_sf.to_hdf('Df_02_sf.h5', 'by_subparticle')
    Df_02_sf.to_excel('Df_02_sf.xlsx', sheet_name='by_subparticle')


# only Water Height 60mm




    Df.to_hdf('Summary_final.h5', 'by_subparticle')
    Df.to_excel('Summary_final.xlsx', sheet_name='by_subparticle')


