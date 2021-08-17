import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


okr_sf = pd.DataFrame()
okr_sf['spatial_frequency'] = [0.019969773282010737,0.039879184493490906,0.059977046362476694,0.07963782730064915,0.11925937727794297,0.15971959818832365,
                            0.019969773282010717,0.039879184493490906,0.059977046362476694,0.07963782730064924,0.11925937727794297,0.15903493558658885,
                            0.019969773282010717,0.03987918449349094,0.059977046362476694,0.07963782730064915,0.11925937727794297,0.15903493558658868,
                            0.019969773282010717,0.039879184493490906,0.05971994553657502,0.07963782730064907,0.11925937727794297,0.15971959818832349]
okr_sf['gain_at_sf'] = [0.3537675131834289,0.6640827850634842,0.8476276754730661,0.8543266059341131,0.7128390011640483,0.34011354781700287,
                     0.32698656913895296,0.576346633419312,0.74145614710311,0.7184726762278729,0.42398493675880744,0.15479024575053768,
                     0.26230309729523366,0.4515448097768274,0.427335754210932,0.4410059454176735,0.1332865493844508,0.05141751827683936,
                     0.22944772748859613,0.3046215921934059,0.31190123473417775,0.2643761185749101,0.041246263829013585,0.027607408473246674]
okr_sf['ang_vel'] = [3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 22.5, 22.5, 22.5, 22.5, 22.5, 22.5]


okr_tf = pd.DataFrame()
okr_tf['temporal_frequency'] = [0.07412115813979138,0.14907691765899728,0.2237231794331364,0.29784371439234314,0.44698095321806736,0.5990411375113217,1.5007231363938391,
                                0.14907691765899728,0.2978437143923427,0.44698095321806736,0.5990411375113219,0.8989948947149232,1.1968361049583192,2.9983243565613824,
                                0.2978437143923428,0.5990411375113217,0.8989948947149232,1.1968361049583185,1.796119633182435,2.407148538257962,5.990411375113223,
                                0.44698095321806736,0.8930320633112735,0.8989948947149232,1.349142437998628,1.8081124372411574,2.713476167769257,3.6124635041003326]
okr_tf['gain_at_tf'] = [0.35579907876749894,0.6641851428253386,0.8584894051000305,0.8584894051000296,0.7168391013988568,0.33893817128820414,0.21296746087746024,
                           0.329664580918035,0.5741642455935717,0.7370040586690033,0.7168391013988572,0.4231613793450885,0.15372588004246893,0.07472982683491881,
                           0.26222536512233874,0.4535513939259253,0.42610641477311567,0.4411419039499153,0.13289051237631017,0.0513858297294321,0.04861239162480231,
                           0.2330611470182414,0.3075755485502486,0.3097161524226465,0.3140421571126837,0.2677385107523492,0.04144478390225936,0.027527009715487407]
okr_tf['ang_vel'] = [3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 22.5, 22.5, 22.5, 22.5, 22.5, 22.5, 22.5]


okr_sf_2 = pd.DataFrame()
okr_sf_2['spatial_frequency'] = [0.019969773282010717,0.03987918449349094,0.059977046362476694,0.07963782730064915,0.11925937727794297,0.15903493558658868,
                            0.019969773282010717,0.039879184493490906,0.05971994553657502,0.07963782730064907,0.11925937727794297,0.15971959818832349]
okr_sf_2['gain_at_sf'] = [0.26230309729523366,0.4515448097768274,0.427335754210932,0.4410059454176735,0.1332865493844508,0.05141751827683936,
                     0.22944772748859613,0.3046215921934059,0.31190123473417775,0.2643761185749101,0.041246263829013585,0.027607408473246674]
okr_sf_2['ang_vel'] = [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 22.5, 22.5, 22.5, 22.5, 22.5, 22.5]


# and load my data:
base_folder = './data/'
Df = pd.read_hdf('Summary_final.h5', 'by_subparticle')

wh30 = Df[(np.isclose(Df.water_height, 30))]
wh60 = Df[(np.isclose(Df.water_height, 60))].copy()
wh120 = Df[(np.isclose(Df.water_height, 120))]

vel25 = Df[np.isclose(Df.stim_ang_vel, 26.047) | np.isclose(Df.stim_ang_vel, 27.2) | np.isclose(Df.stim_ang_vel, 29.8)]
an_velo25 = Df[np.isclose(Df.stim_lin_vel, 13.3) | np.isclose(Df.stim_lin_vel, 26.6) | np.isclose(Df.stim_lin_vel, 53.2) | np.isclose(Df.stim_lin_vel, -13.3) | np.isclose(Df.stim_lin_vel, -26.6) | np.isclose(Df.stim_lin_vel, -53.2)]
lin_vel28 = Df[np.isclose(Df.stim_lin_vel_abs, 28)]
sf4 = Df[np.isclose(Df.stim_ang_sf, 0.04)]
sf2 = Df[np.isclose(Df.stim_ang_sf, 0.02)]

# #col='stim_ang_sf', style ='stim_ang_vel_abs',
# ax = sns.relplot(data=an_velo25, x='stim_ang_sf', y='x_lin_gain', hue='water_height', palette='crest_r', kind='line', height=6, aspect=1.2)
# ax.set(xlabel='spatial frequency [cyc/deg]', ylabel='gain')
# ax.set_titles('OKR (upper legend) & OMR (bottom legends) at linear velocity = 28 mm/sec')
# ax._legend.set_title('OMR water height [mm]')
# plt.xlim([0,0.17])
# plt.ylim([0,0.6])
# plt.tight_layout()
# plt.show()
#
# #okr at sf
# ax = sns.relplot(data=okr_sf_2, x='spatial_frequency', y='gain_at_sf', hue='ang_vel', style='ang_vel', markers=True, palette='dark:salmon', kind='line', height=6, aspect=1.3)
# ax.set(xlabel='', ylabel='')
# ax.set_titles('OKR (middle legend) & OMR (right) at water height = 30 mm')
# ax._legend.remove()
# plt.legend(loc='upper right')
# plt.xlim([0,0.17])
# plt.ylim([0,0.6])
# plt.tight_layout()
# plt.show()

sns.set(font_scale=1.3)
sns.set_style('ticks')

import IPython
IPython.embed()

#gain at sf
ax = sns.relplot(data=okr_sf, x='ang_vel', y='gain_at_sf', hue='spatial_frequency', palette='dark', kind='line')
#ax.set(xlabel='spatial frequency [cyc/deg]', ylabel='OKR gain')
#ax._legend.set_title('angular\nvelocity\n[deg/sec]')
plt.tight_layout()
plt.show()

ax = sns.relplot(data=okr_tf, x='ang_vel', y='gain_at_tf', hue='temporal_frequency',  markers=True, palette='dark:salmon', kind='line')
plt.tight_layout()
plt.show()

# gain at tf
ax = sns.relplot(data=okr_tf, x='temporal_frequency', y='gain_at_tf', hue='ang_vel', style='ang_vel', markers=True, palette='dark:salmon', kind='line', height=6, aspect=1.3)
ax.set(xlabel='', ylabel='')
ax.set_titles('OKR (middle legend) & OMR (right) at water height = 30 mm')
ax._legend.remove()
plt.legend(loc='upper right')
# ax._legend.texts[0].set_text("hi")
# ax._legend.set_title("New title")
# ax._legend._legend_box.sep = -5
# ax._legend.set_title('angular\nvelocity\n[deg/sec]')
plt.xlim([0,6.5])
plt.ylim([0,0.9])
plt.tight_layout()
plt.show()


# bring together with my OMR gain figure:
ax = sns.relplot(data=an_velo25, x='stim_ang_tf_abs', y='x_lin_gain', hue='water_height', style='stim_ang_vel_abs', col='stim_lin_vel_abs', palette='crest_r', kind='line', height=6, aspect=1.2)
ax.set(xlabel='temporal frequency [cyc/sec]', ylabel='gain')
ax.set_titles('OKR (upper legend) & OMR (bottom legends) at linear velocity = 28 mm/sec')
ax._legend.set_title('OMR parameters:')
plt.xlim([0,6.5])
plt.ylim([0,0.9])
plt.tight_layout()
plt.show()
# absolute GAIN GEGEN TF (eigl in gain_tuning_func.py)

# each water height:
#wh30
ax = sns.relplot(data=wh30, x='stim_ang_tf_abs', y='x_lin_gain', hue='stim_ang_vel_abs', col='water_height', palette='crest_r', kind='line', height=6, aspect=1.2)
ax.set(xlabel='temporal frequency [cyc/sec]', ylabel='gain')
ax.set_titles('water height for OMR = 30 mm')
#ax._legend.remove()
#plt.legend(loc='upper right')
ax._legend.set_title('OMR retinal speed [deg/sec]')
plt.xlim([0,6.5])
plt.ylim([0,0.9])
new_labels = ['29.8','58.5','110.1','141.5','160.2']
for t, l in zip(ax._legend.texts, new_labels): t.set_text(l)
plt.tight_layout()
plt.show()


# wh60
ax = sns.relplot(data=wh60, x='stim_ang_tf_abs', y='x_lin_gain', hue='stim_ang_vel_abs', palette='crest_r', col='water_height',kind='line', height=6, aspect=1.2)
ax.set(xlabel='temporal frequency [cyc/sec]', ylabel='gain')
ax.set_titles('water height for OMR = 60 mm')
#ax._legend.remove()
#plt.legend(loc='upper right')
ax._legend.set_title('OMR retinal speed [deg/sec]')
plt.xlim([0,6.5])
plt.ylim([0,0.9])
new_labels = ['27.2', '28.6', '54.0', '104.9', '138.0']
for t, l in zip(ax._legend.texts, new_labels): t.set_text(l)
plt.tight_layout()
plt.show()
# ist halt nicht gebinnt

#wh120
ax = sns.relplot(data=wh120, x='stim_ang_tf_abs', y='x_lin_gain', hue='stim_ang_vel_abs', palette='dark', col='water_height',kind='line')
ax.set(xlabel='temporal frequency (magnitude) [cyc/sec]', ylabel='absolute gain')
ax.set_titles('water height = 120 mm')
ax._legend.set_title('angular\nvelocity\n[deg/sec]')
plt.tight_layout()
plt.show()



# GAIN AT SF
# okr gain at sf
ax = sns.relplot(data=okr_sf, x='spatial_frequency', y='gain_at_sf', hue='ang_vel', style='ang_vel', markers=True, palette='dark:salmon', kind='line', height=6, aspect=1.3)
ax.set(xlabel='', ylabel='')
ax.set_titles('OKR (middle legend) & OMR (right) at water height = 30 mm')
ax._legend.remove()
plt.legend(loc='upper right')
plt.xlim([0,0.17])
plt.ylim([0,0.9])
plt.tight_layout()
plt.show()

# wh30
ax = sns.relplot(data=wh30, x='stim_ang_sf', y='x_lin_gain', hue='stim_ang_vel_abs', palette='crest_r', col='water_height',kind='line', height=6, aspect=1.2)
ax.set(xlabel='spatial frequency [cyc/deg]', ylabel='gain')
ax.set_titles('OKR (upper legend) & OMR (bottom legend) at water height = 30 mm')
ax._legend.set_title('angular\nvelocity\n[deg/sec]')
plt.xlim([0,0.17])
plt.ylim([0,0.9])
plt.tight_layout()
plt.show()

# wh60
ax = sns.relplot(data=wh60, x='stim_ang_sf', y='x_lin_gain', hue='stim_ang_vel_abs', palette='crest_r', col='water_height',kind='line', height=6, aspect=1.2)
ax.set(xlabel='spatial frequency [cyc/deg]', ylabel='gain')
ax.set_titles('OKR (upper legend) & OMR (bottom legend) at water height = 60 mm')
ax._legend.set_title('angular\nvelocity\n[deg/sec]')
plt.xlim([0,0.17])
plt.ylim([0,0.9])
plt.tight_layout()
plt.show()


import IPython
IPython.embed()


exit()






#okr['temporal_frequency_in_cyc_per_sec'] = [0.07412115813979138, 0.149076917658997286, 0.2237231794331364, 0.29784371439234314, 0.44698095321806736, 0.5990411375113217, 1.5007231363938391,


#sf
# v375_x = [0.019969773282010737,0.039879184493490906,0.059977046362476694,0.07963782730064915,0.11925937727794297,0.15971959818832365]
# v375_y = [0.3537675131834289,0.6640827850634842,0.8476276754730661,0.8543266059341131,0.7128390011640483,0.34011354781700287]
# v75_x = [0.019969773282010717,0.039879184493490906,0.059977046362476694,0.07963782730064924,0.11925937727794297,0.15903493558658885]
# v75_y = [0.32698656913895296,0.576346633419312,0.74145614710311,0.7184726762278729,0.42398493675880744,0.15479024575053768]
# v15_x = [0.019969773282010717,0.03987918449349094,0.059977046362476694,0.07963782730064915,0.11925937727794297,0.15903493558658868]
# v15_y = [0.26230309729523366,0.4515448097768274,0.427335754210932,0.4410059454176735,0.1332865493844508,0.05141751827683936]
# v225_x = [0.019969773282010717,0.039879184493490906,0.05971994553657502,0.07963782730064907,0.11925937727794297,0.15971959818832349]
# v225_y = [0.22944772748859613,0.3046215921934059,0.31190123473417775,0.2643761185749101,0.041246263829013585,0.027607408473246674]
# okr['v375_x'] = v375_x
# okr['v375_y'] = v375_y
# okr['v75_x'] = v75_x
# okr['v75_y'] = v75_y
# okr['v15_x'] = v15_x
# okr['v15_y'] = v15_y
# okr['v225_x'] = v225_x
# okr['v225_y'] = v225_y
# plt.plot(v375_x, v375_y, label='3.75 cyc/deg')
# plt.plot(v75_x,v75_y, label='7.5 cyc/deg')
# plt.plot(v15_x,v15_y, label='15.0 cyc/deg')
# plt.plot(v225_x,v225_y, label='22.5 cyc/deg')
# plt.legend()
# plt.show()
#SF
v375_x = [0.019969773282010737, 0.039879184493490906, 0.059977046362476694,0.07963782730064915,0.11925937727794297,0.15971959818832365]
v375_y = [0.3537675131834289,0.6640827850634842,0.8476276754730661,0.8543266059341131,0.7128390011640483,0.34011354781700287,]
v75_x = [0.019969773282010717,0.039879184493490906,0.059977046362476694,0.07963782730064924,0.11925937727794297,0.15903493558658885]
v75_y = [0.32698656913895296,0.576346633419312,0.74145614710311,0.7184726762278729,0.42398493675880744,0.15479024575053768]
v15_x = [0.019969773282010717,0.03987918449349094,0.059977046362476694,0.07963782730064915,0.11925937727794297,0.15903493558658868]
v15_y = [0.26230309729523366,0.4515448097768274,0.427335754210932,0.4410059454176735,0.1332865493844508,0.05141751827683936]
v225_x = [0.019969773282010717,0.039879184493490906,0.05971994553657502,0.07963782730064907,0.11925937727794297,0.15971959818832349]
v225_y = [0.22944772748859613,0.3046215921934059,0.31190123473417775,0.2643761185749101,0.041246263829013585,0.027607408473246674]

