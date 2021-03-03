import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn import preprocessing, decomposition, svm
from sklearn.neural_network import MLPClassifier

annot_path = 'annotations'

def parse_annotation_file(path):
    annotations = dict()
    with open(path) as f:
        for line in f.readlines():
            parts = line.split(':')
            phase_id = int(parts[0])
            try:
                fish_ids = [int(s) for s in parts[1].strip(' ').strip('\n').split(',') if bool != '']
            except Exception as exc:
                print(f'Exception in PHASE {phase_id}')
                import traceback
                print(traceback.print_exc())
                quit()
            annotations[phase_id] = fish_ids

    return annotations


def add_annotations(base_path, Df):
    for filename in os.listdir(base_path):
        path = os.path.join(base_path, filename)
        if os.path.isdir(path):
            add_annotations(path, Df)
        else:
                name = filename.split('.')[0]
                if (name == Df.folder).sum() > 0:
                    foundmsg = 'Dataset found in Df'
                else:
                    foundmsg = 'Dataset NOT found in Df'
                print(f'> Annotate {path} // {foundmsg}')

                for phase_id, fish_ids in parse_annotation_file(path).items():
                    print(f'> Phase {phase_id}: {fish_ids}')
                    Df.loc[(Df.folder == name) & (Df.phase_id == phase_id), 'annotated'] = 1
                    Df.loc[(Df.folder == name) & (Df.phase_id == phase_id) & Df.particle.isin(fish_ids), 'is_fish_annot'] = 1


def select_training_subset(Df):
    new_df = pd.DataFrame()
    base_path = os.path.join(annot_path, 'train')
    for filename in os.listdir(base_path):
        name = filename.split('.')[0]
        path = os.path.join(base_path, filename)
        phase_ids = [pid for pid in parse_annotation_file(path).keys()]
        new_df = new_df.append(Df[(Df.folder == name) & (Df.phase_id.isin(phase_ids))])

    return new_df


def calc_minmaxnorm(df,column):
    df[f'norm_{column}'] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())


def calc_metrics(df,grps,column):
    df[f'mean_{column}'] = grps[column].mean()
    df[f'median_{column}'] = grps[column].median()
    df[f'std_{column}'] = grps[column].std()
    df[f'cv_{column}'] = df[f'std_{column}'] / df[f'mean_{column}']


def get_unique(df,grps,column):
    df[column] = grps[column].apply(lambda vals: vals.unique()[0])


def calc_clf_perform(df):
    tp = (df.is_fish_predict == 1) & (df.is_fish_annot == 1)
    fp = (df.is_fish_predict == 1) & (df.is_fish_annot == 0)
    tn = (df.is_fish_predict == 0) & (df.is_fish_annot == 0)
    fn = (df.is_fish_predict == 0) & (df.is_fish_annot == 1)

    tp_rate = tp.sum() / (tp.sum() + fn.sum())
    fn_rate = fn.sum() / (tp.sum() + fn.sum())
    tn_rate = tn.sum() / (tn.sum() + fp.sum())
    fp_rate = fp.sum() / (tn.sum() + fp.sum())

    return tp_rate, fn_rate, tn_rate, fp_rate


if __name__ == '__main__':

    print('Load summary')
    Df = pd.read_hdf('data/Summary.h5','all')

    print('Format summary Df')
    # Abbrev. folder paths
    Df['folder'] = Df['folder'].apply(lambda s: s.split('/')[-1])
    # Calculate approx. phase durations from data directly
    grps = Df.groupby(['folder','phase_id'])
    Df = Df.join(grps['time'].apply(lambda times: times.max() - times.min()),on=['folder','phase_id'],rsuffix='_diff')
    # Normalize
    calc_minmaxnorm(Df,'mass')
    calc_minmaxnorm(Df,'size')
    calc_minmaxnorm(Df,'signal')
    calc_minmaxnorm(Df,'ecc')
    calc_minmaxnorm(Df,'x')
    calc_minmaxnorm(Df,'y')

    # Grouped by particle
    print('Group summary by particle')
    grps = Df.groupby(['folder','phase_id','particle'])
    Dfg = pd.DataFrame()
    # Count frames per particle
    Dfg['fcount'] = grps.apply(lambda df: df.shape[0])
    # X
    calc_metrics(Dfg,grps,'x')
    calc_metrics(Dfg,grps,'norm_x')
    # Y
    calc_metrics(Dfg,grps,'y')
    calc_metrics(Dfg,grps,'norm_y')
    # Mass
    calc_metrics(Dfg,grps,'mass')
    calc_metrics(Dfg,grps,'norm_mass')
    # Size
    calc_metrics(Dfg,grps,'size')
    calc_metrics(Dfg,grps,'norm_size')
    # Signal
    calc_metrics(Dfg,grps,'signal')
    calc_metrics(Dfg,grps,'norm_signal')
    # Ecc
    calc_metrics(Dfg,grps,'ecc')
    calc_metrics(Dfg,grps,'norm_ecc')
    # Uniques
    get_unique(Dfg,grps,'u_lin_velocity')
    get_unique(Dfg,grps,'u_spat_period')

    # Norm framecount
    Dfg['norm_fcount'] = Dfg.fcount.apply(lambda df: df / Dfg.fcount.max())
    # Particle duration in s
    Dfg['duration'] = grps.time.apply(lambda times: times.max() - times.min())

    # Filter single-frame particles and reset index for readability
    Dfg = Dfg[Dfg.fcount > 1].reset_index()

    print('Select classifier data subset')
    Dfg['annotated'] = 0
    Dfg['is_fish_annot'] = 0
    Dfg['is_fish_predict'] = 0
    add_annotations(annot_path, Dfg)

    Df_annot = Dfg[Dfg.annotated == 1]
    Df_train = select_training_subset(Df_annot).copy()
    Df_test = Df_annot[~Df_annot.index.isin(Df_train.index)].copy()

    print(f'Using {Df_train.shape[0]} out of {Df_annot.shape[0]} for fitting')


    print('Run Classifier')
    #features = ['mean_norm_mass', 'cv_norm_mass', 'mean_norm_size', 'cv_norm_size', 'mean_norm_signal', 'cv_norm_signal', 'mean_ecc', 'cv_ecc', 'norm_fcount']
    features = ['mean_norm_mass','cv_norm_mass','mean_norm_size','cv_norm_size','mean_norm_signal','cv_norm_signal',
                'mean_ecc','cv_ecc','norm_fcount', 'cv_norm_x', 'cv_norm_y']
    X = Df_train[features].to_numpy()
    y = Df_train['is_fish_annot'].to_numpy()
    # MLP
    # Good:
    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, max_iter=10000, random_state=1, hidden_layer_sizes=(100,5))
    clf = svm.SVC(kernel='rbf', probability=True)
    clf.fit(X, y)
    Df_test['is_fish_predict_proba'] = clf.predict_proba(Df_test[features].to_numpy())[:,1]
    Df_test['is_fish_predict'] = Df_test['is_fish_predict_proba']  > 0.75
    #Df_test['is_fish_predict'] = clf.predict(Df_test[features].to_numpy())


    # Pre-filter
    # Df_test = Df_test[(Df_test.mean_norm_mass > Df_test.mean_norm_mass.quantile(0.10))
    #                 & (Df_test.mean_norm_signal > Df_test.mean_norm_signal.quantile(0.10))
    #                 & (Df_test.norm_fcount > Df_test.norm_fcount.quantile(0.10))].copy()
    #
    # score = clf.score(Df_test[features].to_numpy(),Df_test['is_fish_annot'].to_numpy())
    # tp_rate,fn_rate,tn_rate,fp_rate = calc_clf_perform(Df_test)


    statistics = pd.DataFrame()

    for layer_size in np.linspace(30, 200, 20, dtype=np.uint):
        for layer_num in np.arange(1, 6, 1, dtype=np.uint):


            clf = MLPClassifier(solver='sgd', alpha=1e-5, max_iter=10000, random_state=1, hidden_layer_sizes=(layer_size, layer_num))
            #clf = MLPClassifier(solver='sgd', alpha=1e-5, max_iter=10000, random_state=1)
            clf.fit(X, y)
            Df_test['is_fish_predict'] = clf.predict(Df_test[features].to_numpy())

            # Select annotated test data
            #Df_test = Dfg[(Dfg.annotated == 1) & (~Dfg.folder.isin(Df_train.folder.unique()))]

            # Check if it sums up

            # Test
            tp_rate, fn_rate, tn_rate, fp_rate = calc_clf_perform(Df_test)
            score = clf.score(Df_test[features].to_numpy(), Df_test['is_fish_annot'].to_numpy())
            #
            # tp = (Df_test.is_fish_predict == 1) & (Df_test.is_fish_annot == 1)
            # fp = (Df_test.is_fish_predict == 1) & (Df_test.is_fish_annot == 0)
            # tn = (Df_test.is_fish_predict == 0) & (Df_test.is_fish_annot == 0)
            # fn = (Df_test.is_fish_predict == 0) & (Df_test.is_fish_annot == 1)
            #
            # tp_rate = tp.sum() / (tp.sum() + fn.sum())
            # fn_rate = fn.sum() / (tp.sum() + fn.sum())
            # tn_rate = tn.sum() / (tn.sum() + fp.sum())
            # fp_rate = fp.sum() / (tn.sum() + fp.sum())

            data = dict(layer_size=layer_size,
                        layer_num=layer_num,
                        tp_rate=tp_rate,
                        fn_rate=fn_rate,
                        tn_rate=tn_rate,
                        fp_rate=fp_rate,
                        score=score,
                        clf=clf)

            statistics = statistics.append(pd.Series(data), ignore_index=True)
            print(data)

            # print('True positive {:.1f}%'.format(100 * tp_rate))
            # print('False positive {:.1f}%'.format(100 * fp_rate))
            # print('True negative {:.1f}%'.format(100 * tn_rate))
            # print('False negative {:.1f}%'.format(100 * fn_rate))


    # Filter unsuccessful models
    rates = ['tp_rate', 'fn_rate', 'tn_rate', 'fp_rate']
    for datatype in rates:
        statistics.loc[np.isclose(statistics[datatype], 0.) | np.isclose(statistics[datatype], 1.), datatype] = np.nan

    # Plot heatmap
    statistics['tp_fp_rate'] = statistics['tp_rate'] - statistics['fp_rate']
    for datatype in ['tp_fp_rate', 'tp_rate', 'fp_rate', 'tn_rate', 'fn_rate']:
        title = f'SGD {datatype}'
        plt.figure(num=title)
        sns.heatmap(statistics.pivot('layer_size', 'layer_num', datatype), annot=True, cmap='seismic', center=0.)
        plt.suptitle(title)
    plt.show()


    # sns.jointplot(data=Dfg, x='meanmass', y='meanx', hue='folder')

    # Mass*dur may be useful:
    # for folder in Dfg1.folder.unique():
    #     ax = sns.jointplot(data=Dfg1[(Dfg1.folder == folder)], x='mass*dur', y='xvel')
    #     ax.fig.suptitle(folder)
    #
    #
    # for folder in Dfg1.folder.unique():
    #     df = Dfg1[(Dfg1.folder == folder)]
    #     ax = sns.jointplot(data=df[df['mass*dur'] > 200], x='mass*dur', y='gain', hue='stimtf_abs', legend='full')
    #     ax.fig.suptitle(folder)
    #     #ax.ax_joint.set_xscale('log')

    #
    # # 3d
    # for folder in Dfg.folder.unique():
    #     df = Dfg[(Dfg.folder == folder)]
    #
    #     #df = df[df['mass*dur'] > 200]
    #
    #     # Mean
    #     # savestr = f'figures/{folder}_meanmass_meansize_meanecc.png'
    #     # x, y, z = df.normmass_mean.values, df.normsize_mean.values, df.ecc_mean.values
    #     # Std
    #     savestr = f'figures/{folder}_stdmass_stdsize_stdecc.png'
    #     x, y, z = df.normmass_std.values, df.normsize_std.values, df.ecc_std.values
    #
    #
    #     fig = plt.figure(figsize=(10,10))
    #     ax = fig.add_subplot(projection='3d')
    #     scat = ax.scatter(x,y,z, c=df.is_fish.values)
    #     cbar = plt.colorbar(scat)
    #
    #     fig.suptitle(folder)

        #plt.savefig(savestr)

    # pairplot
    #features = ['normmass_mean', 'normmass_std', 'normsize_mean', 'normsize_std', 'normsig_mean', 'normsig_std', 'ecc_mean', 'ecc_std', 'normfcount']
    use_df = Df_annot
    for folder in use_df.folder.unique():
        df = use_df[(use_df.folder == folder)]
        #df = Dfg[(Dfg.folder == folder)]
        #df = Df_train[(Df_train.folder == folder)]

        if df.shape[0] == 0:
            continue

        sns.pairplot(data=df, vars=features, hue='is_fish_annot', plot_kws={"s": 3.})
        plt.suptitle(folder)
        plt.tight_layout()
        plt.savefig(f'figures/pairplots/{folder}_annot.pdf', format='pdf')
        plt.savefig(f'figures/pairplots/{folder}_annot.png', format='png')




    # PCA on recordings:
    for folder in Dfg.folder.unique():
        df = Dfg[(Dfg.folder == folder)]

        data = df.loc[:,['meanmass', 'stdmass', 'meansize', 'stdsize', 'meanecc', 'stdecc', 'fcount']].values
        data = preprocessing.StandardScaler().fit_transform(data)

        pca = decomposition.PCA(n_components=3)
        components = pca.fit_transform(data)


        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection='3d')
        scat = ax.scatter(components[:,0], components[:,1], components[:,2], c=df.is_fish.values)

        fig.suptitle(folder)


    # PCA on all data:

    data = Dfg1.loc[:,['meanmass', 'meansize', 'meanecc', 'duration']].values
    data = preprocessing.StandardScaler().fit_transform(data)

    pca = decomposition.PCA(n_components=3)
    components = pca.fit_transform(data)


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    scat = ax.scatter(components[:,0], components[:,1], components[:,2])
