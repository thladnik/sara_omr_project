import os
import pandas as pd
from sklearn import  svm

annot_path = None

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


def run_classification(filepath):
    Df = pd.read_hdf(filepath, 'all')

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
    get_unique(Dfg,grps,'folder')
    get_unique(Dfg,grps,'phase_id')
    get_unique(Dfg,grps,'particle')

    # Norm framecount
    Dfg['norm_fcount'] = Dfg.fcount.apply(lambda df: df / Dfg.fcount.max())
    # Particle duration in s
    Dfg['duration'] = grps.time.apply(lambda times: times.max() - times.min())

    # Filter single-frame particles and reset index for readability
    Dfg = Dfg[Dfg.fcount > 1]#.reset_index()

    print('Select classifier data subset')
    Dfg['annotated'] = 0
    Dfg['is_fish_annot'] = 0
    Dfg['is_fish_predict'] = 0

    print('Add annotations from file')
    add_annotations(annot_path, Dfg)

    Df_annot = Dfg[Dfg.annotated == 1]
    Df_train = select_training_subset(Df_annot).copy()
    Df_test = Df_annot[~Df_annot.index.isin(Df_train.index)].copy()
    print(f'Using {Df_train.shape[0]} out of {Df_annot.shape[0]} for fitting')


    features = ['mean_norm_mass','cv_norm_mass','mean_norm_size',
                'cv_norm_size','mean_norm_signal','cv_norm_signal',
                'mean_ecc','cv_ecc','norm_fcount', 'cv_norm_x', 'cv_norm_y']
    print('Run Classifier on features', features)

    # Train classifier
    clf = svm.SVC(kernel='rbf', probability=True)
    clf.fit(Df_train[features].to_numpy(), Df_train['is_fish_annot'].to_numpy())
    # Test
    Df_test['is_fish_predict_proba'] = clf.predict_proba(Df_test[features].to_numpy())[:,1]
    Df_test['is_fish_predict'] = Df_test['is_fish_predict_proba']  > probability_threshold
    tp_rate,fn_rate,tn_rate,fp_rate = calc_clf_perform(Df_test)
    print('Classifier stats for probability threshold {:.2f}: '
          'Hit {:.1f}% / False alarm {:.1f}%'.format(probability_threshold, 100 * tp_rate, 100 * fp_rate))

    print('Apply classifier to whole dataset')
    Dfg['is_fish_predict_proba'] = clf.predict_proba(Dfg[features].to_numpy())[:,1]
    Dfg['is_fish_predict'] = Dfg['is_fish_predict_proba'] > probability_threshold

    print('Write results to summary')
    Df['is_fish'] = False
    folder = None
    folder_select = []
    phase_id = None
    phase_select = []
    for idx, row in Dfg.iterrows():

        if row.folder != folder:
            folder = row.folder
            folder_select = Df.folder == folder
            # print(f'Folder {folder}')

        if row.phase_id != phase_id:
            phase_id = row.phase_id
            phase_select = Df.phase_id == phase_id
            print(f'> {folder}/{phase_id}')

        Df.loc[folder_select
               & phase_select
               & (Df.particle == row.particle),'is_fish'] = row.is_fish_predict

    print('Save to file')
    Df.to_hdf(f'{filepath}.clf', 'all')
    print('Classification completed successfully.')


def load_summary(filepath) -> pd.DataFrame:

    print('Load summary')
    Df_sum = pd.read_hdf(filepath, 'all')

    if os.path.exists(f'{filepath}.clf'):

        print('Found existing classification, checking for differences')
        Df = pd.read_hdf(f'{filepath}.clf', 'all')
        run_clf = False

        if Df.shape[0] != Df_sum.shape[0]:
            s = input('Number of summary rows don\'t match. Re-run classification? [Y/n]')
            if s.lower() == 'y':
                run_clf = True

        if run_clf:
            run_classification(filepath)
        else:
            return Df
    else:
        run_classification(filepath)

    return pd.read_hdf(f'{filepath}.clf', 'all')


def filter_fish(Df):
    return Df[Df.is_fish]


if __name__ == '__main__':

    probability_threshold = 0.75

    annot_path = 'annotations'

    Df = filter_fish(load_summary('//172.25.250.112/arrenberg_data/shared/Sara_Widera/Ausgewertet/Summary.h5'))

    import IPython
    IPython.embed()