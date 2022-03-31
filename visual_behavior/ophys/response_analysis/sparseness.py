import numpy as np
import matplotlib.pyplot as plt
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
import visual_behavior.data_access.loading as loading

def get_response_df(oeid):
    dataset = loading.get_ophys_dataset(oeid)
    analysis = ResponseAnalysis(dataset, use_extended_stimulus_presentations=True, use_events=True, filter_events=False)
    sdf = analysis.get_response_df(df_name='stimulus_response_df')
    return dataset, analysis, sdf

def compute_population_sparseness(sdf,change='all'):
    if change == 'all':
        df =sdf.groupby(['image_index','cell_specimen_id'])['mean_response'].mean().unstack('image_index')
    elif change =='non-change':
        df =sdf.query('pre_change==True').groupby(['image_index','cell_specimen_id'])['mean_response'].mean().unstack('image_index')
    elif change=='change':
        df =sdf.query('is_change').groupby(['image_index','cell_specimen_id'])['mean_response'].mean().unstack('image_index')           
    else:
        raise Exception('bad change keyword')

    sparsity = {}
    for i in [0,1,2,3,4,5,6,7]:
        sparsity[str(i)] = sparseness(df[i])
    return sparsity

def sparseness(r):
    '''
        Taking this definition from "Pseudosparse neural coding in the visual
        system of primates"
    '''
    avg_r = np.mean(r)
    std_r = np.std(r)
    top = np.sum((r-avg_r)**4)
    bottom = (len(r)-1)*(std_r**4)
    s = top/bottom - 3
    return s

def compute_mouse(mouse_id = 456916):
    experiment_table = loading.get_platform_paper_experiment_table()
    mouse_df = experiment_table.query('mouse_id ==@mouse_id').copy()
    
    for oeid in mouse_df.index.values:
        print(oeid)
        dataset, analysis, sdf = get_response_df(oeid)
        sparse_dict = compute_population_sparseness(sdf,change='all')
        for i in sparse_dict.keys():
            mouse_df.loc[oeid, 'all_sparse_'+i] = sparse_dict[i]

        sparse_dict = compute_population_sparseness(sdf,change='non-change')
        for i in sparse_dict.keys():
            mouse_df.loc[oeid, 'non_change_sparse_'+i] = sparse_dict[i]

        sparse_dict = compute_population_sparseness(sdf,change='change')
        for i in sparse_dict.keys():
            mouse_df.loc[oeid, 'change_sparse_'+i] = sparse_dict[i]

    return mouse_df

def plot_mouse(mouse_df,metric='all_sparse'):
    plt.figure()
    for i in range(0,8):
        plt.plot(mouse_df[metric+'_'+str(i)].values,'k',alpha=.25)

    x =[x for x in mouse_df.columns.values if metric in x]
    avg = mouse_df[x[0:-1]].mean(axis=1).values
    sem = mouse_df[x[0:-1]].sem(axis=1).values
    plt.errorbar(range(0,len(mouse_df)),avg,yerr=sem,color='k',alpha=1)

    #plt.plot(mouse_df['sparse_8'].values,'b',alpha=.25)
    plt.xticks(range(0,len(mouse_df)),mouse_df['experience_level'].values, rotation=90)
    plt.ylabel('Sparsity')
    plt.xlabel('Experience Level')
    plt.ylim(bottom=0)
    plt.title(mouse_df.iloc[0]['targeted_structure']+', '+str(mouse_df.iloc[0]['imaging_depth'])+', '+mouse_df.iloc[0]['cre_line'])
    plt.tight_layout()

## TODO
# Do we want to exclude change images
# do we want to only look at change images
# do we want to compare change and post-change images?    









