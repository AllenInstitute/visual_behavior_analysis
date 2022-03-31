import os
import bz2
import pandas as pd
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt
import visual_behavior.data_access.loading as loading
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis

output_dir = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/sparsity/'

def load_experiment_metrics(oeid):
    '''
        Loads a dictionary of pre-computed metrics
    '''
    filename = output_dir+'experiment_files/'+str(oeid)+'.pbz2'
    if os.path.isfile(filename):
        out_dict = bz2.BZ2File(filename, 'rb')
        out_dict = cPickle.load(out_dict)
        return out_dict
    else:
        print('File not found')
        return None

def compute_experiment(oeid):
    '''
        Computes a dictionary of population sparseness metrics and saves to a pkl file
    '''

    # Load data
    dataset, analysis, sdf = get_response_df(oeid)
    sdf = add_post_event(dataset, sdf)

    # Compute Sparsity Metrics
    all_sparse_dict = compute_image_population_sparseness(sdf,change='all')
    pre_change_sparse_dict = compute_image_population_sparseness(sdf,change='pre_change')
    post_change_sparse_dict = compute_image_population_sparseness(sdf,change='post_change')
    change_sparse_dict = compute_image_population_sparseness(sdf,change='change')
    omission_dict = compute_omission_population_sparseness(sdf)

    # Combine results and save out
    out_dict ={**all_sparse_dict,**pre_change_sparse_dict,**post_change_sparse_dict,**change_sparse_dict,**omission_dict}
    out_dict['num_cells'] = len(sdf['cell_specimen_id'].unique())
    out_dict['ophys_experiment_id'] = oeid

    # Save out a dictionary
    filename = output_dir+'experiment_files/'+str(oeid)+'.pbz2'
    with bz2.BZ2File(filename, 'w') as f:
        cPickle.dump(out_dict,f) 

def get_response_df(oeid):
    dataset = loading.get_ophys_dataset(oeid)
    analysis = ResponseAnalysis(dataset, use_extended_stimulus_presentations=True, use_events=True, filter_events=False)
    sdf = analysis.get_response_df(df_name='stimulus_response_df')
    return dataset, analysis, sdf

def add_post_event(dataset, sdf):
    # compute annotations on stimulus presentations table
    dataset.stimulus_presentations['post_omission'] = dataset.stimulus_presentations['omitted'].shift(1,fill_value=False)
    dataset.stimulus_presentations['post_change'] = dataset.stimulus_presentations['is_change'].shift(1,fill_value=False)

    # many to one merge onto sdf
    sdf = pd.merge(sdf, dataset.stimulus_presentations.reset_index()[['stimulus_presentations_id','post_omission','post_change']], on='stimulus_presentations_id',validate='many_to_one')    
    return sdf

def compute_image_population_sparseness(sdf,change='all'):
    if change == 'all':
        df =sdf.groupby(['image_index','cell_specimen_id'])['mean_response'].mean().unstack('image_index')
    elif change =='pre_change':
        df =sdf.query('pre_change==True').groupby(['image_index','cell_specimen_id'])['mean_response'].mean().unstack('image_index')
    elif change=='post_change':
        df =sdf.query('post_change==True').groupby(['image_index','cell_specimen_id'])['mean_response'].mean().unstack('image_index')
    elif change=='change':
        df =sdf.query('is_change').groupby(['image_index','cell_specimen_id'])['mean_response'].mean().unstack('image_index')           
    else:
        raise Exception('bad change keyword')

    sparsity = {}
    for i in [0,1,2,3,4,5,6,7]:
        sparsity[change+'_sparse_'+str(i)] = sparseness(df[i])
    return sparsity

def compute_omission_population_sparseness(sdf):
    # omission sparseness
    sparsity= {}
    df = sdf.query('omitted').groupby(['cell_specimen_id'])['mean_response'].mean()
    sparsity['omission'] = sparseness(df)

    # pre-omission sparseness
    df = sdf.query('pre_omitted==True').groupby(['image_index','cell_specimen_id'])['mean_response'].mean().unstack('image_index')
    for i in [0,1,2,3,4,5,6,7]:
        sparsity['pre_omission_'+str(i)] = sparseness(df[i])

    # post-omission_sparseness
    df = sdf.query('post_omission==True').groupby(['image_index','cell_specimen_id'])['mean_response'].mean().unstack('image_index')
    for i in [0,1,2,3,4,5,6,7]:
        sparsity['post_omission_'+str(i)] = sparseness(df[i])
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


### Analysis functions
def compute_mouse(mouse_id = 456916):
    experiment_table = loading.get_platform_paper_experiment_table()
    mouse_df = experiment_table.query('mouse_id ==@mouse_id').copy()
    
    for oeid in mouse_df.index.values:
        print(oeid)
        sparse_dict = load_experiment_metrics(oeid)
        for i in sparse_dict.keys():
            mouse_df.loc[oeid, i] = sparse_dict[i]

    return mouse_df

def plot_mouse(mouse_df,metric='all_sparse'):
    plt.figure()
    if metric == "omission":
        plt.plot(mouse_df[metric].values,'k',alpha=1)   
    else:
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








