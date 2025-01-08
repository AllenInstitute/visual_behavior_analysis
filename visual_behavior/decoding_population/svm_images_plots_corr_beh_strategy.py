"""
This script is called in "svm_images_plots_setVars_blocks.py"
correlate classification accuracy with behavioral strategy.

it will be executed if trial_type =='changes_vs_nochanges' or trial_type =='hits_vs_misses'

Created on Mon Apr 26 17:48:05 2021
@author: farzaneh

"""

import matplotlib.pyplot as plt

'''
pool_all_stages = True #True
fit_reg = False # lmplots: add the fitted line to the catter plot or not
superimpose_all_cre = False
plot_testing_shufl = 0 # if 0 correlate testing data with strategy dropout index; if 1, correlated shuffled data with strategy dropout index
'''

#%% correlate classification accuracy with behavioral strategy

# loading.get_behavior_model_summary_table()
directory = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/behavior_model_output/'
model_table = pd.read_csv(directory+'_summary_table.csv')

# strategy dropout
model_table_allsess = model_table[model_table['ophys_session_id'].isin(all_sess0['session_id'])]    
cn = 'strategy_dropout_index'  # 'visual_only_dropout_index' # 'timing_only_dropout_index'  # 'strategy_dropout_index'
sdi = model_table_allsess[cn].values
sdi0 = sdi +0

if project_codes == ['VisualBehavior']:
    print(f'{len(model_table_allsess)} / {len(all_sess0)} sessions of all_sess have behavioral strategy data')
else:    
    print(f'{len(model_table_allsess)} / {len(all_sess0)/num_planes} sessions of all_sess have behavioral strategy data')


# set allsess for sessions in model table, and get some vars out of it
all_sess0_modeltable = all_sess0[all_sess0['session_id'].isin(model_table['ophys_session_id'])]        
peak_amp_all = np.vstack(all_sess0_modeltable['peak_amp_trainTestShflChance'].values) # 744 x 4        
bl_all = np.vstack(all_sess0_modeltable['bl_pre0'].values) # 744 x 4
stages = np.vstack(all_sess0_modeltable['stage'].values).flatten() # 744 

stages_each = np.reshape(stages, (num_planes, len(model_table_allsess)), order='F') # 8 x 93
#     np.unique(stages_each) # 'OPHYS_1_images_A', 'OPHYS_3_images_A', 'OPHYS_4_images_B', 'OPHYS_6_images_A', 'OPHYS_6_images_B'

if trial_type =='changes_vs_nochanges':
    stagesall = [1,2,3,4,5,6]
elif trial_type =='hits_vs_misses':
    stagesall = [1,3,4,6]


if pool_all_stages:
    stagesall = [0]

##### loop over each ophys stage
for stage_2_analyze in stagesall: # stage_2_analyze = stagesall[0]

    if stage_2_analyze==0: # pool all stages
        stinds = np.full((len(stages)), 6)
    else:
        stinds = np.array([stages[istage].find(str(stage_2_analyze)) for istage in range(len(stages))])
    stages2an = (stinds != -1)
    nsnow = int(sum(stages2an) / num_planes)

    if stage_2_analyze==0: # pool all stages
        ophys_title = f'all stages'
        print(f'{nsnow} sessions; pooled ophys stages')
    else:
        ophys_title = f'ophys{stage_2_analyze}'
        print(f'{nsnow} sessions are ophys {stage_2_analyze}')

    sessionsNow = np.reshape(stages2an, (num_planes, len(model_table_allsess)), order='F') # 8 x 93
    sessionsNow = sessionsNow[0,:] # get values from only 1 plane

    # get sdi for sessions in the current stage
    sdi = sdi0[sessionsNow]

    # testing data class accuracy
    peak_amp_test = peak_amp_all[stages2an, 1]
    bl_now = bl_all[stages2an, 1]
    # compute average of the 8 planes
    a = np.reshape(peak_amp_test, (num_planes, nsnow), order='F') # 8 x 93
    b = np.reshape(bl_now, (num_planes, nsnow), order='F')
    testing_accur_ave_planes = np.nanmean(a, axis=0)
    bl_avp = np.nanmean(b, axis=0)

    if baseline_subtract: # subtract the baseline (CA average during baseline, ie before time 0) from the evoked CA (classification accuracy)
        testing_accur_ave_planes = testing_accur_ave_planes - bl_avp



    # shuffled data class accuracy
    peak_amp_shfl = peak_amp_all[stages2an, 2]
    bl_now = bl_all[stages2an, 2]
    # compute average of the 8 planes
    a = np.reshape(peak_amp_shfl, (num_planes, nsnow), order='F')
    b = np.reshape(bl_now, (num_planes, nsnow), order='F')
    shfl_accur_ave_planes = np.nanmean(a, axis=0)
    bl_avp = np.nanmean(b, axis=0)

    yls = 'Decoder accuracy'
    if baseline_subtract: # subtract the baseline (CA average during baseline, ie before time 0) from the evoked CA (classification accuracy)
        shfl_accur_ave_planes = shfl_accur_ave_planes - bl_avp
        yls = f'{yls} (rel. baseline)'


    ##############################
    # what to plot?
    if plot_testing_shufl==0:
        topn = testing_accur_ave_planes # testing data class accuracy
        fignamets = 'testing_'
    elif plot_testing_shufl==1:
        topn = shfl_accur_ave_planes
        fignamets = 'shfl_'
#     topn = testing_accur_ave_planes - shfl_accur_ave_planes # testing-shuffled data class accuracy


    ##############################
    ### get r2 values (corr coeffs) between decoder accuracy and behavioral strategy across all sessions
    ### all cre lines
    # compute, for each session, corrcoef between strategy dropout index and average testing_class_accuracy across the 8 planes
    c, p = ma.corrcoef(ma.masked_invalid(sdi), ma.masked_invalid(topn))
    pallcre = p

    #### cre line for each session
    cre_mt = all_sess0_modeltable['cre'].values
    cre_mt = np.reshape(cre_mt, (num_planes, len(model_table_allsess)), order='F')[0,:] # len sessions
    cre_mt = cre_mt[sessionsNow]

    # cre names
    a = np.unique(cre_mt.astype(str))
    cren = a[a!='nan']
    print(cren)
    # cren = ['Slc17a7-IRES2-Cre', 'Sst-IRES-Cre', 'Vip-IRES-Cre']

    #### compute r2 for each cre line separately
    c_allcre = []
    for icremt in range(len(cren)):
        sdin = sdi[cre_mt == cren[icremt]]
        tan = topn[cre_mt == cren[icremt]]

        c, p = ma.corrcoef(ma.masked_invalid(sdin), ma.masked_invalid(tan))    
        c_allcre.append(p[0])
    print(cren, c_allcre)


    #################################
    ############## plots ############
    #################################


    #################################
    # lmplot: Plot each cre line in a separate subplot; color code by cre line

    if project_codes == ['VisualBehavior']:
        vb_ms = ''
    else:
        vb_ms = '(average of 8 planes)'
    
    df = pd.DataFrame([], columns=[f'{yls} {vb_ms}', f'{cn}', 'cre'])
    df[f'{yls} {vb_ms}'] = topn #testing_accur_ave_planes
    df[f'{cn}'] = sdi
    df['cre'] = cre_mt

    g = sns.lmplot(f'{cn}', f'{yls} {vb_ms}', data=df, hue='cre', size=2.5, scatter_kws={"s": 20}, col='cre', fit_reg=fit_reg)

    #### add cc to the titles
    # lmplot plots cre lines in the same order as cre_mt; lets get that array of cre lines
    cre_mt[[type(cre_mt[i])==float for i in range(len(cre_mt))]] = 'remove' # change nans in cre_mt to 'remove'
    a = cre_mt[cre_mt!='remove'] # remove nans from cre_mt
    indexes = np.unique(a, return_index=True)[1]
    lms = [a[index] for index in sorted(indexes)]

    # now lets match this array with cren; we need it for the plot below
    lm_order_in_cren = [np.argwhere(np.in1d(cren, lms[i])).flatten()[0] for i in range(len(lms))]


    ii = -1
    for icremt in lm_order_in_cren: #[1,0,2]:
        ii = ii+1
        g.axes[0][ii].set_title(f'{cren[icremt][:3]}, r2 = {c_allcre[icremt]:.2f}')
    plt.suptitle(f'{ophys_title}, r2={pallcre[0]:.2f}', y=1.1);


    
    ####
    if dosavefig:

#         snn = [str(sn) for sn in whichStages]
#         snn = '_'.join(snn)

        whatSess = ophys_title #f'_summaryStages_{snn}'

        fgn = '' #f'{whatSess}'
        if same_num_neuron_all_planes:
            fgn = fgn + '_sameNumNeursAllPlanes'

        if svm_blocks==-1:
            if iblock==0:
                word = 'disengaged_'
            elif iblock==1:
                word = 'engaged_'
        elif svm_blocks==-101:
            word = 'only_engaged_'
        elif ~np.isnan(svm_blocks):
            word = 'blocks_'
        else:
            word = ''

        if use_events:
            word = word + 'events'

        fgn = f'{fgn}_{word}_frames{frames_svm[0]}to{frames_svm[-1]}'   
            
        fgn = fgn + '_ClassAccur'
#         if project_codes == ['VisualBehavior']:
        fgn = f'{fgn}_{project_codes[0]}'
        
        nam = f'{whatSess}_{fignamets}beh_strategy_corr_{trial_type}{fgn}_{now}' # {crenow[:3]}

        fign = os.path.join(dir0, 'svm', dir_now, nam+fmt)
        print(fign)

        
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    

    
        
    
    

    #################################
    # Plot all cre lines: scatter plot
    
    if superimpose_all_cre:
    #     plt.figure(figsize=(3,3))
    #     plt.plot(sdi, topn, '.')
        g = sns.lmplot(f'{cn}', f'Decoder accuracy {vb_ms}', data=df, hue='cre', size=2.5, scatter_kws={"s": 20}, fit_reg=False)            
        plt.xlabel(f'{cn}')
        plt.ylabel('average decoder accuracy of 8 planes\n rel. baseline')
        plt.title(f'{ophys_title}, r2={pallcre[0]:.2f}');



    #################################
    # Plot each cre line in a separate figure
    # compute corrcoeff for each cre line
#             c_allcre = []
    '''
    plt.figure(figsize=(3,9))
    for icremt in range(len(cren)):
        sdin = sdi[cre_mt == cren[icremt]]
        tan = topn[cre_mt == cren[icremt]]

#                 c, p = ma.corrcoef(ma.masked_invalid(sdin), ma.masked_invalid(tan))    
#                 c_allcre.append(p[0])

        plt.subplot(3,1,icremt+1)
        plt.plot(sdin, tan, '.')
        if icremt==2:
            plt.xlabel(f'{cn}')
        if icremt==1:
            plt.ylabel('Average decoder accuracy of 8 planes\n rel. baseline')
        plt.title(f'{ophys_title}, {cren[icremt][:3]}, r2={c_allcre[icremt]:.2f}')
        makeNicePlots(plt.gca())

    plt.subplots_adjust(hspace=0.5)
    '''



    #################################
#     NOTE: parts below need work

    '''
    # color code by area

    # repeat strategy index for each plane
    a = np.tile(sdi, (num_planes,1)) # 8 x num_sess
    sdi_rep = np.reshape(a, (num_planes*a.shape[1]), order='F')
#     a.shape

    cre_rep = all_sess0_modeltable['cre'].values
    area_rep = all_sess0_modeltable['area'].values
#     peak_amp_test.shape


    ##### plot all cre lines
    df = pd.DataFrame([], columns=['Decoder accuracy (each plane)', f'{cn}', 'cre', 'area'])
    df['Decoder accuracy (each plane)'] = peak_amp_test
    df[f'{cn}'] = sdi_rep
    df['cre'] = cre_rep
    df['area'] = area_rep

    # all cre lines
    dfn = df
    # a single cre line
#     dfn = df[df['cre']==cren[2]]    
    sns.lmplot(f'{cn}', 'Decoder accuracy (each plane)', data=dfn, hue='area', fit_reg=False)


    ##### plot each cre line separately
    plt.figure(figsize=(3,6))
    for icremt in range(len(cren)):
        sdin = sdi_rep[cre_rep == cren[icremt]]
        tan = peak_amp_test[cre_rep == cren[icremt]]
        aan = area_rep[cre_rep == cren[icremt]]

        plt.subplot(3,1,icremt+1)
        plt.plot(sdin[aan=='VISl'], tan[aan=='VISl'], 'b.', label='VISl')
        plt.plot(sdin[aan=='VISp'], tan[aan=='VISp'], 'r.', label='VISp')
        if icremt==0:
            plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False)
        if icremt==2:
            plt.xlabel(f'{cn}')
        if icremt==1:
            plt.ylabel('Decoder accuracy of each plane\n change vs no change')

    plt.subplots_adjust(hspace=0.5)


    #################################
    # color code by depth

    '''
    
    
