"""
Gets called in svm_images_plots_setVars.py

Here, we use svm_df from all projects to plot the decoding traces (timeseries), averaged across project codes, for each experience level

Vars needed here are set in svm_images_plots_setVars_sumMice3_svmdf.py

Created on Sat Nov 13 09:26:05 2021
@author: farzaneh

"""

# scientifica
'''
frame_dur
0.032

time_trace
array([-0.48 , -0.448, -0.416, -0.384, -0.352, -0.32 , -0.288, -0.256,
       -0.224, -0.192, -0.16 , -0.128, -0.096, -0.064, -0.032,  0.   ,
        0.032,  0.064,  0.096,  0.128,  0.16 ,  0.192,  0.224,  0.256,
        0.288,  0.32 ,  0.352,  0.384,  0.416,  0.448,  0.48 ,  0.512,
        0.544,  0.576,  0.608,  0.64 ,  0.672,  0.704])
'''

# mesoscope
'''
frame_dur
0.093

time_trace
array([-0.465, -0.372, -0.279, -0.186, -0.093,  0.   ,  0.093,  0.186,
        0.279,  0.372,  0.465,  0.558,  0.651])
'''

# svm_df_all.keys()
# svm_df_all['project_code'].unique()


#########################

time_trace_vb = np.array([-0.48 , -0.448, -0.416, -0.384, -0.352, -0.32 , -0.288, -0.256,
       -0.224, -0.192, -0.16 , -0.128, -0.096, -0.064, -0.032,  0.   ,
        0.032,  0.064,  0.096,  0.128,  0.16 ,  0.192,  0.224,  0.256,
        0.288,  0.32 ,  0.352,  0.384,  0.416,  0.448,  0.48 ,  0.512,
        0.544,  0.576,  0.608,  0.64 ,  0.672,  0.704])

time_trace_ms = np.array([-0.465, -0.372, -0.279, -0.186, -0.093,  0.   ,  0.093,  0.186,
        0.279,  0.372,  0.465,  0.558,  0.651])


colors = utils.get_experience_level_colors() # will always be in order of Familiar, Novel 1, Novel >1

plt.figure(figsize=(16,4))
icre = 0
for cren in cres: # cren = cres[0]    
    icre = icre+1
    plt.subplot(1,3,icre)
    iel = -1
    h = []
    for el in exp_level_all: # el = exp_level_all[0]
        iel = iel+1
        sdf = svm_df_all[svm_df_all['project_code']=='VisualBehaviorMultiscope']
        sdf = sdf[sdf['cre_allPlanes']==cren]
        sdf = sdf[sdf['experience_levels']==el]
        traces_ms = np.vstack(sdf['av_test_data_allPlanes'].values)
#         print(traces_ms.shape)

        sdf = svm_df_all[svm_df_all['project_code']=='VisualBehavior']
        sdf = sdf[sdf['cre_allPlanes']==cren]
        sdf = sdf[sdf['experience_levels']==el]
        traces_vb = np.vstack(sdf['av_test_data_allPlanes'].values)
#         print(traces_vb.shape)

        sdf = svm_df_all[svm_df_all['project_code']=='VisualBehaviorTask1B']
        sdf = sdf[sdf['cre_allPlanes']==cren]
        sdf = sdf[sdf['experience_levels']==el]
        traces_1b = np.vstack(sdf['av_test_data_allPlanes'].values)
#         print(traces_1b.shape)

        ### for omission decoding, set the frame right before omission to nan; so we dont have to deal with the dip we see there; which you have an explanation for (check OneNote, Research, SVM notes); in brief it is because frame -1 represents one of the classes, so when training SVM on that frame, frame -1 is representing both classes (ie the same data represents both classes!), this causes the classifier to get trained on a certail class for a given observation, but in the testing set see a different class for that same observation, hence the lower than chance performance.
        if trial_type=='baseline_vs_nobaseline':
            traces_ms[:, np.argwhere(time_trace_ms==0)-1] = np.nan
            traces_vb[:, np.argwhere(time_trace_vb==0)-1] = np.nan
            traces_1b[:, np.argwhere(time_trace_vb==0)-1] = np.nan
            omit_aligned = 1
        else:
            omit_aligned = 0

        ### upsample mesoscope traces to match scientifica data
        x = time_trace_vb
        xp = time_trace_ms
        traces_ms_interp = []
        for i in range(traces_ms.shape[0]):
            fp = traces_ms[i]
            traces_ms_interp.append(np.interp(x, xp, fp))
        traces_ms_interp = np.array(traces_ms_interp)
        print(traces_ms_interp.shape)

        
        ############################### plots ###############################
        # plot individual project codes
#         plt.plot(xp, np.nanmean(traces_ms, axis=0), color='b')
#         plt.plot(x, np.nanmean(traces_ms_interp, axis=0), color='r')
#         plt.plot(x, np.nanmean(traces_vb, axis=0), color='k')
#         plt.plot(x, np.nanmean(traces_1b, axis=0), color='g')


        # plot the average traces across project codes
        m = np.concatenate((traces_ms_interp, traces_vb, traces_1b))
        print(m.shape)

        hn = plt.plot(x, np.nanmean(m, axis=0), color=colors[iel], label=el)
        h.append(hn)

        
    #### done with all exp levels for a given cre line
    handles, labels = plt.gca().get_legend_handles_labels();
#     lims = [np.min(np.min(lims_v1lm, axis=1)), np.max(np.max(lims_v1lm, axis=1))]
    plot_flashLines_ticks_legend([], handles, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace_vb, xmjn=xmjn, bbox_to_anchor=bb, ylab=ylabel, xlab='Time rel. trial onset (sec)', omit_aligned=omit_aligned)
    plt.xlim(xlim);
    plt.title(cren, fontsize=13, y=1); # np.unique(area)
    # mark time_win: the window over which the response quantification (peak or mean) was computed 
    lims = plt.gca().get_ylim();
    plt.hlines(lims[1], time_win[0], time_win[1], color='gray')
    plt.subplots_adjust(wspace=.8)


#%%
if dosavefig:

    whatSess = f'_timeCourse_experienceLevels'

    fgn = '' #f'{whatSess}'
    if same_num_neuron_all_planes:
        fgn = fgn + '_sameNumNeursAllPlanes'

    if svm_blocks==-1:
        word = 'engaged_disengaged_blocks_'
    elif svm_blocks==-101:
        word = 'only_engaged_'
    elif ~np.isnan(svm_blocks):
        word = 'blocks_'
    else:
        word = ''

    if use_events:
        word = word + 'events'

#     frames_svmn = np.arange(-15,23)
    frames_svmf = -np.argwhere(time_trace_vb==0).squeeze()
    frames_svml = len(time_trace_vb)-np.argwhere(time_trace_vb==0).squeeze()-1

    fgn = f'{fgn}_{word}_frames{frames_svmf}to{frames_svml}'
    fgn = fgn + '_ClassAccur'
    fgn = f'{fgn}_allProjects'

    nam = f'AllCre{whatSess}_aveExpPooled{fgn}_{now}'
    fign = os.path.join(dir0, 'svm', dir_now, nam+fmt)
    print(fign)

    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    
    
    
    
    
    
    
'''
# build a tidy dataframe

list_of_cell_dfs = []
for i in range(traces.shape[0]): # loop over sessions
    cell_df = pd.DataFrame({
        'timestamps': time_trace,
        'decoded_events': traces[i]})

    # append the dataframe for this cell to the list of cell dataframes
    list_of_cell_dfs.append(cell_df)

# concatenate all dataframes in the list
tidy_df = pd.concat(list_of_cell_dfs)    
tidy_df

tidy_df.shape
np.product(np.shape(traces))

neural_data = tidy_df

'''

