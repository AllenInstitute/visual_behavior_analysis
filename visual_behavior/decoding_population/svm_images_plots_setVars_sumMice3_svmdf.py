"""
Gets called in svm_images_plots_setVars.py

Sets svm_df, a proper pandas table, that will be used to make summary plots for experience levels in svm_images_plots_compare_ophys_experience_levels.py

Vars needed here are set in svm_images_plots_setVars_sumMice3_svmdf.py

Created on Fri Oct 29 22:02:05 2021
@author: farzaneh

"""

            
##########################################################################################
##########################################################################################
############# Create svm_df, a proper pandas table #######################################
############# we will use it to make summary plots for experience levels ################
##########################################################################################    
##########################################################################################            

#%% Function to concatenate data in svm_allMice_sessPooled0 in the following way: 
# concatenate data from plane 1 all sessions, then plane 2 all sessions, then plane 3 all sessions,..., then plane 8 all sessions. Therefore, first data from one area (the first 4 planes), then data from another area (the last 4 planes) 

def concatall(df, col):
    # df = svm_allMice_sessPooled0.copy()
    # col = 'av_test_data_allPlanes'    
    # df[col].iloc[0].shape # size: sess   or    planes x sess    or    planes x sess x time  (it must have )
    
    df = df.copy()
    
    if np.ndim(df[col].iloc[0])==1: # data is for all sessions but only 1 plane; we need to replicate it so the size becomes planes x sessions
        for i in range(df.shape[0]): #i=0
            df[col].iloc[i] = [df[col].iloc[i][:] for j in range(8)] # planes x sess

    a = np.concatenate((df[col].iloc[0]))
#     print(a.shape)
    for i in np.arange(1, df.shape[0]): #i=0
        a = np.concatenate((a, np.concatenate((df[col].iloc[i]))))
    print(a.shape)
    a = list(a) # we do this because the matrix that we want to assign to a column of df at once has to be a LIST… it cannot be an array.

    return(a)


################################################
#%% Initiate svm_df out of svm_allMice_sessPooled0. The idea is to create a proper pandas table out of the improper table svm_allMice_sessPooled0!!! Proper because each entry is for one experiment! 
# rows in svm_df are like: plane 1 all sessions, then plane 2 all sessions, etc

svm_df = pd.DataFrame()
cnt = -1
for i in range(svm_allMice_sessPooled0.shape[0]): #i=0 # go through each row of svm_allMice_sessPooled0, ie an ophys stage
    nsess = len(svm_allMice_sessPooled0['experience_levels'].iloc[i])    
    for iplane in range(num_planes): #iplane=0
        for isess in range(nsess): #isess=0
            cnt = cnt + 1
            session_id = svm_allMice_sessPooled0['session_ids'].iloc[i][iplane, isess]
            experiment_id = all_sess0[all_sess0['session_id']==session_id]['experiment_id'].iloc[iplane]
            
            svm_df.at[cnt, 'session_id'] = session_id
            svm_df.at[cnt, 'experiment_id'] = experiment_id
            svm_df.at[cnt, 'session_labs'] = svm_allMice_sessPooled0['session_labs'].iloc[i][0]    
# svm_df.head(300)


################################################
#%% For the rest of the columns we create an entire array/matrix and then add it to the df; as opposed to looping over each value:

#%% Array columns; ie each entry in df is 1 element

# create arrays/matrices for each column of df
cre_allPlanes_allExp = concatall(svm_allMice_sessPooled0, 'cre_allPlanes')
mouse_id_allPlanes_allExp = concatall(svm_allMice_sessPooled0, 'mouse_id_allPlanes')
area_allPlanes_allExp = concatall(svm_allMice_sessPooled0, 'area_allPlanes')
depth_allPlanes_allExp = concatall(svm_allMice_sessPooled0, 'depth_allPlanes')
experience_level_allExp = concatall(svm_allMice_sessPooled0, 'experience_levels')
# create matrix columns
av_test_data_allPlanes_allExp = concatall(svm_allMice_sessPooled0, 'av_test_data_allPlanes')
av_test_shfl_allPlanes_allExp = concatall(svm_allMice_sessPooled0, 'av_test_shfl_allPlanes')
peak_amp_allPlanes_allExp = concatall(svm_allMice_sessPooled0, 'peak_amp_allPlanes')
# session_id_allExp = concatall(svm_allMice_sessPooled0, 'session_ids')
# session_labs_allExp = concatall(svm_allMice_sessPooled0, 'session_labs') # we cant use it with the function concatall because svm_allMice_sessPooled0['session_labs'] is only for 1 session.

# now add columns at once to svm_df 
svm_df['cre_allPlanes'] = cre_allPlanes_allExp
svm_df['mouse_id_allPlanes'] = mouse_id_allPlanes_allExp
svm_df['area_allPlanes'] = area_allPlanes_allExp
svm_df['depth_allPlanes'] = depth_allPlanes_allExp
svm_df['experience_levels'] = experience_level_allExp
svm_df['av_test_data_allPlanes'] = av_test_data_allPlanes_allExp
svm_df['av_test_shfl_allPlanes'] = av_test_shfl_allPlanes_allExp
svm_df['peak_amp_allPlanes_allExp'] = peak_amp_allPlanes_allExp

svm_df #.head(300)

# svm_allMice_sessPooled0.keys()


################################################################################################
### Create a dataframe: resp_amp_sum_df, that includes the mean and stdev of response amplitude across all experiments of all sessions
################################################################################################
exp_level_all = svm_df['experience_levels'].unique()
cresdf = svm_df['cre_allPlanes'].unique()
resp_amp_sum_df = pd.DataFrame()
cnt = -1
for cre in cresdf: # cre = cresdf[0]
    for i in range(len(exp_level_all)):
        cnt = cnt+1
        # svm_df for a given cre and experience level
        thiscre = svm_df[svm_df['cre_allPlanes']==cre]
        thiscre = thiscre[thiscre['experience_levels']==exp_level_all[i]]
        
        depthav = thiscre['depth_allPlanes'].mean()
#         areasu = thiscre['area_allPlanes'].unique()        
        ampall = np.vstack(thiscre['peak_amp_allPlanes_allExp']) # ampall.shape # exp x 4 # pooled_experiments x 4_trTsShCh
        nexp = sum(~np.isnan(ampall[:,1]))

        # testing data
        testav = np.nanmean(ampall[:,1])
        testsd = np.nanstd(ampall[:,1]) / np.sqrt(ampall[:,1].shape[0])
        
        # shuffled
        shflav = np.nanmean(ampall[:,2])  
        shflsd = np.nanstd(ampall[:,2]) / np.sqrt(nexp) # ampall[:,2].shape[0]

        # create the summary df
        resp_amp_sum_df.at[cnt, 'cre'] = cre
        resp_amp_sum_df.at[cnt, 'experience_level'] = exp_level_all[i]
        resp_amp_sum_df.at[cnt, 'depth_av'] = depthav
        resp_amp_sum_df.at[cnt, 'n_experiments'] = nexp
        
        resp_amp_sum_df.at[cnt, 'test_av'] = testav
        resp_amp_sum_df.at[cnt, 'test_sd'] = testsd        
        resp_amp_sum_df.at[cnt, 'shfl_av'] = shflav
        resp_amp_sum_df.at[cnt, 'shfl_sd'] = shflsd
        
resp_amp_sum_df
# [areasu for x in resp_amp_sum_df['cre']]










