"""
Gets called in svm_images_plots_setVars.py

Here, we use svm_df from all projects to set resp_amp_sum_df, a df that includes the mean and stdev of decoding magnitude (aka response amplitude) across all experiments of all sessions

Vars needed here are set in svm_images_plots_setVars_sumMice3_svmdf.py

Created on Fri Oct 29 22:02:05 2021
@author: farzaneh

"""

################################################################################################
### Create a dataframe: resp_amp_sum_df, that includes the mean and sterr of response amplitude across all experiments of all sessions
################################################################################################

svm_df_all = pd.concat(svm_df_allpr) # concatenate data from all project codes
# svm_df_all = svm_df_allpr[2] # run the code below for a single project code
print(len(svm_df_all))

exp_level_all = svm_df_all['experience_levels'].unique()
cresdf = svm_df_all['cre_allPlanes'].unique()
resp_amp_sum_df = pd.DataFrame()

cnt = -1
for cre in cresdf: # cre = cresdf[0]
    for i in range(len(exp_level_all)): # i=0
        cnt = cnt+1
        
        # svm_df for a given cre and experience level
        thiscre = svm_df_all[svm_df_all['cre_allPlanes']==cre]
        thiscre = thiscre[thiscre['experience_levels']==exp_level_all[i]]
        print(len(thiscre))
        
        depthav = thiscre['depth_allPlanes'].mean()
#         areasu = thiscre['area_allPlanes'].unique()        
        ampall = np.vstack(thiscre['peak_amp_allPlanes_allExp']) # ampall.shape # exp x 4 # pooled_experiments x 4_trTsShCh
        nexp = sum(~np.isnan(ampall[:,1]))

        # testing data
        testav = np.nanmean(ampall[:,1])
        testsd = np.nanstd(ampall[:,1]) / np.sqrt(nexp) #ampall[:,1].shape[0])
        
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







################################################################################################
### The following will be used for making area & depth comparison plots:
### Create dataframes: resp_amp_sum_df_depth & resp_amp_sum_df_area.
### they include the mean and sterr of response amplitude across all experiments of all sessions with the same binned depth (resp_amp_sum_df_depth), or the same area (resp_amp_sum_df_area).
################################################################################################

resp_amp_sum_df_depth = pd.DataFrame()
resp_amp_sum_df_area = pd.DataFrame()
# cnt = -1
cntd = -1
cnta = -1
for cre in cresdf: # cre = cresdf[0]
    for i in range(len(exp_level_all)): # i=0
#         cnt = cnt+1
        
        # svm_df for a given cre and experience level
        thiscre = svm_df_all[svm_df_all['cre_allPlanes']==cre]
        thiscre = thiscre[thiscre['experience_levels']==exp_level_all[i]]
        print(len(thiscre))
        
        
        # area
        area_all = thiscre['area_allPlanes'].values

        # binned depth
        binned_depth_all = thiscre['binned_depth'].values

        # testing data
        test_all = np.array([thiscre[ 'peak_amp_allPlanes_allExp'].iloc[ii][1] for ii in range(len(thiscre))])

        # shuffled data
        shfl_all = np.array([thiscre[ 'peak_amp_allPlanes_allExp'].iloc[ii][2] for ii in range(len(thiscre))])

        # number of experiments
        nexp = sum(~np.isnan(test_all))

        
        ##############################
        # create a dataframe for the above vars
        df = pd.DataFrame()
        
        df['area'] = area_all
        df['binned_depth'] = binned_depth_all
        df['test'] = test_all
        df['shfl'] = shfl_all

        # binned depth
        # average classification accuracies across experiments with the same binned depth; do this for both testing and shuffle data
        ave_test_shfl_binned_depth = df.groupby('binned_depth').mean()
        # ste (note that count function does not count NaN vals)
        ne = df.groupby('binned_depth').count()['test'].values # number of experiments for each area
        nee = np.vstack([ne, ne]).T # set it in a format that can be used to divide std by.        
        ste_test_shfl_binned_depth = df.groupby('binned_depth').std() / np.sqrt(nee)
        
        # area
        # average classification accuracies across experiments with the same area; do this for both testing and shuffle data
        ave_test_shfl_area = df.groupby('area').mean()
        # ste
        ne = df.groupby('area').count()['test'].values # number of experiments for each area
        nee = np.vstack([ne, ne]).T # set it in a format that can be used to divide std by.
        ste_test_shfl_area = df[['area', 'test', 'shfl']].groupby('area').std() / np.sqrt(nee)
        
        
        ##############################
        # set a df like resp_amp_sum_df, but each row will be a given cre, exp level, binned_depth
        # set another df just like this one, but for areas

        # binned depth
        # cntd = -1
        for j in range(ave_test_shfl_binned_depth.shape[0]): # loop through the 4 binned depths
            cntd = cntd + 1
            
            resp_amp_sum_df_depth.at[cntd, 'cre'] = cre
            resp_amp_sum_df_depth.at[cntd, 'experience_level'] = exp_level_all[i]
            
            resp_amp_sum_df_depth.at[cntd, 'binned_depth'] = ave_test_shfl_binned_depth.index[j]
            resp_amp_sum_df_depth.at[cntd, 'test_av'] = ave_test_shfl_binned_depth['test'].iloc[j]
            resp_amp_sum_df_depth.at[cntd, 'shfl_av'] = ave_test_shfl_binned_depth['shfl'].iloc[j]
            resp_amp_sum_df_depth.at[cntd, 'test_sd'] = ste_test_shfl_binned_depth['test'].iloc[j]
            resp_amp_sum_df_depth.at[cntd, 'shfl_sd'] = ste_test_shfl_binned_depth['shfl'].iloc[j]
        

        # area
        # cnta = -1
        for j in range(ave_test_shfl_area.shape[0]): # loop through the 2 areas
            cnta = cnta + 1
            
            resp_amp_sum_df_area.at[cnta, 'cre'] = cre
            resp_amp_sum_df_area.at[cnta, 'experience_level'] = exp_level_all[i]            
            
            resp_amp_sum_df_area.at[cnta, 'area'] = ave_test_shfl_area.index[j]
            resp_amp_sum_df_area.at[cnta, 'test_av'] = ave_test_shfl_area['test'].iloc[j]
            resp_amp_sum_df_area.at[cnta, 'shfl_av'] = ave_test_shfl_area['shfl'].iloc[j]
            resp_amp_sum_df_area.at[cnta, 'test_sd'] = ste_test_shfl_area['test'].iloc[j]
            resp_amp_sum_df_area.at[cnta, 'shfl_sd'] = ste_test_shfl_area['shfl'].iloc[j]
        
        
        