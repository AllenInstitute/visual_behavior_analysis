"""
Gets called in svm_images_plots_setVars.py

Here, we use svm_df from all projects to set resp_amp_sum_df, a df that includes the mean and stdev of decoding magnitude (aka response amplitude) across all experiments of all sessions

Vars needed here are set in svm_images_plots_setVars_sumMice3_svmdf.py

Created on Fri Oct 29 22:02:05 2021
@author: farzaneh

"""

################################################################################################
### Create a dataframe: resp_amp_sum_df, that includes the mean and stdev of response amplitude across all experiments of all sessions
################################################################################################

svm_df_all = pd.concat(svm_df_allpr)
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



