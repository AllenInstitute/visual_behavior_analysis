"""
Gets called in svm_images_plots_setVars.py

Here, we use svm_df from all projects to set resp_amp_sum_df for area and depth comparison. The dfs include the mean and stdev of decoding magnitude (aka response amplitude) across all experiments of all sessions with the same binned depth or the same brain area.

We also compute ttest and p-value between areas/ depth for each experience level and cre line.

svm_images_plots_compare_experience_levels_area_depth.py will use the vars set here to make plots.

Vars needed here are set in svm_images_plots_setVars_sumMice3_svmdf.py

Created on Fri Oct 29 22:02:05 2021
@author: farzaneh

"""

import scipy.stats as st

################################################################################################
### Create a dataframe: resp_amp_sum_df, that includes the mean and sterr of response amplitude across all experiments of all sessions
################################################################################################

# svm_df_all = pd.concat(svm_df_allpr) # concatenate data from all project codes

####### get svm_df only for mesoscope data; because we make area/depth plots only for ms data.

svm_df_ms = svm_df_allpr[2] # run the code below for a single project code
print(len(svm_df_ms))

exp_level_all = svm_df_ms['experience_levels'].unique()

cres = ['Slc17a7', 'Sst', 'Vip']

################################################################################################
### The following will be used for making area & depth comparison plots:
### Create dataframes: resp_amp_sum_df_depth & resp_amp_sum_df_area.
### they include the mean and sterr of response amplitude across all experiments of all sessions with the same binned depth (resp_amp_sum_df_depth), or the same area (resp_amp_sum_df_area).
################################################################################################

p_depth = np.full((len(cres), len(exp_level_all)), np.nan) # p value for superficial vs deep layer comparison of class accuracy
p_area = np.full((len(cres), len(exp_level_all)), np.nan) # p value for V1 vs LM area comparison of class accuracy
p_depth_area_df = pd.DataFrame()
resp_amp_sum_df_depth = pd.DataFrame()
resp_amp_sum_df_area = pd.DataFrame()
icre = -1
cnt = -1
cntd = -1
cnta = -1
for cre in cres: # cre = cres[0]
    icre = icre+1    
    for i in range(len(exp_level_all)): # i=0
        cnt = cnt+1
        
        # svm_df for a given cre and experience level
        thiscre = svm_df_ms[svm_df_ms['cre_allPlanes']==cre]
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
        dff = pd.DataFrame()
        
        dff['area'] = area_all
        
        # use 4 binned depth
#         dff['binned_depth'] = binned_depth_all
        
        # create 2 binned depths: <200 and >200um; we will just label them as 100 and 300um depth.
        bd2 = binned_depth_all
        bd2[binned_depth_all<200] = 100
        bd2[binned_depth_all>200] = 300
        dff['binned_depth'] = bd2
        
        dff['test'] = test_all
        dff['shfl'] = shfl_all
        
        ##############################
        # compute ttest and p-value between areas/ depth for each experience level and cre line
        
        # depth
        a_amp = np.vstack(dff[dff['binned_depth']==100]['test']).flatten()
        b_amp = np.vstack(dff[dff['binned_depth']==300]['test']).flatten()        
        
        print(a_amp.shape, b_amp.shape)
        print(sum(~np.isnan(a_amp)), sum(~np.isnan(b_amp)))
        
        _, p = st.ttest_ind(a_amp, b_amp, nan_policy='omit') #, axis=1, equal_var=equal_var)
        p_depth[icre, i] = p
        

        # area
        a_amp = np.vstack(dff[dff['area']=='VISl']['test']).flatten()
        b_amp = np.vstack(dff[dff['area']=='VISp']['test']).flatten()        
        
        print(a_amp.shape, b_amp.shape)
        print(sum(~np.isnan(a_amp)), sum(~np.isnan(b_amp)))
        
        _, p = st.ttest_ind(a_amp, b_amp, nan_policy='omit') #, axis=1, equal_var=equal_var)
        p_area[icre, i] = p
        
        
        # set a df to keep p values
        p_depth_area_df.at[cnt, 'cre'] = cre
        p_depth_area_df.at[cnt, 'exp_level'] = exp_level_all[i]  
        
        p_depth_area_df.at[cnt, 'p_depth'] = p_depth[icre, i]      
        p_depth_area_df.at[cnt, 'p_area'] = p_area[icre, i]  
        
#         df_new = pd.concat([p_depth_area_df, pd.DataFrame(p_depth)], axis=1)
        
        ##############################
        # compute averages across experiments with the same depth/are
        
        # binned depth
        # average classification accuracies across experiments with the same binned depth; do this for both testing and shuffle data
        ave_test_shfl_binned_depth = dff.groupby('binned_depth').mean()
        # ste (note that count function does not count NaN vals)
        ne = dff.groupby('binned_depth').count()['test'].values # number of experiments for each area
        nee = np.vstack([ne, ne]).T # set it in a format that can be used to divide std by.        
        ste_test_shfl_binned_depth = dff.groupby('binned_depth').std() / np.sqrt(nee)
        
        # area
        # average classification accuracies across experiments with the same area; do this for both testing and shuffle data
        ave_test_shfl_area = dff.groupby('area').mean()
        # ste
        ne = dff.groupby('area').count()['test'].values # number of experiments for each area
        nee = np.vstack([ne, ne]).T # set it in a format that can be used to divide std by.
        ste_test_shfl_area = dff[['area', 'test', 'shfl']].groupby('area').std() / np.sqrt(nee)
        
        
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
        

#%% Add sig columns to p_depth_area_df

p_depth_area_df['sig_depth'] = p_depth_area_df['p_depth']<=sigval
p_depth_area_df.loc[p_depth_area_df['sig_depth']==False, 'sig_depth'] = np.nan

p_depth_area_df['sig_area'] = p_depth_area_df['p_area']<=sigval
p_depth_area_df.loc[p_depth_area_df['sig_area']==False, 'sig_area'] = np.nan


