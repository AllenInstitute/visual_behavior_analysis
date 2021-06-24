#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is called in omissions_traces_peaks_plots_sumMice.py

Created on Tue Jun 22 2021
@author: farzaneh
"""

####################################################################################
####################################################################################
### ANOVA on response quantifications (image and omission evoked response amplitudes)
####################################################################################
####################################################################################    
print(f'\n\n==============================\n========== {cre} ==========\n==============================\n\n')

aa_all = [pa_all[:, cre_all[0,:]==cre], paf_all[:, cre_all[0,:]==cre]] # 2 x 8 x sessions
aan = 'omission', 'image'

for of in range(2): # omission, image
    print(f'\n========== {aan[of]} ==========')
    aa = aa_all[of]
    ####
    cols = ['area', 'depth', 'value']
    a_data = pd.DataFrame([], columns = cols)
    i_at = -1
    for i_depth in range(4):
        inds_now = inds_v1
        a_now = aa[inds_now[i_depth],:][~np.isnan(aa[inds_now[i_depth]])]
        for i_a in range(len(a_now)):
            i_at = i_at+1
            a_data.at[i_at, 'area'] = 'V1'
            a_data.at[i_at, 'depth'] = int(depth_ave[i_depth])
            a_data.at[i_at, 'value'] = a_now[i_a]

    for i_depth in range(4):
        inds_now = inds_lm
        a_now = aa[inds_now[i_depth],:][~np.isnan(aa[inds_now[i_depth]])]
        for i_a in range(len(a_now)):
            i_at = i_at+1
            a_data.at[i_at, 'area'] = 'LM'
            a_data.at[i_at, 'depth'] = int(depth_ave[i_depth])
            a_data.at[i_at, 'value'] = a_now[i_a]
    a_data.at[:,'value'] = a_data['value'].values.astype(float)
#             a_data

    #######################################################
    #######################################################
    # Do anova and tukey for each area: compare depths
    #######################################################
    #######################################################            
    for ars in ['V1', 'LM']:
        print(ars)
        a_data_now = a_data[a_data['area'].values==ars]
#         a_data_now = a_data

        ### ANOVA
        # https://reneshbedre.github.io/blog/anova.html        
        # https://pythonhealthcare.org/2018/04/13/55-statistics-multi-comparison-with-tukeys-test-and-the-holm-bonferroni-method/
        # https://help.xlstat.com/s/article/how-to-interpret-contradictory-results-between-anova-and-multiple-pairwise-comparisons?language=es
        # https://pythonhealthcare.org/2018/04/13/55-statistics-multi-comparison-with-tukeys-test-and-the-holm-bonferroni-method/
        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        ######## ANOVA: compare depths for a given area
        # Ordinary Least Squares (OLS) model
        # 2-way
        # C(Genotype):C(years) represent interaction term        
#         model = ols('value ~ C(area) + C(depth) + C(area):C(depth)', data=a_data).fit()
        # 1-way
        model = ols('value ~ C(depth)', data=a_data_now).fit() 
#         anova_table = sm.stats.anova_lm(model, typ=3)
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)        
        print('\n')
        # scipy anova: same result as above
#                 a = aa[inds_v1,:]
#                 fvalue, pvalue = st.f_oneway(a[0][~np.isnan(a[0])], a[1][~np.isnan(a[1])], a[2][~np.isnan(a[2])], a[3][~np.isnan(a[3])])
#                 print(fvalue, pvalue)


        ### TUKEY HSD: compare depths for a given area        
        from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)

#         v = a_data['value']
#         f = a_data['depth']
        v = a_data_now['value'] #a_data['value']
        f = a_data_now['depth'] #a_data['depth']

        MultiComp = MultiComparison(v, f)
        print(MultiComp.tukeyhsd().summary()) # Show all pair-wise comparisons



    #######################################################
    #######################################################
    # Do ttest for each depth: compare the 2 areas
    #######################################################
    #######################################################            

    import scipy.stats as stats

    depun = np.unique(a_data['depth'].values)
    for idps in range(len(depun)): # idps = 0
        ittest = ittest+1
#         print(idps)
        a_data_now = a_data[a_data['depth'].values==depun[idps]]
#         a_data_now = a_data
        a = a_data_now[a_data_now['area']=='V1']['value'].values
        b = a_data_now[a_data_now['area']=='LM']['value'].values
#                 a.shape, b.shape

        _, p = stats.ttest_ind(a, b, nan_policy='omit')
#         print(p)

        ab = np.nanmean(a), np.nanmean(b) # ave_resp_V1_LM

        p_areas_each_depth.at[ittest, ['cre', 'event', 'depth', 'p-value', 'ave_resp_V1_LM']] = cre, aan[of], depun[idps], p, ab

### show significant p values (comparing the 2 areas for each depth)
print(p_areas_each_depth[p_areas_each_depth['p-value'].values <= .05])


