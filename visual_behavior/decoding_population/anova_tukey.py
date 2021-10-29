"""

Created on Wed Apr 14 19:10:05 2021
@author: farzaneh

"""

####################################################################################
####################################################################################
### ANOVA on response quantifications (image and omission evoked response amplitudes)
####################################################################################
####################################################################################    

import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.formula.api import ols

def do_anova_tukey(summary_vars_all, crenow, stagenow, inds_v1, inds_lm, inds_pooled):
    
#     print(f'_______________')
#     print(f'\n{crenow}, ophys {stagenow}\n')

    tc = summary_vars_all[summary_vars_all['cre'] == crenow]
    tc = tc[tc['stage'] == stagenow]

    pa_all = tc['resp_amp'].values[0] 
    aa = pa_all[:,:,1] # testing data
#     aa.shape # 8 x sessions

    depth_ave = tc['depth_ave'].values[0]

    ###############
    cols = ['area', 'depth', 'value', 'idepth']
    
    inds_now = inds_v1    
    a_data = pd.DataFrame([], columns = cols)
    i_at = -1
    for i_depth in range(len(inds_now)): # i_depth=0
        a_now = aa[inds_now[i_depth],:][~np.isnan(aa[inds_now[i_depth]])]
#         print(a_now.shape) # number of valid sessions per depth
        for i_a in range(len(a_now)):
            i_at = i_at+1
            a_data.at[i_at, 'area'] = 'V1'
            a_data.at[i_at, 'depth'] = int(depth_ave[i_depth])
            a_data.at[i_at, 'idepth'] = i_depth
            a_data.at[i_at, 'value'] = a_now[i_a]

            
    if ~np.isnan(inds_lm[0]):
        inds_now = inds_lm        
        for i_depth in range(len(inds_now)):
            a_now = aa[inds_now[i_depth],:][~np.isnan(aa[inds_now[i_depth]])]
    #             print(a_now.shape) # number of valid sessions per depth    
            for i_a in range(len(a_now)):
                i_at = i_at+1
                a_data.at[i_at, 'area'] = 'LM'
                a_data.at[i_at, 'depth'] = int(depth_ave[i_depth])
                a_data.at[i_at, 'idepth'] = i_depth
                a_data.at[i_at, 'value'] = a_now[i_a]
        a_data.at[:,'value'] = a_data['value'].values.astype(float)

        
#     if project_codes == ['VisualBehaviorMultiscope']:        
    inds_now = inds_pooled        
    for i_depth in range(4):
        a_now = aa[inds_now[i_depth],:].flatten()[~np.isnan(aa[inds_now[i_depth]].flatten())]
#         print(a_now.shape) # number of valid sessions per depth    
        for i_a in range(len(a_now)):
            i_at = i_at+1
            a_data.at[i_at, 'area'] = 'V1-LM'
            a_data.at[i_at, 'depth'] = int(depth_ave[i_depth])
            a_data.at[i_at, 'idepth'] = i_depth
            a_data.at[i_at, 'value'] = a_now[i_a]
    a_data.at[:,'value'] = a_data['value'].values.astype(float)
    a_data

    
    
    ########### Do anova and tukey for each area
    tukey_all = []
#     if project_codes == ['VisualBehaviorMultiscope']:
    for ars in ['V1', 'LM', 'V1-LM']: # ars = 'V1'
#         print(ars)
        a_data_now = a_data[a_data['area'].values==ars]
    #     a_data_now = a_data

        ### ANOVA
        # https://reneshbedre.github.io/blog/anova.html        
        # https://pythonhealthcare.org/2018/04/13/55-statistics-multi-comparison-with-tukeys-test-and-the-holm-bonferroni-method/
        # https://help.xlstat.com/s/article/how-to-interpret-contradictory-results-between-anova-and-multiple-pairwise-comparisons?language=es
        # https://pythonhealthcare.org/2018/04/13/55-statistics-multi-comparison-with-tukeys-test-and-the-holm-bonferroni-method/

        # Ordinary Least Squares (OLS) model
        # 2-way
        # C(Genotype):C(years) represent interaction term        
    #         model = ols('value ~ C(area) + C(depth) + C(area):C(depth)', data=a_data).fit()

        # 1-way
        model = ols('value ~ C(depth)', data=a_data_now).fit() 
    #         anova_table = sm.stats.anova_lm(model, typ=3)
        anova_table = sm.stats.anova_lm(model, typ=2)

#         print(anova_table)        
#         print('\n')

        # scipy anova: same result as above
    #                 a = aa[inds_v1,:]
    #                 fvalue, pvalue = st.f_oneway(a[0][~np.isnan(a[0])], a[1][~np.isnan(a[1])], a[2][~np.isnan(a[2])], a[3][~np.isnan(a[3])])
    #                 print(fvalue, pvalue)


        ### TUKEY HSD        
        from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)

    #     v = a_data['value']
    #     f = a_data['depth']

        v = a_data_now['value'] #a_data['value']
        f = a_data_now['idepth'] #a_data['depth']
#         f = a_data_now['idepth'] # if you want to have depth (instead of depth index) in the summary table use this. it's easier to go with depth index because sometimes some depth are nan and missing, and matching depth is harder than matching depth indices.

        MultiComp = MultiComparison(v, f)
#         print(MultiComp.tukeyhsd().summary()) # Show all pair-wise comparisons

        tukey_all.append(MultiComp.tukeyhsd().summary())

        
    return tukey_all




####################################################################################################
####################################################################################################

def add_tukey_lines(tukey_all, toana, ax, col, inds_v1, inds_lm, inds_pooled, top, top_sd, x_new): 
    # toana = 'v1', or 'lm', or 'v1-lm'
    
    # ['V1', 'LM']    this is the order of indices in tukey_all
    '''
    if (inds_now == inds_v1).all():
        inds_v1_lm = 0
    elif (inds_now == inds_lm).all():
        inds_v1_lm = 1
    '''
    
    if toana == 'v1':
        inds_v1_lm = 0
        inds_now = inds_v1
    elif toana == 'lm':
        inds_v1_lm = 1
        inds_now = inds_lm
    elif toana == 'v1-lm':
        inds_v1_lm = 2
        inds_now = range(4)
        
        
    #####    
    tukey = tukey_all[inds_v1_lm]
#     print(tukey)
        
    y_new = top[inds_now, 1] + top_sd[inds_now, 1]
    mn = np.nanmin(top[:,2]-top_sd[:,2])
    mx = np.nanmax(top[:,1]+top_sd[:,1])
    
    t = np.array(tukey.data)
#     print(t.shape)
    # depth index for group 1 in tukey table
    g1inds = np.unique(np.array([t[i][[0,1]] for i in np.arange(1,t.shape[0])]).astype(int)[:,0])
    g2inds = np.unique(np.array([t[i][[0,1]] for i in np.arange(1,t.shape[0])]).astype(int)[:,1])
#     print(g1inds, g2inds)
    cnt = 0
    cntr = 0    
    for group1_ind in g1inds: #range(3):
        for group2_ind in g2inds[g2inds > group1_ind]: #np.arange(group1_ind+1, 4):
#             print(group1_ind, group2_ind)
            cnt = cnt+1
#             x_new = xnowall[0] # ophys3
            
            if tukey.data[cnt][-1] == False:
                txtsig = "ns" 
            else:
                txtsig = "*"
                cntr = cntr+1
                r = cntr*((mx-mn)/10)

                x1, x2 = x_new[group1_ind], x_new[group2_ind]   
                y, h, col = np.max([y_new[group1_ind], y_new[group2_ind]]) + r, (mx-mn)/20, col

#                 trans = ax.get_xaxis_transform()
#                 ax.annotate(txtsig, xy=((x1+x2)*.5, y+h), xycoords=trans, ha="center", va='top', color=col)
                ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, clip_on=True) #, transform=trans)
                ax.text((x1+x2)*.5, y+h, txtsig, ha='center', va='bottom', color=col)
                
                # plot the line outside, but it didnt work:
#                 https://stackoverflow.com/questions/47597534/how-to-add-horizontal-lines-as-annotations-outside-of-axes-in-matplotlib
                
    ylim = ax.get_ylim()
    
    
    return ylim



'''
from statannot import add_stat_annotation

x = x+xgap
y = top[inds_v1, 1]
tc
add_stat_annotation(ax, data=df, x=x, y=y, order=order,
                    box_pairs=[("Thur", "Fri"), ("Thur", "Sat"), ("Fri", "Sun")],
                    test='Mann-Whitney', text_format='star', loc='outside', verbose=2)
'''

                
'''
# heatmap
plt.imshow(top_allstage[:,:,1].T)
plt.colorbar()

plt.imshow(np.diff(top_allstage[:,:,1], axis=0).T)
plt.colorbar()
'''

        

