"""
Gets called in svm_images_plots_setVars.py

Makes summary errorbar plots for svm decoding accuracy comparing depth and area for each experience level.

Vars needed here are set in svm_images_plots_setVars_sumMice3_resp_sum_area_depth.py

Created on Fri Oct 29 22:02:05 2021
@author: farzaneh

"""

import matplotlib.gridspec as gridspec
import seaborn
import visual_behavior.visualization.utils as utils
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from anova_tukey_fn import *

sigval = .05 # value for ttest significance
fmt_all = ['o', 'x']
fmt_now = fmt_all[0]
if baseline_subtract: # subtract the baseline (CA average during baseline, ie before time 0) from the evoked CA (classification accuracy)
    ylabs = '% Class accuracy rel. baseline' #'Amplitude'
else:
    ylabs = '% Classification accuracy' #'Amplitude'    

cres = ['Slc17a7', 'Sst', 'Vip']


###### 
# labsad = ['LM', 'V1']
labsad = ['<200um', '>200um']

# svm_df_all = pd.concat(svm_df_allpr)


##############################################################################################################################        
##############################################################################################################################
#%% Plot response amplitude for depth/area comparison within each experience levels: 
# errorbars comparing SVM decoding accuracy (averaged across all experiemnts) between V1/LM or superifical/deep layers for each experience level; 
##############################################################################################################################
##############################################################################################################################
    

### Area comparison plots
        
##############################################################################################################################                
#%% Plot error bars for the SVM decoding accuracy across the 3 experience levels, for each cre line
##############################################################################################################################        

group_col = 'area'
# group_col = 'binned_depth'

# colors = utils.get_experience_level_colors() # will always be in order of Familiar, Novel 1, Novel >1

colors_bin1 = ['gray', 'gray', 'gray'] # lm; first element of resp_amp_sum_df_area for each cre and exp level
colors_bin2 = ['k', 'k', 'k'] # v1; second element of resp_amp_sum_df_area for each cre and exp level
colors = np.vstack([colors_bin1, colors_bin2]).T.flatten()

if project_codes_all == ['VisualBehaviorMultiscope']:
    x = np.array([0,1,2,3])*len(whichStages)*1.1
else:
    x = np.array([0])

if len(project_codes_all)==1:
    areasn = ['V1', 'LM', 'V1,LM']
else:
    areasn = ['V1,LM', 'V1,LM', 'V1,LM'] # distinct_areas  #['VISp']

addleg = 0
xgapg = .15*len(exp_level_all)/1.1  # gap between sessions within a depth    
    
if np.isnan(svm_blocks) or svm_blocks==-101: # svm was run on the whole session (no block by block analysis)    

    icre = -1
    for crenow in cres: # crenow = cres[0]
        icre = icre+1
        
        #############################################################################
        ####### set axes
        plt.figure(figsize=(6,2.5))  
        gs1 = gridspec.GridSpec(1,3) #, width_ratios=[3, 1]) 
        gs1.update(bottom=.15, top=0.8, left=0.05, right=0.95, wspace=.55, hspace=.5)

        allaxes = []
        ax1 = plt.subplot(gs1[0])
        allaxes.append(ax1)
        ax2 = plt.subplot(gs1[1])
        allaxes.append(ax2)
#         ax3 = plt.subplot(gs1[2])
#         allaxes.append(ax3)
            
        
        #############################################################################
        ####### set df for all experience levels of a given cre line   
        df = resp_amp_sum_df_area[resp_amp_sum_df_area['cre']==crenow] 
#         df = resp_amp_sum_df_depth[resp_amp_sum_df_depth['cre']==crenow] 
        
        mn = np.nanmin(df['shfl_av']-df['shfl_sd'])
        mx = np.nanmax(df['test_av']+df['test_sd'])            

        
        #############################################################################
        ####### set some vars for plotting
        stcn = -1
        xnowall = []
#         mn_mx_allstage = [] # min, max for each experience level    
        for expl in exp_level_all: # expl = exp_level_all[0]
            stcn = stcn+1
            xnow = x + xgapg*stcn            
            xnowall.append(xnow)
#             mn = np.nanmin(df['shfl_av']-df['shfl_sd'])
#             mx = np.nanmax(df['test_av']+df['test_sd'])            
#             mn_mx_allstage.append([mn, mx])            
            
        xnowall0 = copy.deepcopy(xnowall)
        xnowall = np.hstack([xnowall, xnowall]).flatten()
        
        legs = np.hstack([df[group_col][:2].values, '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_'])
        
        
        #############################################################################
        ####### plot the errorbars, showing for each cre line, the svm decoding accuracy for the 3 experience levels
        # ax1: plot test and shuffled
        # testing data
        for pos, y, err, c,l in zip(xnowall, df['test_av'], df['test_sd'], colors, legs):    
            ax1.errorbar(pos, y, err, fmt=fmt_now, markersize=5, color = c, label=l) # capsize = 0, capthick = 4, lw = 2, 

        # shuffled data
        for pos, y, err, c in zip(xnowall, df['shfl_av'], df['shfl_sd'], colors):                
            ax1.errorbar(pos, y, err, fmt=fmt_now, markersize=5, color = 'lightsteelblue')
            
        # ax2: plot test - shuffle
        for pos, y, err, c in zip(xnowall, df['test_av']-df['shfl_av'], df['test_sd']-df['shfl_sd'], colors):    
            ax2.errorbar(pos, y, err, fmt=fmt_now, markersize=5, color = c) 
        
        
        #############################################################################
        ####### plot an asterisk if p_area is significant

        p = p_depth_area_df[p_depth_area_df['cre']==crenow]['sig_area'].values # p_area_sigval[icre]
        ax1.plot(np.array(xnowall0).flatten(), p*mx-np.diff([mn,mx])*.03, color='tomato', marker='*', linestyle='') # cols_stages[stcn]
        

        #############################################################################    
        ####### take care of plot labels, etc; do this for each figure; ie each cre line 
#         ylims_now = [np.nanmin(ylims), np.nanmax(ylims)]

        plt.suptitle(crenow, fontsize=18, y=1.15)    
    
        iax = -1 # V1, LM
        for ax in allaxes: #[ax1,ax2,ax3]:
            iax=iax+1
            if project_codes_all == ['VisualBehaviorMultiscope']:
                ax.set_xticks(x) # depth
                ax.set_xticklabels(xticklabs, rotation=45)
                ax.set_xlim([-1, xnowall[-1][-1]+1]) # x[-1]+xgap+.5 # -.5-xgapg
            else:
                ax.set_xticks(np.array(xnowall0).flatten())
                ax.set_xticklabels(exp_level_all, rotation=45)
                ax.set_xlim([-.5, xnowall0[-1][-1]+.5]) # x[-1]+xgap+.5

            ax.tick_params(labelsize=10)            
#             ax.set_xlabel(xlabs, fontsize=12)
#             ax.set_ylim(ylims_now)
            if ax==ax1:
                ax.set_ylabel(ylabs, fontsize=12) #, labelpad=35) # , rotation=0
                ax.set_title(f'data\nall depths', y=1.1)
            if ax==ax2:
                ax.set_title(f'data-shuffle\nall depths', y=1.1)

                
#             if ax==ax3: # print number of experiments per experience level
#                 ax.set_title(f"n experiments\n{df['n_experiments'].values.astype(int)}", y=1.1)

                
        #############################################################################
        ####### add legend
        bb = (.4, 1.2)
        if project_codes_all == ['VisualBehaviorMultiscope']:
            ax = ax3
        else:
            ax = ax1
            
        if 1: #addleg:
            ax.legend(loc='center left', bbox_to_anchor=[bb[0]+xgapg, bb[1]], frameon=True, handlelength=1, fontsize=12, numpoints=1, prop={'size': 10})

        seaborn.despine()        
        
        



        

        ####
        if dosavefig:

            whatSess = f'_summaryExperienceLevelsArea'
            
            if use_matched_cells==123:
                whatSess = whatSess + '_matched_cells_FN1N2' #Familiar, N1, N+1
            elif use_matched_cells==12:
                whatSess = whatSess + '_matched_cells_FN1'
            elif use_matched_cells==23:
                whatSess = whatSess + '_matched_cells_N1Nn'
            elif use_matched_cells==13:
                whatSess = whatSess + '_matched_cells_FNn'        
            
            
            fgn = '' #f'{whatSess}'
            if same_num_neuron_all_planes:
                fgn = fgn + '_sameNumNeursAllPlanes'
            
            if baseline_subtract==1:
                bln = f'timewin{time_win}_blSubtracted'
            else:
                bln = f'timewin{time_win}_blNotSubtracted'                

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
            
#             word = word + '_anovaTukey'

                
            fgn = f'{fgn}_{word}'
            if len(project_codes_all)==1:
                fgn = f'{fgn}_frames{frames_svm[0]}to{frames_svm[-1]}'                        
            fgn = fgn + '_ClassAccur'
            fgn = f'{fgn}_allProjects'

            pcn = project_codes_all[2][0] + '_' # 'VisualBehaviorMultiscope'
#             if len(project_codes_all)==1:
#                 pcn = project_codes_all[0] + '_'
#             else:
#                 pcn = ''
#                 for ipc in range(len(project_codes_all)):
#                     pcn = pcn + project_codes_all[ipc][0] + '_'
            pcn = pcn[:-1]
            
            fgn = f'{fgn}_{pcn}'            
                
            nam = f'{crenow[:3]}{whatSess}_{bln}_aveExpPooled{fgn}_{now}'
            
            fign = os.path.join(dir0, 'svm', dir_now, nam+fmt)
            print(fign)
            
            
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    

        
        

########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################

# Depth comparison plots

        
##############################################################################################################################                
#%% Plot error bars for the SVM decoding accuracy across the 3 experience levels, for each cre line
##############################################################################################################################        

# group_col = 'area'
group_col = 'binned_depth'

# labsad = ['LM', 'V1']
# labsad = ['<200um', '>200um']

# colors = utils.get_experience_level_colors() # will always be in order of Familiar, Novel 1, Novel >1

colors_bin1 = ['gray', 'gray', 'gray'] # lm; first element of resp_amp_sum_df_area for each cre and exp level
colors_bin2 = ['k', 'k', 'k'] # v1; second element of resp_amp_sum_df_area for each cre and exp level
colors = np.vstack([colors_bin1, colors_bin2]).T.flatten()

if project_codes_all == ['VisualBehaviorMultiscope']:
    x = np.array([0,1,2,3])*len(whichStages)*1.1
else:
    x = np.array([0])

if len(project_codes_all)==1:
    areasn = ['V1', 'LM', 'V1,LM']
else:
    areasn = ['V1,LM', 'V1,LM', 'V1,LM'] # distinct_areas  #['VISp']

addleg = 0
xgapg = .15*len(exp_level_all)/1.1  # gap between sessions within a depth    
    
if np.isnan(svm_blocks) or svm_blocks==-101: # svm was run on the whole session (no block by block analysis)    

    icre = -1
    for crenow in cres: # crenow = cres[0]
        icre = icre+1
        
        ####### set axes
        plt.figure(figsize=(6,2.5))  
        gs1 = gridspec.GridSpec(1,3) #, width_ratios=[3, 1]) 
        gs1.update(bottom=.15, top=0.8, left=0.05, right=0.95, wspace=.55, hspace=.5)

        allaxes = []
        ax1 = plt.subplot(gs1[0])
        allaxes.append(ax1)
        ax2 = plt.subplot(gs1[1])
        allaxes.append(ax2)
#         ax3 = plt.subplot(gs1[2])
#         allaxes.append(ax3)
            
        
        ####### set df for all experience levels of a given cre line   
#         df = resp_amp_sum_df_area[resp_amp_sum_df_area['cre']==crenow] 
        df = resp_amp_sum_df_depth[resp_amp_sum_df_depth['cre']==crenow] 
        
        mn = np.nanmin(df['shfl_av']-df['shfl_sd'])
        mx = np.nanmax(df['test_av']+df['test_sd'])            

        
        #############################################################################
        ####### set some vars for plotting
        stcn = -1
        xnowall = []
#         mn_mx_allstage = [] # min, max for each experience level    
        for expl in exp_level_all: # expl = exp_level_all[0]
            stcn = stcn+1
            xnow = x + xgapg*stcn            
            xnowall.append(xnow)
#             mn = np.nanmin(df['shfl_av']-df['shfl_sd'])
#             mx = np.nanmax(df['test_av']+df['test_sd'])            
#             mn_mx_allstage.append([mn, mx])            
            
        xnowall0 = copy.deepcopy(xnowall)
        xnowall = np.hstack([xnowall, xnowall]).flatten()        
        
#         legs = np.hstack([df[group_col][:2].values, '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_'])
        legs = np.hstack([labsad, '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_'])

    
        #############################################################################
        ####### plot the errorbars, showing for each cre line, the svm decoding accuracy for the 3 experience levels
        # ax1: plot test and shuffled
        for pos, y, err, c,l in zip(xnowall, df['test_av'], df['test_sd'], colors, legs):    
            ax1.errorbar(pos, y, err, fmt=fmt_now, markersize=5, color = c, label=l) # capsize = 0, capthick = 4, lw = 2, 

        for pos, y, err, c in zip(xnowall, df['shfl_av'], df['shfl_sd'], colors):                
            ax1.errorbar(pos, y, err, fmt=fmt_now, markersize=5, color = 'lightsteelblue')
            
        # ax2: plot test - shuffle
        for pos, y, err, c in zip(xnowall, df['test_av']-df['shfl_av'], df['test_sd']-df['shfl_sd'], colors):    
            ax2.errorbar(pos, y, err, fmt=fmt_now, markersize=5, color = c) 
        

            
        #############################################################################
        ####### plot an asterisk if p_depth is significant

        p = p_depth_area_df[p_depth_area_df['cre']==crenow]['sig_depth'].values # p_depth_sigval[icre]
        ax1.plot(np.array(xnowall0).flatten(), p*mx-np.diff([mn,mx])*.03, color='tomato', marker='*', linestyle='') # cols_stages[stcn]

        
        #############################################################################
        ####### take care of plot labels, etc; do this for each figure; ie each cre line 
#         ylims_now = [np.nanmin(ylims), np.nanmax(ylims)]

        plt.suptitle(crenow, fontsize=18, y=1.15)    
    
        iax = -1 # V1, LM
        for ax in allaxes: #[ax1,ax2,ax3]:
            iax=iax+1
            if project_codes_all == ['VisualBehaviorMultiscope']:
                ax.set_xticks(x) # depth
                ax.set_xticklabels(xticklabs, rotation=45)
                ax.set_xlim([-1, xnowall[-1][-1]+1]) # x[-1]+xgap+.5 # -.5-xgapg
            else:
                ax.set_xticks(np.array(xnowall0).flatten())
                ax.set_xticklabels(exp_level_all, rotation=45)
                ax.set_xlim([-.5, xnowall0[-1][-1]+.5]) # x[-1]+xgap+.5

            ax.tick_params(labelsize=10)            
#             ax.set_xlabel(xlabs, fontsize=12)
#             ax.set_ylim(ylims_now)
            if ax==ax1:
                ax.set_ylabel(ylabs, fontsize=12) #, labelpad=35) # , rotation=0
                ax.set_title(f'data\n{areasn[iax]}', y=1.1)
            if ax==ax2:
                ax.set_title(f'data-shuffle\n{areasn[iax]}', y=1.1)

                
#             if ax==ax3: # print number of experiments per experience level
#                 ax.set_title(f"n experiments\n{df['n_experiments'].values.astype(int)}", y=1.1)

                
        #############################################################################
        ####### add legend
        bb = (.4, 1.2)
        if project_codes_all == ['VisualBehaviorMultiscope']:
            ax = ax3
        else:
            ax = ax1
            
        if 1: #addleg:
            ax.legend(loc='center left', bbox_to_anchor=[bb[0]+xgapg, bb[1]], frameon=True, handlelength=1, fontsize=12, numpoints=1, prop={'size': 10})

        seaborn.despine()        
        
        



        

        ####
        if dosavefig:

            whatSess = f'_summaryExperienceLevelsDepth'

            if use_matched_cells==123:
                whatSess = whatSess + '_matched_cells_FN1N2' #Familiar, N1, N+1
            elif use_matched_cells==12:
                whatSess = whatSess + '_matched_cells_FN1'
            elif use_matched_cells==23:
                whatSess = whatSess + '_matched_cells_N1Nn'
            elif use_matched_cells==13:
                whatSess = whatSess + '_matched_cells_FNn'        

                
            fgn = '' #f'{whatSess}'
            if same_num_neuron_all_planes:
                fgn = fgn + '_sameNumNeursAllPlanes'
            
            if baseline_subtract==1:
                bln = f'timewin{time_win}_blSubtracted'
            else:
                bln = f'timewin{time_win}_blNotSubtracted'                

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
            
#             word = word + '_anovaTukey'

                
            fgn = f'{fgn}_{word}'
            if len(project_codes_all)==1:
                fgn = f'{fgn}_frames{frames_svm[0]}to{frames_svm[-1]}'                        
            fgn = fgn + '_ClassAccur'
            fgn = f'{fgn}_allProjects'

            pcn = project_codes_all[2][0] + '_' # 'VisualBehaviorMultiscope'            
#             if len(project_codes_all)==1:
#                 pcn = project_codes_all[0] + '_'
#             else:
#                 pcn = ''
#                 for ipc in range(len(project_codes_all)):
#                     pcn = pcn + project_codes_all[ipc][0] + '_'
            pcn = pcn[:-1]
            
            fgn = f'{fgn}_{pcn}'            
                
            nam = f'{crenow[:3]}{whatSess}_{bln}_aveExpPooled{fgn}_{now}'
            
            fign = os.path.join(dir0, 'svm', dir_now, nam+fmt)
            print(fign)
            
            
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    

        
        

