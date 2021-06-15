"""
This script is called in "svm_images_plots_setVars_blocks.py"
Compare svm class accuracy across ophys stages
Do statistical tests (ttest; anova/tukey) between actual and shuffle; across depths; across ophys stages

Created on Tue Apr 27 12:09:05 2021
@author: farzaneh

"""


# whichStages: compare svm class accuracy for which stages?
# ttest will compare whichStages[0] and whichStages[1]
whichStages = [1,2,3,4,5,6] #[1,3,4,6] #[1,2,3] #[1,2] #[3,4] #stages to plot on top of each other to compare

ttest_actShfl_stages = 1 # 0: plot ttet comparison of actual and shuffled (it will be the default when len(whichStages)=1)); 1: plot ttest comparison between ophys stages
show_depth_stats = 1 # if 1, plot the bars that show anova/tukey comparison across depths


import visual_behavior.visualization.utils as utils
colors = utils.get_colors_for_session_numbers()
colors[2] = colors[0]
# familiar: colors[0]
# novel: colors[3]

if len(whichStages)==1:
    ttest_actShfl_stages = 0

fmt_now = fmt_all[0]


################################################################
################################################################                
#%% plot and compare class accuracy between different ophys stages                
################################################################
################################################################

import scipy.stats as st
import anova_tukey

cres = ['Slc17a7', 'Sst', 'Vip']
sigval = .05 # value for ttest significance
equal_var = False # needed for ttest # equal_var: If True (default), perform a standard independent 2 sample test that assumes equal population variances [1]. If False, perform Welch’s t-test, which does not assume equal population variance

ophys_stage_labels = ['Familiar 1', 'Familiar 2', 'Familiar 3', 'Novel 1', 'Novel 2', 'Novel 3']
inds_pooled = [[inds_v1[idepth] , inds_lm[idepth]] for idepth in range(len(inds_v1))]


#%% Compute ttest stats between actual and shuffled 

if trial_type == 'hits_vs_misses':
    stages_alln = [1,3,4,6]
else:
    stages_alln = [1,2,3,4,5,6]

p_act_shfl = np.full((len(cres), len(stages_alln), 8), np.nan)
p_act_shfl_pooled = np.full((len(cres), len(stages_alln), 4), np.nan) # pooled areas
icre = -1
for crenow in cres: # crenow = cres[0]
    icre = icre+1
    istage = -1
    for which_stage in stages_alln: # which_stage = 1 
        istage = istage+1
        a = summary_vars_all[np.logical_and(summary_vars_all['cre']==crenow , summary_vars_all['stage']==which_stage)]
        a_amp = a['resp_amp'].values[0][:,:,1] # actual # 8depths x sessions
        b_amp = a['resp_amp'].values[0][:,:,2] # shuffled

        # pool areas
        if project_codes == ['VisualBehaviorMultiscope']:
            a_amp_pooled = np.array([a_amp[inds_pooled[idepth]].flatten() for idepth in range(num_depth)])
            b_amp_pooled = np.array([b_amp[inds_pooled[idepth]].flatten() for idepth in range(num_depth)])
        
#         print(a_amp.shape, b_amp_pooled.shape)

        _, p = st.ttest_ind(a_amp, b_amp, nan_policy='omit', axis=1, equal_var=equal_var)
        p_act_shfl[icre, istage, :] = p
        
        # pooled areas
        if project_codes == ['VisualBehaviorMultiscope']:
            _, p = st.ttest_ind(a_amp_pooled, b_amp_pooled, nan_policy='omit', axis=1, equal_var=equal_var)
            p_act_shfl_pooled[icre, istage, :] = p
        

p_act_shfl_sigval = p_act_shfl+0 
p_act_shfl_sigval[p_act_shfl <= sigval] = 1
p_act_shfl_sigval[p_act_shfl > sigval] = np.nan
# print(p_act_shfl_sigval)

if project_codes == ['VisualBehaviorMultiscope']:
    p_act_shfl_sigval_pooled = p_act_shfl_pooled+0 
    p_act_shfl_sigval_pooled[p_act_shfl_pooled <= sigval] = 1
    p_act_shfl_sigval_pooled[p_act_shfl_pooled > sigval] = np.nan
    # print(p_act_shfl_sigval_pooled)
    
    
    
###############################################################
###############################################################
###############################################################

#%% Compute ttest stats between ophys 3 and 4, for each cre line

if len(whichStages)>1:
    if np.isnan(svm_blocks) or svm_blocks==-101:

        p_all = []
        p_all_pooled = []
        for crenow in cres: # crenow = cres[0]

            a = summary_vars_all[np.logical_and(summary_vars_all['cre']==crenow , summary_vars_all['stage']==whichStages[0])]
            b = summary_vars_all[np.logical_and(summary_vars_all['cre']==crenow , summary_vars_all['stage']==whichStages[1])]
            a_amp = a['resp_amp'].values[0][:,:,1] # 8depths x sessions
            b_amp = b['resp_amp'].values[0][:,:,1] 

            # pool areas # 4pooled depths x sessions
            if project_codes == ['VisualBehaviorMultiscope']:
                a_amp_pooled = np.array([a_amp[inds_pooled[idepth]].flatten() for idepth in range(num_depth)])
                b_amp_pooled = np.array([b_amp[inds_pooled[idepth]].flatten() for idepth in range(num_depth)])
            
#             print(a_amp.shape, b_amp.shape, a_amp_pooled.shape, b_amp_pooled.shape)

            _, p = st.ttest_ind(a_amp, b_amp, nan_policy='omit', axis=1, equal_var=equal_var)
            p_all.append(p)
            
            # shuffled
            if project_codes == ['VisualBehaviorMultiscope']:
                _, p = st.ttest_ind(a_amp_pooled, b_amp_pooled, nan_policy='omit', axis=1, equal_var=equal_var)
                p_all_pooled.append(p)
            
        p_all = np.array(p_all)
        p_all_pooled = np.array(p_all_pooled)

        p_sigval = p_all+0 
        p_sigval[p_all <= sigval] = 1
        p_sigval[p_all > sigval] = np.nan

        # pooled
        p_sigval_pooled = p_all_pooled+0 
        p_sigval_pooled[p_all_pooled <= sigval] = 1
        p_sigval_pooled[p_all_pooled > sigval] = np.nan
        
#         print(p_sigval)
#         print(p_sigval_pooled)



###############################################################
###############################################################
#%% Plot response amplitude (errorbars) comparing ophys stages; also do anova/tukey

###############################################################
#%% Set min and max for each cre line across all ophys stages that will be plotted below
if np.isnan(svm_blocks) or svm_blocks==-101:
    
    mn_mx_allcre = np.full((len(cres), 2), np.nan)
    for crenow in cres: # crenow = cres[0]
        
        tc = summary_vars_all[summary_vars_all['cre'] == crenow]
        
        icre = np.argwhere(np.in1d(cres, crenow))[0][0] #icre+1
        mn_mx_allstage = []
        
        for stagenow in whichStages: # stagenow = whichStages[0]
            tc_stage = tc[tc['stage'] == stagenow]

            pa_all = tc_stage['resp_amp'].values[0] 

            top = np.nanmean(pa_all, axis=1) # 8_planes x 4_trTsShCh
            top_sd = np.nanstd(pa_all, axis=1) / np.sqrt(pa_all.shape[1])        

            mn = np.nanmin(top[:,2]-top_sd[:,2])
            mx = np.nanmax(top[:,1]+top_sd[:,1])
            mn_mx_allstage.append([mn, mx])
    #         top_allstage.append(top)

        # top_allstage = np.array(top_allstage)
        mn_mx_allstage = np.array(mn_mx_allstage)
        mn_mx = [np.nanmin(mn_mx_allstage), np.nanmax(mn_mx_allstage)]

        mn_mx_allcre[icre,:] = mn_mx
    
    

###############################################################
#%% Plot response amplitude (errorbars) comparing ophys stages; also do anova/tukey

if np.isnan(svm_blocks) or svm_blocks==-101:
    
    if project_codes == ['VisualBehaviorMultiscope']:
        x = np.array([0,2,4,6])
    else:
        x = np.array([0])
    xgapg = .15*len(whichStages)/2
    areasn = ['V1', 'LM', 'V1-LM']
#     inds_now_all = [inds_v1, inds_lm]
    cols_stages = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # icre = -1
    for crenow in cres: # crenow = cres[0]

        tc = summary_vars_all[summary_vars_all['cre'] == crenow]
        icre = np.argwhere(np.in1d(cres, crenow))[0][0] #icre+1

        plt.figure(figsize=(6,2.5))  
        gs1 = gridspec.GridSpec(1,3) #, width_ratios=[3, 1]) 
        gs1.update(bottom=.15, top=0.8, left=0.05, right=0.95, wspace=.55, hspace=.5)

        ax1 = plt.subplot(gs1[0])
        ax2 = plt.subplot(gs1[1])
        ax3 = plt.subplot(gs1[2])

        xgap = 0
        top_allstage = []
        mn_mx_allstage = []    
        stcn = -1
        xnowall = []

        for stagenow in whichStages: # stagenow = whichStages[0]

            stcn = stcn+1
            tc_stage = tc[tc['stage'] == stagenow]


            pa_all = tc_stage['resp_amp'].values[0] 
    #         print(pa_all.shape)

            # average across both sessions and areas
            if project_codes == ['VisualBehaviorMultiscope']:
                top_pooled = np.array([np.nanmean(pa_all[inds_pooled[idepth]], axis=(0,1)) for idepth in range(num_depth)]) # 4depth (pooled areas) x 4_trTsShCh
                top_sd_pooled = np.array([np.nanstd(pa_all[inds_pooled[idepth]], axis=(0,1)) / np.sqrt(2*pa_all.shape[1]) for idepth in range(num_depth)]) # 4depth (pooled areas) x 4_trTsShCh

            top = np.nanmean(pa_all, axis=1) # 8_planes x 4_trTsShCh
            top_sd = np.nanstd(pa_all, axis=1) / np.sqrt(pa_all.shape[1])        

            depth_ave = tc_stage['depth_ave'].values[0] 


#             mn = np.nanmin(top[:,1]-top_sd[:,1])
            mn = np.nanmin(top[:,2]-top_sd[:,2])
            mx = np.nanmax(top[:,1]+top_sd[:,1])
            top_allstage.append(top)
            mn_mx_allstage.append([mn, mx])

            xnow = x + xgapg*stcn
            xnowall.append(xnow)
    #         print(x+xgap)

            # test
            ax1.errorbar(xnow, top[inds_v1, 1], yerr=top_sd[inds_v1, 1], fmt=fmt_now, markersize=5, capsize=0, label=f'{ophys_stage_labels[stagenow-1]}', color=colors[stagenow-1]) #cols_stages[stcn]
            if ~np.isnan(inds_lm[0]):
                ax2.errorbar(xnow, top[inds_lm, 1], yerr=top_sd[inds_lm, 1], fmt=fmt_now, markersize=5, capsize=0, label=f'{ophys_stage_labels[stagenow-1]}', color=colors[stagenow-1])
            # shuffle
            ax1.errorbar(xnow, top[inds_v1, 2], yerr=top_sd[inds_v1, 2], fmt=fmt_now, markersize=3, capsize=0, color='gray')
            if ~np.isnan(inds_lm[0]):
                ax2.errorbar(xnow, top[inds_lm, 2], yerr=top_sd[inds_lm, 2], fmt=fmt_now, markersize=3, capsize=0, color='gray')
            
            ##### areas pooled
            if project_codes == ['VisualBehaviorMultiscope']:
                # test
                ax3.errorbar(xnow, top_pooled[:, 1], yerr=top_sd_pooled[:, 1], fmt=fmt_now, markersize=5, capsize=0, label=f'{ophys_stage_labels[stagenow-1]}', color=colors[stagenow-1]) #cols_stages[stcn]
                # shuffle
                ax3.errorbar(xnow, top_pooled[:, 2], yerr=top_sd_pooled[:, 2], fmt=fmt_now, markersize=3, capsize=0, color='gray')

                
            ####### do anova and tukey hsd for pairwise comparison of depths per area
            ylims = []
            if project_codes == ['VisualBehaviorMultiscope'] and show_depth_stats:
                tukey_all = anova_tukey.do_anova_tukey(summary_vars_all, crenow, stagenow, inds_v1, inds_lm, inds_pooled)
#                 if show_depth_stats:
                a = anova_tukey.add_tukey_lines(tukey_all, 'v1', ax1, colors[stagenow-1], inds_v1, inds_lm, inds_pooled, top, top_sd, xnowall[stcn]) # cols_stages[stcn]
                b = anova_tukey.add_tukey_lines(tukey_all, 'lm', ax2, colors[stagenow-1], inds_v1, inds_lm, inds_pooled, top, top_sd, xnowall[stcn]) # cols_stages[stcn]
                c = anova_tukey.add_tukey_lines(tukey_all, 'v1-lm', ax3, colors[stagenow-1], inds_v1, inds_lm, inds_pooled, top_pooled, top_sd_pooled, xnowall[stcn]) # cols_stages[stcn]

                ylims.append(a)        
                ylims.append(b)
                ylims.append(c)

            else:
                ylims.append(ax1.get_ylim())
                ylims.append(ax2.get_ylim())
                ylims.append(ax3.get_ylim())
                
                
            ####### compare actual and shuffled for each ophys stage
            if ttest_actShfl_stages == 0: 
                ax1.plot(xnow, p_act_shfl_sigval[icre, stagenow-1, inds_v1]*mn_mx_allcre[icre][1]-np.diff(mn_mx_allcre[icre])*.03, color=colors[stagenow-1], marker='*', linestyle='') # cols_stages[stcn]
                ax2.plot(xnow, p_act_shfl_sigval[icre, stagenow-1, inds_lm]*mn_mx_allcre[icre][1]-np.diff(mn_mx_allcre[icre])*.03, color=colors[stagenow-1], marker='*', linestyle='')
                ax3.plot(xnow, p_act_shfl_sigval_pooled[icre, stagenow-1, :]*mn_mx_allcre[icre][1]-np.diff(mn_mx_allcre[icre])*.03, color=colors[stagenow-1], marker='*', linestyle='')                


        top_allstage = np.array(top_allstage)
        mn_mx_allstage = np.array(mn_mx_allstage)
        mn_mx = [np.nanmin(mn_mx_allstage), np.nanmax(mn_mx_allstage)]
        xlabs = 'Depth (um)'
        xticklabs = np.round(depth_ave).astype(int)  # x = np.arange(num_depth)

        ylims_now = [np.nanmin(ylims), np.nanmax(ylims)]

        # add a star if ttest is significant (between ophys 3 and 4)
        if ttest_actShfl_stages == 1: # compare ophys stages
            ax1.plot(x+xgapg/2, p_sigval[icre, inds_v1]*mn_mx[1]-np.diff(mn_mx)*.03, 'k*')
            if ~np.isnan(inds_lm[0]):
                ax2.plot(x+xgapg/2, p_sigval[icre, inds_lm]*mn_mx[1]-np.diff(mn_mx)*.03, 'k*')
            if project_codes == ['VisualBehaviorMultiscope']:
                ax3.plot(x+xgapg/2, p_sigval_pooled[icre, :]*mn_mx[1]-np.diff(mn_mx)*.03, 'k*')
        
        
        iax = 0 # V1, LM
        for ax in [ax1,ax2,ax3]:
#             ax.hlines(0, x[0], x[-1], linestyle='dashed', color='gray')
            ax.set_xticks(x)
            ax.set_xticklabels(xticklabs, rotation=45)
            ax.tick_params(labelsize=10)
            ax.set_xlim([-.5, xnowall[-1][-1]+.5]) # x[-1]+xgap+.5
            ax.set_xlabel(xlabs, fontsize=12)

    #         ax.set_ylim(mn_mx)
            ax.set_ylim(ylims_now)
            if ax==ax1:
                ax.set_ylabel(ylabs, fontsize=12) #, labelpad=35) # , rotation=0
            ax.set_title(areasn[iax])
            iax=iax+1

        bb = (.97,.8)
        ax3.legend(loc='center left', bbox_to_anchor=[bb[0]+xgapg, bb[1]], frameon=True, handlelength=1, fontsize=12)        

        plt.suptitle(crenow, fontsize=18, y=1.1)    
        seaborn.despine()





        ####
        if dosavefig:

            snn = [str(sn) for sn in whichStages]
            snn = '_'.join(snn)
            whatSess = f'_summaryStages_{snn}'

            fgn = '' #f'{whatSess}'
            if same_num_neuron_all_planes:
                fgn = fgn + '_sameNumNeursAllPlanes'

            if svm_blocks==-1:
                word = 'engagement_'
            else:
                word = ''
            
            if use_events:
                word = word + 'events'
            
            if show_depth_stats:
                word = word + '_anova'
                
            fgn = f'{fgn}_{word}_frames{frames_svm[0]}to{frames_svm[-1]}'                        
            fgn = fgn + '_ClassAccur'
            if project_codes == ['VisualBehavior']:
                fgn = f'{fgn}_{project_codes[0]}'

            nam = f'{crenow[:3]}{whatSess}_aveMice_aveSessPooled{fgn}_{now}'
            fign = os.path.join(dir0, 'svm', dir_now, nam+fmt)

            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    

        

        

        
        
#%% ################################################################################################        
####################################################################################################    
####################################################################################################        
######################## same as above but when svm was run on blocks of trials ####################
####################################################################################################    
####################################################################################################        
#%% ################################################################################################

if ~np.isnan(svm_blocks) and svm_blocks!=-101:
    
    whichStages_all = [[1], [2], [3], [4], [5], [6]]

    for whichStages in whichStages_all: #[1] #[1,2] #[3,4] #[1,3,4,6] # stages to plot on top of each other to compare

        blocks_toplot = [1] # which block to compare between stages in "whichStages" # only care about it if compare_stages_or_blocks = 'stages'; it will reset if compare_stages_or_blocks='stages'
        compare_stages_or_blocks = 'blocks' # 'blocks' # if "stages", we compare class accuracy of block "blocks_toplot" between the stages in "whichStages";  # if "blocks", we compare class accuracy of blocks in "blocks_toplot" for the stage "whichStages";  

        if compare_stages_or_blocks == 'blocks':
            blocks_toplot = br

        #%% Compute ttest stats between ophys 3 and 4, for each cre line

        cres = ['Slc17a7', 'Sst', 'Vip']

        equal_var = False # needed for ttest # equal_var: If True (default), perform a standard independent 2 sample test that assumes equal population variances [1]. If False, perform Welch’s t-test, which does not assume equal population variance
        sigval = .05 # value for ttest significance

        p_all = []
        for crenow in cres:

            if compare_stages_or_blocks == 'blocks':
                this_cre_stage_block_0 = np.all(np.array([summary_vars_all['cre']==crenow , summary_vars_all['stage']==whichStages[0] , summary_vars_all['block'] == 0]), axis=0)
                this_cre_stage_block_1 = np.all(np.array([summary_vars_all['cre']==crenow , summary_vars_all['stage']==whichStages[0] , summary_vars_all['block'] == 1]), axis=0)

            elif compare_stages_or_blocks == 'stages':
                this_cre_stage_block_0 = np.all(np.array([summary_vars_all['cre']==crenow , summary_vars_all['stage']==whichStages[0] , summary_vars_all['block'] == blocks_toplot[0]]), axis=0)
                this_cre_stage_block_1 = np.all(np.array([summary_vars_all['cre']==crenow , summary_vars_all['stage']==whichStages[1] , summary_vars_all['block'] == blocks_toplot[0]]), axis=0)


            a = summary_vars_all[this_cre_stage_block_0]
            b = summary_vars_all[this_cre_stage_block_1]
            a_amp = a['resp_amp'].values[0][:,:,1]
            b_amp = b['resp_amp'].values[0][:,:,1]

            print(a_amp.shape, b_amp.shape)

            _, p = st.ttest_ind(a_amp, b_amp, nan_policy='omit', axis=1, equal_var=equal_var)

            p_all.append(p)

        p_all = np.array(p_all)
        p_all


        p_sigval = p_all+0 
        p_sigval[p_all <= sigval] = 1
        p_sigval[p_all > sigval] = np.nan
        p_sigval



        #%% Plot response amplitude (errorbars) comparing ophys stages; also do anova/tukey

        x = np.array([0,2,4,6])
        xgapg = .15*len(whichStages)/2
        areasn = ['V1', 'LM']
        inds_now_all = [inds_v1, inds_lm]
        cols_stages = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # icre = -1
        for crenow in cres: #crenow = cres[0]

            tc = summary_vars_all[summary_vars_all['cre'] == crenow]
            icre = np.argwhere(np.in1d(cres, crenow))[0][0] #icre+1

            plt.figure(figsize=(4,2.5))  
            gs1 = gridspec.GridSpec(1,2) #, width_ratios=[3, 1]) 
            gs1.update(bottom=.15, top=0.8, left=0.05, right=0.95, wspace=.55, hspace=.5)

            xgap = 0
            top_allstage = []
            mn_mx_allstage = []    
            stcn = -1
            xnowall = []

            for stagenow in whichStages: # stagenow = whichStages[0]

                stcn = stcn+1

                for iblock in blocks_toplot: # np.unique(blocks_all): # iblock=0 ; iblock=np.nan

                    tc_stage = tc[np.logical_and(tc['stage'] == stagenow , tc['block'] == iblock)]

                    linestyle_now = linestyle_all[iblock]
                    cols_area_now = cols_area_all[iblock]
                    cols_stages_now = [cols_stages , ['cyan', 'pink']][iblock]


                    pa_all = tc_stage['resp_amp'].values[0] 
            #         print(pa_all.shape)

                    top = np.nanmean(pa_all, axis=1) # 8_planes x 4_trTsShCh
                    top_sd = np.nanstd(pa_all, axis=1) / np.sqrt(pa_all.shape[1])        

                    depth_ave = tc_stage['depth_ave'].values[0] 


                    if svm_blocks==-1:
        #                 lab_b = f'e{iblock}' # engagement
                        if iblock==0:
                            lab_b = f'disengaged'
                        elif iblock==1:
                            lab_b = f'engaged'            
                    else:
                        lab_b = f'b{iblock}' # block

                    lab = f'ophys{stagenow}, {lab_b}'


            #         mn = np.nanmin(top[:,1]-top_sd[:,1])
                    mn = np.nanmin(top[:,2]-top_sd[:,2])
                    mx = np.nanmax(top[:,1]+top_sd[:,1])
                    top_allstage.append(top)
                    mn_mx_allstage.append([mn, mx])

                    xnow = x + xgapg*stcn
                    xnowall.append(xnow)
            #         print(x+xgap)


                    ax1 = plt.subplot(gs1[0])
                    ax2 = plt.subplot(gs1[1])

                    # test
                    ax1.errorbar(xnow, top[inds_v1, 1], yerr=top_sd[inds_v1, 1], fmt=fmt_now, markersize=5, capsize=0, label=lab, color=cols_stages_now[stcn]) #, linestyle=linestyle_now)
                    ax2.errorbar(xnow, top[inds_lm, 1], yerr=top_sd[inds_lm, 1], fmt=fmt_now, markersize=5, capsize=0, label=lab, color=cols_stages_now[stcn]) #, linestyle=linestyle_now)

                    # shuffle
            #         ax1.errorbar(x+xgap, top[inds_v1, 2], yerr=top_sd[inds_v1, 2], fmt=fmt_now, markersize=3, capsize=0, color='gray')
            #         ax2.errorbar(x+xgap, top[inds_lm, 2], yerr=top_sd[inds_lm, 2], fmt=fmt_now, markersize=3, capsize=0, color='gray')




                ####### do anova and tukey hsd for pairwise comparison of depths per area
                tukey_all = anova_tukey.do_anova_tukey(summary_vars_all, crenow, stagenow, inds_v1, inds_lm)

                a = anova_tukey.add_tukey_lines(tukey_all, inds_v1, ax1, cols_stages_now[stcn], inds_v1, inds_lm, top, top_sd, xnowall[stcn])
                b = anova_tukey.add_tukey_lines(tukey_all, inds_lm, ax2, cols_stages_now[stcn], inds_v1, inds_lm, top, top_sd, xnowall[stcn])
                ylims = []        
                ylims.append(a)        
                ylims.append(b)


            top_allstage = np.array(top_allstage)
            mn_mx_allstage = np.array(mn_mx_allstage)
            mn_mx = [np.nanmin(mn_mx_allstage), np.nanmax(mn_mx_allstage)]
            xlabs = 'Depth (um)'
            xticklabs = np.round(depth_ave).astype(int)  # x = np.arange(num_depth)

            ylims_now = [np.nanmin(ylims), np.nanmax(ylims)]

            # add a star if ttest is significant (between ophys 3 and 4)
            ax1.plot(x+xgapg/2, p_sigval[icre, inds_v1]*mn_mx[1]-np.diff(mn_mx)*.03, 'k*')
            ax2.plot(x+xgapg/2, p_sigval[icre, inds_lm]*mn_mx[1]-np.diff(mn_mx)*.03, 'k*')

            iax = 0 # V1, LM
            for ax in [ax1,ax2]:
                ax.hlines(0, x[0], x[-1], linestyle='dashed', color='gray')
                ax.set_xticks(x)
                ax.set_xticklabels(xticklabs, rotation=45)
                ax.tick_params(labelsize=10)
                ax.set_xlim([-.5, x[-1]+xgap+.5])
                ax.set_xlabel(xlabs, fontsize=12)

        #         ax.set_ylim(mn_mx)
                ax.set_ylim(ylims_now)
                if ax==ax1:
                    ax.set_ylabel(ylabs, fontsize=12) #, labelpad=35) # , rotation=0
                ax.set_title(areasn[iax])
                iax=iax+1


            ax2.legend(loc='center left', bbox_to_anchor=[bb[0]+xgapg, bb[1]], frameon=True, handlelength=1, fontsize=12)        

            plt.suptitle(crenow, fontsize=18)    
            seaborn.despine()





            ####
            if dosavefig:

                snn = [str(sn) for sn in whichStages]
                snn = '_'.join(snn)
                whatSess = f'_summaryStages_{snn}'

                fgn = '' #f'{whatSess}'
                if same_num_neuron_all_planes:
                    fgn = fgn + '_sameNumNeursAllPlanes'

                if svm_blocks==-1:
                    word = 'engagement'
                elif svm_blocks==-101:
                    word = 'only_engaged'
                elif ~np.isnan(svm_blocks):
                    word = 'blocks'

                if use_events:
                    word = word + '_events'

                fgn = f'{fgn}_{word}_frames{frames_svm[0]}to{frames_svm[-1]}'                        
                fgn = fgn + '_ClassAccur'
                if project_codes == ['VisualBehavior']:
                    fgn = f'{fgn}_{project_codes[0]}'

                nam = f'{crenow[:3]}{whatSess}_aveMice_aveSessPooled{fgn}_{now}'
                fign = os.path.join(dir0, 'svm', dir_now, nam+fmt)
                print(fign)

                plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    




