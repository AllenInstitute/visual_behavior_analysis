"""
This script is called in "svm_images_plots_setVars_blocks.py"
Plot and compare svm class accuracy across ophys stages
Also make plots comparing experience levels
Do statistical tests (ttest; anova/tukey) between actual and shuffle; across depths; across ophys stages


Created on Tue Apr 27 12:09:05 2021
@author: farzaneh

"""

whichStages_pval_allComparisonParis = [[3,4], [4,6]] # compute ttest between stages 3,4 also between 4,6. # whichStages_pval = [3,4] # compute ttest between these 2 stages
ttest_actShfl_stages = 1 # 0: plot ttet comparison of actual and shuffled (it will be the default when len(whichStages)=1)); 1: plot ttest comparison between ophys stages, as long as len(whichStages)=2
show_depth_stats = 0 #1 # if 1, plot the bars that show anova/tukey comparison across depths


# whichStages: compare svm class accuracy for which stages?
# ttest will compare whichStages[0] and whichStages[1]
if trial_type =='hits_vs_misses':
    whichStages = [1,3,4,6]
else:    
    if summary_which_comparison == 'novelty': 
        whichStages = [1,3,4,6] #[1,2,3,4,5,6] #[1,2] #[3,4] #stages to plot on top of each other to compare
    elif summary_which_comparison == 'engagement':
        whichStages = [1,2,3] #
    elif summary_which_comparison == 'all':
        whichStages = [1,2,3,4,5,6]
    else:
        whichStages = summary_which_comparison

whichStages = np.array(whichStages)        
if svm_blocks==-1 and engagement_pupil_running==0: # because the engagement metric relies on lick measurement, there wont be data for passive sessions
    whichStages = whichStages[~np.in1d(whichStages, [2,5])]
    print(f'Removing passive sessions because there is no data for them!')
    



import visual_behavior.visualization.utils as utils
colors = utils.get_colors_for_session_numbers()
# colors[2] = colors[0]
# familiar: colors[0]
# novel: colors[3]

if len(whichStages)==1:
    ttest_actShfl_stages = 0

fmt_now = fmt_all[0]



#%%
import scipy.stats as st
import anova_tukey

cres = ['Slc17a7', 'Sst', 'Vip']
sigval = .05 # value for ttest significance
equal_var = False # needed for ttest # equal_var: If True (default), perform a standard independent 2 sample test that assumes equal population variances [1]. If False, perform Welch’s t-test, which does not assume equal population variance

ophys_stage_labels = ['Familiar 1', 'Familiar 2', 'Familiar 3', 'Novel 1', 'Novel 2', 'Novel 3']
inds_pooled = [[inds_v1[idepth] , inds_lm[idepth]] for idepth in range(len(inds_v1))]

inds_lm_orig = inds_lm
if len(project_codes_all)>1:
    inds_v1 = [0]
    inds_lm = [np.nan]





###############################################################
#%% function to plot the errorbars quanitfying response amplityde for each stage
###############################################################

def errorbar_respamp(compare_areas=False):
    
#     tc = summary_vars_all[summary_vars_all['cre'] == crenow]

    if np.isnan(svm_blocks) or svm_blocks==-101:
        tc_stage = tc[tc['stage'] == stagenow]
    else: # svm was run on blocks of trials
        tc_stage = tc[np.logical_and(tc['stage'] == stagenow , tc['block'] == iblock)]
        

    #################
    if len(project_codes_all)==1:
        pa_all = tc_stage['resp_amp'].values[0] # 8_planes x sessions x 4_trTsShCh
#             print(pa_all.shape)
        depth_ave = tc_stage['depth_ave'].values[0]

    else:
        # reshape data from each project to to (planes x sessions) x 4
        r_allpr = []
        for ipc in range(len(project_codes_all)):
            pa_all = tc_stage['resp_amp'].values[ipc] # 8_planes x sessions x 4_trTsShCh
            # reshape to (planes x sessions) x 4
            r = np.reshape(pa_all, (pa_all.shape[0] * pa_all.shape[1], pa_all.shape[2]), order='F') # first all planes of session 1, then all planes of session 2, etc  # r[:,-3], pa_all[:,1,-3]
            r_allpr.append(r)

        # concatenate data from both projects
        rr = np.concatenate((r_allpr)) # (planes x sessions) x 4
        pa_all = rr[np.newaxis, :] # 1 x (planes x sessions) x 4

        # pool and average depths across both projects
        depth_ave = [np.nanmean(np.concatenate((tc_stage['depth_ave'].values)))]

        inds_v1 = [0]
        inds_lm = [np.nan]
#             pa_all.shape


    # average across both sessions and areas
    if project_codes_all == ['VisualBehaviorMultiscope']:
        top_pooled = np.array([np.nanmean(pa_all[inds_pooled[idepth]], axis=(0,1)) for idepth in range(num_depth)]) # 4depth (pooled areas) x 4_trTsShCh
        top_sd_pooled = np.array([np.nanstd(pa_all[inds_pooled[idepth]], axis=(0,1)) / np.sqrt(2*pa_all.shape[1]) for idepth in range(num_depth)]) # 4depth (pooled areas) x 4_trTsShCh
    else:
        top_pooled = []
        top_sd_pooled = []
        
    # average across sessions
    top = np.nanmean(pa_all, axis=1) # 8_planes x 4_trTsShCh
    top_sd = np.nanstd(pa_all, axis=1) / np.sqrt(pa_all.shape[1])        
    #################


#         mn = np.nanmin(top[:,1]-top_sd[:,1])
    mn = np.nanmin(top[:,2]-top_sd[:,2])
    mx = np.nanmax(top[:,1]+top_sd[:,1])
#     top_allstage.append(top)
#     mn_mx_allstage.append([mn, mx])
    
    if compare_areas:
        xnow = x + xgapg*areai
    else:
        xnow = x + xgapg*stcn
#     xnowall.append(xnow)
#     print(x+xgap)

    return xnow, top, top_sd, mn, mx, depth_ave, top_pooled, top_sd_pooled, pa_all




##############################################################################    
####### take care of some plot labels etc
##############################################################################                


def add_plot_labs(top_allstage, mn_mx_allstage, ylims, addleg=1, compare_areas=False):
    
    top_allstage = np.array(top_allstage) # we are not using this in this function!!!
    mn_mx_allstage = np.array(mn_mx_allstage)
    
    mn_mx = [np.nanmin(mn_mx_allstage), np.nanmax(mn_mx_allstage)]
#     print(f'icre {icre}, stage {stagenow-1}, mn_mx: {mn_mx}')
    
    if project_codes_all == ['VisualBehaviorMultiscope']:
        xlabs = 'Depth (um)'
        xticklabs = np.round(depth_ave).astype(int)  # x = np.arange(num_depth)
    else:
        xlabs = 'Session'

    ylims_now = [np.nanmin(ylims), np.nanmax(ylims)]


    if compare_areas:
        p2plot = p_areas_sigval[icre, istage] # stagenow-1 # 4depths
    else:
        if len(project_codes_all)==1:
            p2plot = p_sigval[icre][:, inds_v1] # len(whichStages_pval_allComparisonParis) x 4  # previously: 4
        else:
            p2plot = p_sigval[icre] # len(whichStages_pval_allComparisonParis)
        
        
    # add an asterisk if ttest is significant between whichStages_pval (eg ophys 3 and 4)
    if compare_areas or (ttest_actShfl_stages == 1 and sum(np.in1d(whichStages, whichStages_pval_allComparisonParis))>=2): # compare ophys stages

        if compare_areas==False: # place it in between the x values for sessions 3 and 4
            cn = 'k'
            # compute x to place the asterisk
            cwp=-1
            for whichStages_pval in whichStages_pval_allComparisonParis:
                xs1 = np.argwhere(whichStages == whichStages_pval[0])[0][0]
                xs2 = np.argwhere(whichStages == whichStages_pval[1])[0][0]
                x_whichStages_pval = (xnowall[xs1] + xnowall[xs2]) / 2 # x+xgapg/2
                cwp = cwp+1
                
                # plot an asterisk if p2plot is significant    # print(f'test: {x_whichStages_pval, p2plot*mn_mx[1]-np.diff(mn_mx)*.03}')
                ax1.plot(x_whichStages_pval, p2plot[cwp]*mn_mx[1]-np.diff(mn_mx)*.03, '*', color=cn)
 
                if ~np.isnan(inds_lm[0]):
                    ax2.plot(x_whichStages_pval, p_sigval[icre, cwp, inds_lm]*mn_mx[1]-np.diff(mn_mx)*.03, 'k*')
                if project_codes_all == ['VisualBehaviorMultiscope']:
                    ax3.plot(x_whichStages_pval, p_sigval_pooled[icre, cwp, :]*mn_mx[1]-np.diff(mn_mx)*.03, 'k*')

        else:
            cn = 'r'
            xs1 = 0
            xs2 = 1
            x_whichStages_pval = (xnowall[xs1] + xnowall[xs2]) / 2 # x+xgapg/2
        
            # plot an asterisk if p2plot is significant
    #         print(f'test: {x_whichStages_pval, p2plot*mn_mx[1]-np.diff(mn_mx)*.03}')
            ax1.plot(x_whichStages_pval, p2plot*mn_mx[1]-np.diff(mn_mx)*.03, '*', color=cn)
        


    iax = 0 # V1, LM
    for ax in allaxes: #[ax1,ax2,ax3]:
    #             ax.hlines(0, x[0], x[-1], linestyle='dashed', color='gray')
        if project_codes_all == ['VisualBehaviorMultiscope']:
            ax.set_xticks(x) # depth
            ax.set_xticklabels(xticklabs, rotation=45)
            ax.set_xlim([-1, xnowall[-1][-1]+1]) # x[-1]+xgap+.5 # -.5-xgapg
        else:
            ax.set_xticks(np.array(xnowall).flatten()) # stages # 
            ax.set_xticklabels(whichStages, rotation=45)
            ax.set_xlim([-1, xnowall[-1][-1]+1]) # x[-1]+xgap+.5

        ax.tick_params(labelsize=10)            
        ax.set_xlabel(xlabs, fontsize=12)

    #                 ax.hlines(0, x[0], x[-1], linestyle='dashed', color='gray')

    #         ax.set_ylim(mn_mx)
#         print(f'test ylim: {ylims_now}')
        ax.set_ylim(ylims_now)
        if ax==ax1:
            ax.set_ylabel(ylabs, fontsize=12) #, labelpad=35) # , rotation=0
            
        if compare_areas:
            ax.set_title(f'ophys {stagenow}')
        else:
            ax.set_title(areasn[iax])
            
        iax=iax+1

    bb = (.97,.8)
    if project_codes_all == ['VisualBehaviorMultiscope']:
        ax = ax3
    else:
        ax = ax1
    if compare_areas==True:
        ax = ax1

    if addleg:
        ax.legend(loc='center left', bbox_to_anchor=[bb[0]+xgapg, bb[1]], frameon=True, handlelength=1, fontsize=12, numpoints=1)
    
#     if len(allaxes)==1:
#         ax.set_title(crenow, fontsize=18, y=1.1)    
#     else:

    if compare_areas==False:
        plt.suptitle(crenow, fontsize=18, y=1.1)    
        
    seaborn.despine()




##############################################################################
####### do anova and tukey hsd for pairwise comparison of depths per area, and per session type
##############################################################################

def run_anova_tukey():
    
    ylims = []
    if project_codes_all == ['VisualBehaviorMultiscope'] and show_depth_stats:
        tukey_all = anova_tukey.do_anova_tukey(summary_vars_all, crenow, stagenow, inds_v1, inds_lm, inds_pooled)
        a = anova_tukey.add_tukey_lines(tukey_all, 'v1', ax1, colors[stagenow-1], inds_v1, inds_lm, inds_pooled, top, top_sd, xnowall[stcn]) # cols_stages[stcn]
        b = anova_tukey.add_tukey_lines(tukey_all, 'lm', ax2, colors[stagenow-1], inds_v1, inds_lm, inds_pooled, top, top_sd, xnowall[stcn]) # cols_stages[stcn]
        c = anova_tukey.add_tukey_lines(tukey_all, 'v1-lm', ax3, colors[stagenow-1], inds_v1, inds_lm, inds_pooled, top_pooled, top_sd_pooled, xnowall[stcn]) # cols_stages[stcn]

        ylims.append(a)        
        ylims.append(b)
        ylims.append(c)

    else:
        ylims.append(ax1.get_ylim())
        if ~np.isnan(inds_lm[0]): # ~np.isnan(inds_lm).squeeze():
            ylims.append(ax2.get_ylim())
            ylims.append(ax3.get_ylim())

    return ylims





###############################################################
#%% Set stages_alln: all analyzable stages for a given decoding analysis
###############################################################

if trial_type == 'hits_vs_misses':
    stages_alln = [1,3,4,6]
else:
    if use_matched_cells == 123: # 
        print(f"Cells are matched across stages: {summary_vars_all['stage'].unique()}")
#         if sum(summary_vars_all['stage']==1)>0:
#             print(f'... changing stage 1 to 3, because we care about experience levels, not actual stage numbers!')
#             summary_vars_all.loc[summary_vars_all['stage']==1, 'stage']=3
        stages_alln = [3,4,6]
    else:
        stages_alln = [1,2,3,4,5,6]
stages_alln = np.array(stages_alln)    
if svm_blocks==-1 and engagement_pupil_running==0: # because the engagement metric relies on lick measurement, there wont be data for passive sessions
    stages_alln = stages_alln[~np.in1d(stages_alln, [2,5])]


    
###############################################################
#%% Compute p values and ttest stats between actual and shuffled 
###############################################################

p_act_shfl = np.full((len(cres), len(stages_alln), 8), np.nan)
p_act_shfl_pooled = np.full((len(cres), len(stages_alln), 4), np.nan) # pooled areas
icre = -1

for crenow in cres: # crenow = cres[0]
    icre = icre+1
    istage = -1
    
    for which_stage in stages_alln: # which_stage = 1 
        istage = istage+1

        a = summary_vars_all[np.logical_and(summary_vars_all['cre']==crenow , summary_vars_all['stage']==which_stage)]
        
        if len(project_codes_all)==1:
            a_amp = a['resp_amp'].values[0][:,:,1] # actual # 8depths x sessions
            b_amp = a['resp_amp'].values[0][:,:,2] # shuffled
        else:
            a_amp_allpr = []
            b_amp_allpr = []
            for ipc in range(len(project_codes_all)):
                apr = np.hstack(a['resp_amp'].values[ipc][:,:,1]) # data from 1 plane, all sessions; then 2nd plane all sessions, etc are vectorized.
                bpr = np.hstack(a['resp_amp'].values[ipc][:,:,2]) # data from 1 plane, all sessions; then 2nd plane all sessions, etc are vectorized.
                a_amp_allpr.append(apr) # actual # 8depths x sessions
                b_amp_allpr.append(bpr) # shuffled
            a_amp = np.concatenate((a_amp_allpr))[np.newaxis, :]
            b_amp = np.concatenate((b_amp_allpr))[np.newaxis, :]
            
#         a_amp.shape, b_amp.shape
        
        
        ############## pool areas ##############
        if project_codes_all == ['VisualBehaviorMultiscope']:
            a_amp_pooled = np.array([a_amp[inds_pooled[idepth]].flatten() for idepth in range(num_depth)])
            b_amp_pooled = np.array([b_amp[inds_pooled[idepth]].flatten() for idepth in range(num_depth)])
        
#         print(a_amp.shape, b_amp_pooled.shape)

        _, p = st.ttest_ind(a_amp, b_amp, nan_policy='omit', axis=1, equal_var=equal_var)
        p_act_shfl[icre, istage, :] = p
        
        # pooled areas
        if project_codes_all == ['VisualBehaviorMultiscope']:
            _, p = st.ttest_ind(a_amp_pooled, b_amp_pooled, nan_policy='omit', axis=1, equal_var=equal_var)
            p_act_shfl_pooled[icre, istage, :] = p
        

p_act_shfl_sigval = p_act_shfl+0 
p_act_shfl_sigval[p_act_shfl <= sigval] = 1
p_act_shfl_sigval[p_act_shfl > sigval] = np.nan
# print(p_act_shfl_sigval)

if project_codes_all == ['VisualBehaviorMultiscope']:
    p_act_shfl_sigval_pooled = p_act_shfl_pooled+0 
    p_act_shfl_sigval_pooled[p_act_shfl_pooled <= sigval] = 1
    p_act_shfl_sigval_pooled[p_act_shfl_pooled > sigval] = np.nan
    # print(p_act_shfl_sigval_pooled)
    
    
    
###############################################################
#%% Compute p values and ttest stats between the 2 ophys stages in whichStages (eg between ophys 3 and 4), for each cre line
###############################################################

if np.isnan(svm_blocks) or svm_blocks==-101:

    p_all_allComparisonParis = []
    p_all_pooled_allComparisonParis = []
    
    if sum(np.in1d(whichStages, whichStages_pval_allComparisonParis))>=2: #==2: # not anymore (I extend the code to compute p across any number of stage pairs): only run this section when comparing sessions 3 and 4
        for crenow in cres: # crenow = cres[0]

            p_all = []
            p_all_pooled = []

            for whichStages_pval in whichStages_pval_allComparisonParis: # whichStages_pval_allComparisonParis = [[3,4], [4,6]]
                
                # whichStages_pval = whichStages_pval_allComparisonParis[0]
                
                ############### set a_amp and b_amp vectors to use as input to ttest ###############

                a = summary_vars_all[np.logical_and(summary_vars_all['cre']==crenow , summary_vars_all['stage']==whichStages_pval[0])]
                b = summary_vars_all[np.logical_and(summary_vars_all['cre']==crenow , summary_vars_all['stage']==whichStages_pval[1])]

                if len(project_codes_all)==1:
                    a_amp = a['resp_amp'].values[0][:,:,1] # actual # 8depths x sessions
                    b_amp = b['resp_amp'].values[0][:,:,1] # shuffled
                else:
                    a_amp_allpr = []
                    b_amp_allpr = []
                    for ipc in range(len(project_codes_all)):
                        apr = np.hstack(a['resp_amp'].values[ipc][:,:,1]) # data from 1 plane, all sessions; then 2nd plane all sessions, etc are vectorized.
                        bpr = np.hstack(b['resp_amp'].values[ipc][:,:,1]) # data from 1 plane, all sessions; then 2nd plane all sessions, etc are vectorized.
                        a_amp_allpr.append(apr) # actual # 8depths x sessions
                        b_amp_allpr.append(bpr) # shuffled
                    a_amp = np.concatenate((a_amp_allpr))[np.newaxis, :]
                    b_amp = np.concatenate((b_amp_allpr))[np.newaxis, :]

#                     a_amp.shape, b_amp.shape


                ############## pool areas # 4pooled depths x sessions ##############
                if project_codes_all == ['VisualBehaviorMultiscope']:
                    a_amp_pooled = np.array([a_amp[inds_pooled[idepth]].flatten() for idepth in range(num_depth)])
                    b_amp_pooled = np.array([b_amp[inds_pooled[idepth]].flatten() for idepth in range(num_depth)])

        #             print(a_amp.shape, b_amp.shape, a_amp_pooled.shape, b_amp_pooled.shape)


                ############### compute p value ###############

                _, pi = st.ttest_ind(a_amp, b_amp, nan_policy='omit', axis=1, equal_var=equal_var)
                
                # pooled
                if project_codes_all == ['VisualBehaviorMultiscope']:
                    _, pp = st.ttest_ind(a_amp_pooled, b_amp_pooled, nan_policy='omit', axis=1, equal_var=equal_var)
                    
                    
                ################### done with a given pair in whichStages_pval_allComparisonParis; keep all values for a given cre line ##################                    
                p_all.append(pi)
#                 p_all = np.array(p_all)                
#                 if np.ndim(p_all)==1:
#                     p_all = p_all[:, np.newaxis]

                # pooled
                if project_codes_all == ['VisualBehaviorMultiscope']:                    
                    p_all_pooled.append(pp)
#                 p_all_pooled = np.array(p_all_pooled)

                    
    
            ################### done with looping over all values of whichStages_pval_allComparisonParis for a given cre line; keep all values for all cre lines ##################
            p_all_allComparisonParis.append(p_all) # P_all size: 2: len(whichStages_pval_allComparisonParis)
            p_all_pooled_allComparisonParis.append(p_all_pooled)
                

                
        ############### done with all cre lines ###############        
        p_all_allComparisonParis = np.array(p_all_allComparisonParis) # # p_all_allComparisonParis size: 3x2: num_cre x len(whichStages_pval_allComparisonParis)
        p_all_pooled_allComparisonParis = np.array(p_all_pooled_allComparisonParis)
            
            
        ############### compute significance ###############    
        
        p_sigval = p_all_allComparisonParis + 0 
        p_sigval[p_all_allComparisonParis <= sigval] = 1
        p_sigval[p_all_allComparisonParis > sigval] = np.nan

        # pooled
        p_sigval_pooled = p_all_pooled_allComparisonParis + 0 
        p_sigval_pooled[p_all_pooled_allComparisonParis <= sigval] = 1
        p_sigval_pooled[p_all_pooled_allComparisonParis > sigval] = np.nan

#         print(p_sigval)
#         print(p_sigval_pooled)





###############################################################
#%% Compute p values and ttest stats between visual areas: V1 vs. LM
###############################################################

if project_codes_all == ['VisualBehaviorMultiscope']:
    
    p_areas = np.full((len(cres), len(stages_alln), 4), np.nan) # num_cres x num_stages x 4depths
    icre = -1

    for crenow in cres: # crenow = cres[0]
        icre = icre+1
        istage = -1

        for which_stage in stages_alln: # which_stage = 1 
            istage = istage+1

            a = summary_vars_all[np.logical_and(summary_vars_all['cre']==crenow , summary_vars_all['stage']==which_stage)]

            if len(project_codes_all)==1:
                aa = a['resp_amp'].values[0][:,:,1] # actual # 8depths x sessions
                
                # v1:
                a_amp = aa[inds_v1] # 4depths x sessions                
                # lm:
                b_amp = aa[inds_lm]

            _, p = st.ttest_ind(a_amp, b_amp, nan_policy='omit', axis=1, equal_var=equal_var) # 4
            p_areas[icre, istage, :] = p

    p_areas_sigval = p_areas+0 
    p_areas_sigval[p_areas <= sigval] = 1
    p_areas_sigval[p_areas > sigval] = np.nan
    # print(p_areas_sigval)





###############################################################
#%% Set min and max for each cre line across all ophys stages that will be plotted below
###############################################################

if np.isnan(svm_blocks) or svm_blocks==-101:
    
    mn_mx_allcre = np.full((len(cres), 2), np.nan)
#     mn_mx_allstage_allcre = np.full((len(cres), len(whichStages), 2), np.nan)
    
    for crenow in cres: # crenow = cres[0]
        
        tc = summary_vars_all[summary_vars_all['cre'] == crenow]        
        icre = np.argwhere(np.in1d(cres, crenow))[0][0] #icre+1
        mn_mx_allstage = []
        
        for stagenow in whichStages: # stagenow = whichStages[0]
            tc_stage = tc[tc['stage'] == stagenow]
        
            for ipc in range(len(project_codes_all)):
                pa_all = tc_stage['resp_amp'].values[ipc] # 8_planes x sessions x 4_trTsShCh

                # average across sessions
                top = np.nanmean(pa_all, axis=1) # 8_planes x 4_trTsShCh
                top_sd = np.nanstd(pa_all, axis=1) / np.sqrt(pa_all.shape[1])        

                mn = np.nanmin(top[:,2]-top_sd[:,2]) # shuffled accuracy - sd
                mx = np.nanmax(top[:,1]+top_sd[:,1]) # testing accuracy + sd
                mn_mx_allstage.append([mn, mx])
                
#         mn_mx_allstage_allcre[icre,:] = mn_mx_allstage
        mn_mx_allstage = np.array(mn_mx_allstage)
        mn_mx = [np.nanmin(mn_mx_allstage), np.nanmax(mn_mx_allstage)]

        mn_mx_allcre[icre,:] = mn_mx
    
    
    
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
#%% Plot response amplitude (errorbars) comparing ophys STAGES; also do anova/tukey
## make separate figures for V1 and LM
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

if project_codes_all == ['VisualBehaviorMultiscope']:
    x = np.array([0,1,2,3])*len(whichStages)*1.1 #/1.5 #np.array([0,2,4,6]) # gap between depths
else:
    x = np.array([0])

cols_stages = plt.rcParams['axes.prop_cycle'].by_key()['color']

if len(project_codes_all)==1:
    areasn = ['V1', 'LM', 'V1,LM']
else:
    areasn = ['V1,LM', 'V1,LM', 'V1,LM'] # distinct_areas  #['VISp']

    
    
if np.isnan(svm_blocks) or svm_blocks==-101: # svm was run on the whole session (no block by block analysis)
    
    xgapg = .15*len(whichStages)/1.1  # gap between sessions within a depth
#     inds_now_all = [inds_v1, inds_lm]    
    
    # icre = -1
    for crenow in cres: # crenow = cres[0]

        tc = summary_vars_all[summary_vars_all['cre'] == crenow]
        icre = np.argwhere(np.in1d(cres, crenow))[0][0] #icre+1

        plt.figure(figsize=(6,2.5))  
        gs1 = gridspec.GridSpec(1,3) #, width_ratios=[3, 1]) 
        gs1.update(bottom=.15, top=0.8, left=0.05, right=0.95, wspace=.55, hspace=.5)

        allaxes = []
        ax1 = plt.subplot(gs1[0])
        allaxes.append(ax1)
        if ~np.isnan(inds_lm).squeeze().all():
            ax2 = plt.subplot(gs1[1])
            ax3 = plt.subplot(gs1[2])
            allaxes.append(ax2)
            allaxes.append(ax3)
            
        xgap = 0
        top_allstage = []
        mn_mx_allstage = []    
        stcn = -1
        xnowall = []

        for stagenow in whichStages: # stagenow = whichStages[0]

            stcn = stcn+1
            
            xnow, top, top_sd, mn, mx, depth_ave, top_pooled, top_sd_pooled, pa_all = errorbar_respamp()
            
            top_allstage.append(top)
            mn_mx_allstage.append([mn, mx])            
            xnowall.append(xnow)
    #         print(x+xgap)

            #######################################
            ############# plot errorbars #############        
            #######################################
            # test
            ax1.errorbar(xnow, top[inds_v1, 1], yerr=top_sd[inds_v1, 1], fmt=fmt_now, markersize=5, capsize=0, label=f'{ophys_stage_labels[stagenow-1]}', color=colors[stagenow-1]) #cols_stages[stcn]
            if ~np.isnan(inds_lm[0]):
                ax2.errorbar(xnow, top[inds_lm, 1], yerr=top_sd[inds_lm, 1], fmt=fmt_now, markersize=5, capsize=0, label=f'{ophys_stage_labels[stagenow-1]}', color=colors[stagenow-1])
            # shuffle
            ax1.errorbar(xnow, top[inds_v1, 2], yerr=top_sd[inds_v1, 2], fmt=fmt_now, markersize=3, capsize=0, color='gray')
            if ~np.isnan(inds_lm[0]):
                ax2.errorbar(xnow, top[inds_lm, 2], yerr=top_sd[inds_lm, 2], fmt=fmt_now, markersize=3, capsize=0, color='gray')
            
            ##### areas pooled
            if project_codes_all == ['VisualBehaviorMultiscope']:
                # test
                ax3.errorbar(xnow, top_pooled[:, 1], yerr=top_sd_pooled[:, 1], fmt=fmt_now, markersize=5, capsize=0, label=f'{ophys_stage_labels[stagenow-1]}', color=colors[stagenow-1]) #cols_stages[stcn]
                # shuffle
                ax3.errorbar(xnow, top_pooled[:, 2], yerr=top_sd_pooled[:, 2], fmt=fmt_now, markersize=3, capsize=0, color='gray')

                
                
            ##############################################################################
            ####### do anova and tukey hsd for pairwise comparison of depths per area
            ##############################################################################
            ylims = run_anova_tukey() # ylims for each cre line (after adding anova lines/stars), for each of the 3 axes (ax1,ax2,ax3)                
                
                
            ##############################################################################    
            ####### compare actual and shuffled for each ophys stage
            ##############################################################################
            if ttest_actShfl_stages == 0: 
                ax1.plot(xnow, p_act_shfl_sigval[icre, stagenow-1, inds_v1]*mn_mx_allcre[icre][1]-np.diff(mn_mx_allcre[icre])*.03, color=colors[stagenow-1], marker='*', linestyle='') # cols_stages[stcn]
                ax2.plot(xnow, p_act_shfl_sigval[icre, stagenow-1, inds_lm]*mn_mx_allcre[icre][1]-np.diff(mn_mx_allcre[icre])*.03, color=colors[stagenow-1], marker='*', linestyle='')
                ax3.plot(xnow, p_act_shfl_sigval_pooled[icre, stagenow-1, :]*mn_mx_allcre[icre][1]-np.diff(mn_mx_allcre[icre])*.03, color=colors[stagenow-1], marker='*', linestyle='')                

             
        ##############################################################################    
        ####### take care of plot labels, etc; done with all stages; do this for each cre line 
        ##############################################################################                
        add_plot_labs(top_allstage, mn_mx_allstage, ylims, addleg=1)
        
        
        

        ####
        if dosavefig:

            snn = [str(sn) for sn in whichStages]
            snn = '_'.join(snn)
            whatSess = f'_summaryStages_{snn}'

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
            
            if show_depth_stats:
                word = word + '_anova'

                
            fgn = f'{fgn}_{word}'
            if len(project_codes_all)==1:
                fgn = f'{fgn}_frames{frames_svm[0]}to{frames_svm[-1]}'                        
            fgn = fgn + '_ClassAccur'
#             if project_codes_all == ['VisualBehavior']:
#             fgn = f'{fgn}_{project_codes_all[0]}'

            if len(project_codes_all)==1:
                pcn = project_codes_all[0] + '_'
            else:
                pcn = ''
                for ipc in range(len(project_codes_all)):
                    pcn = pcn + project_codes_all[ipc][0] + '_'
            pcn = pcn[:-1]
            
            fgn = f'{fgn}_{pcn}'            
                
            nam = f'{crenow[:3]}{whatSess}_{bln}_aveMice_aveSessPooled{fgn}_{now}'
            
            fign = os.path.join(dir0, 'svm', dir_now, nam+fmt)
            print(fign)
            
            
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    

        

        
##############################################################################################################################
##############################################################################################################################        
##############################################################################################################################
#%% Plot response amplitude (errorbars) comparing visual AREAS per ophys stage; also do anova/tukey
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################        

if project_codes_all == ['VisualBehaviorMultiscope']:
    x = np.array([0,1,2,3])*len(whichStages)*1.1 #/1.5 #np.array([0,2,4,6]) # gap between depths
# else:
#     x = np.array([0])

    cols_stages = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if len(project_codes_all)==1:
        areasn = ['V1', 'LM', 'V1,LM']
    else:
        areasn = ['V1,LM', 'V1,LM', 'V1,LM'] # distinct_areas  #['VISp']

    areas_col = 'k', 'b'    
    inds_vl = inds_v1, inds_lm_orig

    if np.isnan(svm_blocks) or svm_blocks==-101: # svm was run on the whole session (no block by block analysis)

        xgapg = .15*len(inds_vl)/1.1  # gap between areas within a depth        
#         xgapg = .15*len(whichStages)/1.1  # gap between sessions within a depth
    #     inds_now_all = [inds_v1, inds_lm]    

        # icre = -1
        for crenow in cres: # crenow = cres[0]
            
            plt.figure(figsize=(1,4*len(whichStages)))  
            gs1 = gridspec.GridSpec(len(whichStages),1) #, width_ratios=[3, 1]) 
            gs1.update(bottom=.15, top=0.8, left=0.05, right=0.95, wspace=.55, hspace=1)

            tc = summary_vars_all[summary_vars_all['cre'] == crenow]
            icre = np.argwhere(np.in1d(cres, crenow))[0][0] #icre+1
            istage = -1
            
            for stagenow in whichStages: # stagenow = whichStages[0]
                istage = istage+1
                
                allaxes = []
                ax1 = plt.subplot(gs1[istage])
                allaxes.append(ax1)
#                 if ~np.isnan(inds_lm).squeeze().all():
#                     ax2 = plt.subplot(gs1[1])
#                     ax3 = plt.subplot(gs1[2])
#                     allaxes.append(ax2)
#                     allaxes.append(ax3)

                xgap = 0
                top_allstage = []
                mn_mx_allstage = []    
                areai = -1
                xnowall = []
                ylimsall = []
                
                # loop through areas
                for ia in range(2):
                    areai = areai+1

                    xnow, top, top_sd, mn, mx, depth_ave, top_pooled, top_sd_pooled, pa_all = errorbar_respamp(compare_areas=True)
                    top_allstage.append(top)
                    mn_mx_allstage.append([mn, mx])            
                    xnowall.append(xnow)
            #         print(x+xgap)

                    #######################################
                    ############# plot errorbars #############        
                    #######################################
                    # test
                    ax1.errorbar(xnow, top[inds_vl[ia], 1], yerr=top_sd[inds_vl[ia], 1], fmt=fmt_now, markersize=5, capsize=0, label=f'{areasn[ia]}', color=areas_col[ia])#colors[stagenow-1])#, linestyle = linestyle_all[0]) #cols_stages[stcn]
    #                 if ~np.isnan(inds_lm[0]):
    #                     ax1.errorbar(xnow, top[inds_vl[ia], 1], yerr=top_sd[inds_lm, 1], fmt=fmt_now, markersize=5, capsize=0, label=f'{ophys_stage_labels[stagenow-1]}', color=areas_col[ia])#, linestyle = linestyle_all[1])
                    # shuffle
                    ax1.errorbar(xnow, top[inds_vl[ia], 2], yerr=top_sd[inds_vl[ia], 2], fmt=fmt_now, markersize=3, capsize=0, color='gray')
    #                 if ~np.isnan(inds_lm[0]):
    #                     ax1.errorbar(xnow, top[inds_lm, 2], yerr=top_sd[inds_lm, 2], fmt=fmt_now, markersize=3, capsize=0, color='gray')

                    ##### areas pooled
        #             if project_codes_all == ['VisualBehaviorMultiscope']:
        #                 # test
        #                 ax3.errorbar(xnow, top_pooled[:, 1], yerr=top_sd_pooled[:, 1], fmt=fmt_now, markersize=5, capsize=0, label=f'{ophys_stage_labels[stagenow-1]}', color=colors[stagenow-1]) #cols_stages[stcn]
        #                 # shuffle
        #                 ax3.errorbar(xnow, top_pooled[:, 2], yerr=top_sd_pooled[:, 2], fmt=fmt_now, markersize=3, capsize=0, color='gray')



                    ##############################################################################
                    ####### do anova and tukey hsd for pairwise comparison of depths per area
                    ##############################################################################
        #             ylims = run_anova_tukey()                
        
                    ylims = ax1.get_ylim()
                    ylimsall.append(ylims)
                
                    ##############################################################################    
                    ####### compare actual and shuffled for each ophys stage
                    ##############################################################################
        #             if ttest_actShfl_stages == 0: 
        #                 ax1.plot(xnow, p_act_shfl_sigval[icre, stagenow-1, inds_v1]*mn_mx_allcre[icre][1]-np.diff(mn_mx_allcre[icre])*.03, color=colors[stagenow-1], marker='*', linestyle='') # cols_stages[stcn]
        #                 ax2.plot(xnow, p_act_shfl_sigval[icre, stagenow-1, inds_lm]*mn_mx_allcre[icre][1]-np.diff(mn_mx_allcre[icre])*.03, color=colors[stagenow-1], marker='*', linestyle='')
        #                 ax3.plot(xnow, p_act_shfl_sigval_pooled[icre, stagenow-1, :]*mn_mx_allcre[icre][1]-np.diff(mn_mx_allcre[icre])*.03, color=colors[stagenow-1], marker='*', linestyle='')                


                ##############################################################################    
                ####### take care of plot labels, etc; done with both areas for each stage
                ##############################################################################                
#                 print(f'mn_mx_allstage: {mn_mx_allstage}')
#                 print(f'ylimsall: {ylimsall}')
                ylimsall = np.array(ylimsall) # includes ylims for the two areas
                ylims = [np.nanmin(ylimsall.flatten()) , np.nanmax(ylimsall.flatten())]
#                 ylims = [np.nanmin(mn_mx_allstage), np.nanmax(mn_mx_allstage)] # same as: mn_mx_allstage_allcre[icre, stagenow-1]
                
#                 plt.subplots_adjust(hspace=2)
                add_plot_labs(top_allstage, mn_mx_allstage, ylims, addleg=1, compare_areas=True)
        



            ############ done with all stages, each cre line ############
            plt.suptitle(crenow, fontsize=18, y=.85);    

            if dosavefig:

                snn = [str(sn) for sn in whichStages]
                snn = '_'.join(snn)
                whatSess = f'_summaryAreas_stages{snn}'

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

                if show_depth_stats:
                    word = word + '_anova'


                fgn = f'{fgn}_{word}'
                if len(project_codes_all)==1:
                    fgn = f'{fgn}_frames{frames_svm[0]}to{frames_svm[-1]}'                        
                fgn = fgn + '_ClassAccur'
    #             if project_codes_all == ['VisualBehavior']:
    #             fgn = f'{fgn}_{project_codes_all[0]}'

                if len(project_codes_all)==1:
                    pcn = project_codes_all[0] + '_'
                else:
                    pcn = ''
                    for ipc in range(len(project_codes_all)):
                        pcn = pcn + project_codes_all[ipc][0] + '_'
                pcn = pcn[:-1]

                fgn = f'{fgn}_{pcn}'            

                nam = f'{crenow[:3]}{whatSess}_{bln}_aveMice_aveSessPooled{fgn}_{now}'

                fign = os.path.join(dir0, 'svm', dir_now, nam+fmt)
                print(fign)


                plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    
        
        
        
        
        
        
        

        
        
#%% ################################################################################################        
####################################################################################################    
####################################################################################################        
######################## Same as above but when svm was run on blocks of trials ####################
####################################################################################################    
####################################################################################################        
#%% ################################################################################################

if ~np.isnan(svm_blocks) and svm_blocks!=-101: # svm was run on blocks of trials

    ############################################################################
    #%% Compute ttest stats between blocks or stages (eg ophys 3 and 4), for each cre line
    ############################################################################
    
    whichStages_all = [[whichStages[i]] for i in range(len(whichStages))] #[[1], [2], [3], [4], [5], [6]]

    for wsn in whichStages_all: #wsn=[1] #[1,2] #[3,4] #[1,3,4,6] # stages to plot on top of each other to compare

        blocks_toplot = [1] # which block to compare between stages in "whichStages" # only care about it if compare_stages_or_blocks = 'stages'; it will reset if compare_stages_or_blocks='blocks'
        compare_stages_or_blocks = 'blocks' # 'blocks' # if "stages", we compare class accuracy of block "blocks_toplot" between the stages in "whichStages";  # if "blocks", we compare class accuracy of blocks in "blocks_toplot" for the stage "whichStages";  

        if compare_stages_or_blocks == 'blocks':
            blocks_toplot = br # [0,1]

        equal_var = False # needed for ttest # equal_var: If True (default), perform a standard independent 2 sample test that assumes equal population variances [1]. If False, perform Welch’s t-test, which does not assume equal population variance
        sigval = .05 # value for ttest significance

        cres = ['Slc17a7', 'Sst', 'Vip']

        
        p_all = []
        p_all_pooled = []
        for crenow in cres: # crenow = cres[0]

            if compare_stages_or_blocks == 'blocks':
                stagehere = wsn[0] # compare blocks for this stage
                this_cre_stage_block_0 = np.all(np.array([summary_vars_all['cre']==crenow , summary_vars_all['stage']==stagehere , summary_vars_all['block'] == 0]), axis=0)
                this_cre_stage_block_1 = np.all(np.array([summary_vars_all['cre']==crenow , summary_vars_all['stage']==stagehere , summary_vars_all['block'] == 1]), axis=0)

            elif compare_stages_or_blocks == 'stages':
                blockhere = blocks_toplot[0] # compare stages for this block
                this_cre_stage_block_0 = np.all(np.array([summary_vars_all['cre']==crenow , summary_vars_all['stage']==wsn[0] , summary_vars_all['block'] == blockhere]), axis=0)
                this_cre_stage_block_1 = np.all(np.array([summary_vars_all['cre']==crenow , summary_vars_all['stage']==wsn[1] , summary_vars_all['block'] == blockhere]), axis=0)

            a = summary_vars_all[this_cre_stage_block_0]
            b = summary_vars_all[this_cre_stage_block_1]
            
            if len(project_codes_all)==1:
                a_amp = a['resp_amp'].values[0][:,:,1] # actual # 8depths x sessions
                b_amp = b['resp_amp'].values[0][:,:,1] # shuffled
            else:
                a_amp_allpr = []
                b_amp_allpr = []
                for ipc in range(len(project_codes_all)):
                    apr = np.hstack(a['resp_amp'].values[ipc][:,:,1]) # data from 1 plane, all sessions; then 2nd plane all sessions, etc are vectorized.
                    bpr = np.hstack(b['resp_amp'].values[ipc][:,:,1]) # data from 1 plane, all sessions; then 2nd plane all sessions, etc are vectorized.
                    a_amp_allpr.append(apr) # actual # 8depths x sessions
                    b_amp_allpr.append(bpr) # shuffled
                a_amp = np.concatenate((a_amp_allpr))[np.newaxis, :]
                b_amp = np.concatenate((b_amp_allpr))[np.newaxis, :]

#             a_amp.shape, b_amp.shape

            
            # pool areas # 4pooled depths x sessions
            if project_codes_all == ['VisualBehaviorMultiscope']:
                a_amp_pooled = np.array([a_amp[inds_pooled[idepth]].flatten() for idepth in range(num_depth)])
                b_amp_pooled = np.array([b_amp[inds_pooled[idepth]].flatten() for idepth in range(num_depth)])
            
#             print(a_amp.shape, b_amp.shape, a_amp_pooled.shape, b_amp_pooled.shape)



            _, p = st.ttest_ind(a_amp, b_amp, nan_policy='omit', axis=1, equal_var=equal_var)
            p_all.append(p)
            
            # shuffled
            if project_codes_all == ['VisualBehaviorMultiscope']:
                _, p = st.ttest_ind(a_amp_pooled, b_amp_pooled, nan_policy='omit', axis=1, equal_var=equal_var)
                p_all_pooled.append(p)
            
        p_all = np.array(p_all)
        p_all_pooled = np.array(p_all_pooled)
#         p_all.shape
        if np.ndim(p_all)==1:
            p_all = p_all[:, np.newaxis]

        p_sigval = p_all+0 
        p_sigval[p_all <= sigval] = 1
        p_sigval[p_all > sigval] = np.nan

        # pooled
        p_sigval_pooled = p_all_pooled+0 
        p_sigval_pooled[p_all_pooled <= sigval] = 1
        p_sigval_pooled[p_all_pooled > sigval] = np.nan
        
#         print(p_sigval)
#         print(p_sigval_pooled)
        
     
        
        
    ############################################################################        
    ############################################################################        
    ############################################################################    
    ############################################################################
    #%% Plot response amplitude (errorbars) comparing ophys stages; also do anova/tukey
    ############################################################################

    xgapg = .15*len(whichStages)/2

    for crenow in cres: #crenow = cres[0]

        tc = summary_vars_all[summary_vars_all['cre'] == crenow]
        icre = np.argwhere(np.in1d(cres, crenow))[0][0] #icre+1

#             plt.figure(figsize=(4,2.5))  
#             gs1 = gridspec.GridSpec(1,2) #, width_ratios=[3, 1]) 
#             gs1.update(bottom=.15, top=0.8, left=0.05, right=0.95, wspace=.55, hspace=.5)

        plt.figure(figsize=(6,2.5))  
        gs1 = gridspec.GridSpec(1,3) #, width_ratios=[3, 1]) 
        gs1.update(bottom=.15, top=0.8, left=0.05, right=0.95, wspace=.55, hspace=.5)

        allaxes = []
        ax1 = plt.subplot(gs1[0])
        allaxes.append(ax1)
        if ~np.isnan(inds_lm).squeeze().all():
            ax2 = plt.subplot(gs1[1])
            ax3 = plt.subplot(gs1[2])
            allaxes.append(ax2)
            allaxes.append(ax3)



        xgap = 0
        top_allstage = []
        mn_mx_allstage = []    
        stcn = -1
        xnowall = []

        for stagenow in whichStages: # stagenow = whichStages[0]

            stcn = stcn+1

            for iblock in blocks_toplot: # np.unique(blocks_all): # iblock=0 ; iblock=np.nan

                linestyle_now = linestyle_all[iblock]
                cols_area_now = cols_area_all[iblock]
#                 cols_stages_now = [cols_stages , ['tomato', 'salmon', 'skyblue', 'cyan', 'pink']][iblock]
#                 cols_stages_now = [['tomato', 'tomato', 'tomato', 'tomato', 'tomato', 'tomato'], np.array(colors)[[0,2,3,5],:]][iblock] # ['tomato', 'salmon', 'skyblue', 'green', 'pink']
                cols_stages_now = [['tomato', 'tomato', 'tomato', 'tomato', 'tomato', 'tomato'], ['black', 'black', 'black', 'black', 'black', 'black']][iblock]
                line_styles_now = ['dashed', 'solid']

                if svm_blocks==-1:
    #                 lab_b = f'e{iblock}' # engagement
                    if iblock==0:
                        lab_b = f'disengaged'
                    elif iblock==1:
                        lab_b = f'engaged'            
                else:
                    lab_b = f'b{iblock}' # block

#                 lab = f'ophys{stagenow}, {lab_b}'
                lab = f'{lab_b}'


                xnow, top, top_sd, mn, mx, depth_ave, top_pooled, top_sd_pooled, pa_all = errorbar_respamp()                    

                top_allstage.append(top)
                mn_mx_allstage.append([mn, mx])

                #######################################
                ############# plot errorbars #############        
                #######################################
                
                lab='' if stcn!=0 else lab # only add legend for the 1st time point        
                
                if project_codes_all == ['VisualBehaviorMultiscope']:
                    colnow = colors[stagenow-1]
                else:
                    colnow = cols_stages_now[stcn]
                    
                # test
                eb1 = ax1.errorbar(xnow, top[inds_v1, 1], yerr=top_sd[inds_v1, 1], fmt=fmt_now, markersize=5, capsize=0, label=lab, color=colnow, linestyle = line_styles_now[iblock]) #cols_stages[stcn] # colors[stagenow-1]
                if iblock==0:
                    eb1[-1][0].set_linestyle('--')

                if ~np.isnan(inds_lm[0]):
                    ax2.errorbar(xnow, top[inds_lm, 1], yerr=top_sd[inds_lm, 1], fmt=fmt_now, markersize=5, capsize=0, label=lab, color=colnow, linestyle = line_styles_now[iblock])
                # shuffle
                ax1.errorbar(xnow, top[inds_v1, 2], yerr=top_sd[inds_v1, 2], fmt=fmt_now, markersize=3, capsize=0, color='gray')
                if ~np.isnan(inds_lm[0]):
                    ax2.errorbar(xnow, top[inds_lm, 2], yerr=top_sd[inds_lm, 2], fmt=fmt_now, markersize=3, capsize=0, color='gray')

                ##### areas pooled
                if project_codes_all == ['VisualBehaviorMultiscope']:
                    # test
                    ax3.errorbar(xnow, top_pooled[:, 1], yerr=top_sd_pooled[:, 1], fmt=fmt_now, markersize=5, capsize=0, label=lab, color=colors[stagenow-1], linestyle = line_styles_now[iblock]) #cols_stages[stcn]
                    # shuffle
                    ax3.errorbar(xnow, top_pooled[:, 2], yerr=top_sd_pooled[:, 2], fmt=fmt_now, markersize=3, capsize=0, color='gray')

            xnowall.append(xnow) # do it only for 1 block




            ####### done with plotting blocks_toplot
            ##############################################################################
            ####### do anova and tukey hsd for pairwise comparison of depths per area
            ##############################################################################
            ylims = run_anova_tukey()


        ##############################################################################    
        ####### compare actual and shuffled for each ophys stage
        ##############################################################################
        if ttest_actShfl_stages == 0: 
            ax1.plot(xnow, p_act_shfl_sigval[icre, stagenow-1, inds_v1]*mn_mx_allcre[icre][1]-np.diff(mn_mx_allcre[icre])*.03, color=colors[stagenow-1], marker='*', linestyle='') # cols_stages[stcn]
            ax2.plot(xnow, p_act_shfl_sigval[icre, stagenow-1, inds_lm]*mn_mx_allcre[icre][1]-np.diff(mn_mx_allcre[icre])*.03, color=colors[stagenow-1], marker='*', linestyle='')
            ax3.plot(xnow, p_act_shfl_sigval_pooled[icre, stagenow-1, :]*mn_mx_allcre[icre][1]-np.diff(mn_mx_allcre[icre])*.03, color=colors[stagenow-1], marker='*', linestyle='')                



        ##############################################################################    
        ####### take care of some plot labels etc
        ##############################################################################                
        import matplotlib as mpl        
        mpl.rcParams['legend.handlelength'] = 10
        mpl.rcParams['legend.markerscale'] = 0

        add_plot_labs(top_allstage, mn_mx_allstage, ylims, addleg=1)



        ####
        if dosavefig:

            snn = [str(sn) for sn in whichStages]
            snn = '_'.join(snn)
            whatSess = f'_summaryStages_{snn}'

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

            if show_depth_stats:
                word = word + '_anova'


            fgn = f'{fgn}_{word}'
            if len(project_codes_all)==1:
                fgn = f'{fgn}_frames{frames_svm[0]}to{frames_svm[-1]}'                        
            fgn = fgn + '_ClassAccur'
#             if project_codes_all == ['VisualBehavior']:
#             fgn = f'{fgn}_{project_codes_all[0]}'

            if len(project_codes_all)==1:
                pcn = project_codes_all[0] + '_'
            else:
                pcn = ''
                for ipc in range(len(project_codes_all)):
                    pcn = pcn + project_codes_all[ipc][0] + '_'
                pcn = pcn[:-1]

            fgn = f'{fgn}_{pcn}'            

            nam = f'{crenow[:3]}{whatSess}_{bln}_aveMice_aveSessPooled{fgn}_{now}'

            fign = os.path.join(dir0, 'svm', dir_now, nam+fmt)
            print(fign)


            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    



