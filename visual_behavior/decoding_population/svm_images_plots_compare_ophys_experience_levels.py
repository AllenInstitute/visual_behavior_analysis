"""
Gets called in svm_images_plots_setVars.py

Makes summary errorbar plots for svm decoding accuracy across the experience levels

Vars needed here are set in svm_images_plots_setVars_sumMice3_svmdf.py

Created on Fri Oct 29 22:02:05 2021
@author: farzaneh

"""

        
##############################################################################################################################
##############################################################################################################################        
##############################################################################################################################
#%% Plot response amplitude for **experience levels**: errorbars comparing SVM decoding accuracy across experience levels; also do anova/tukey
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################        
        
#%% Make errorbars for response amplitude (averaged across all experiments for each experience level)

import visual_behavior.visualization.utils as utils
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)

cres = ['Slc17a7', 'Sst', 'Vip']

#%% Do stats; for each cre line, are the 3 experience levels significantly different? anova; then tukey
if np.isnan(svm_blocks) or svm_blocks==-101: # svm was run on the whole session (no block by block analysis)    

    tukey_all = []
    for cre in cres: # cre = cresdf[0]
        
        print(f'\n\n----------- Perfoming ANOVA/TUKEY on {cre} -----------\n')
        thiscre = svm_df[svm_df['cre_allPlanes']==cre]

        a_data_now = thiscre[['cre_allPlanes', 'experience_levels', 'peak_amp_allPlanes_allExp']]
        a_data_now = a_data_now.rename(columns={'peak_amp_allPlanes_allExp': 'value'})

        c = a_data_now.copy()
        print(c.shape)
        
        
        # only take valid values
        c = c[np.array([~np.isnan(c['value'].values[i][0]) for i in range(c.shape[0])])]
        print(c.shape)

        
        # replace Familiar, Novel 1, and Novel >1 in the df with 0, 1, and 2
        cnt = -1
        b = pd.DataFrame()
        for expl in exp_level_all:
            cnt = cnt+1
            a = c[c['experience_levels']==expl]
            a['experience_levels'] = [cnt for x in a['experience_levels']]
            b = pd.concat([b,a])
        c = b
        
        
        #### set the column "value" in the df that goes into anova model
        
        # take test and shuffle values
        test = np.array([c['value'].values[i][1] for i in range(c.shape[0])])
        shfl = np.array([c['value'].values[i][2] for i in range(c.shape[0])])

        tukey_all_ts_tsSh = []
        for j in range(2):
            if j==0: # run stats on test values
                v = test
            elif j==1: # run stats on test-shuffle values        
                v = test-shfl        
    #         v.shape

            c['value'] = list(v)
    #         c


            ############ ANOVA, 1-way ############
            model = ols('value ~ C(experience_levels)', data=c).fit() 
            anova_table = sm.stats.anova_lm(model, typ=2)
            print(anova_table)
            print('\n')


            ### TUKEY HSD        
            v = c['value']
            f = c['experience_levels']

            MultiComp = MultiComparison(v, f)
    #             MultiComp.tukeyhsd().summary()        
            print(MultiComp.tukeyhsd().summary()) # Show all pair-wise comparisons

        
            tukey_all_ts_tsSh.append(MultiComp.tukeyhsd().summary())
            
        tukey_all.append(tukey_all_ts_tsSh) # cres x 2 (test-shfl ; test) x tukey_table (ie 4 x7)
            


        
#%% Plot error bars for the SVM decoding accuracy across the 3 experience levels, for each cre line

colors = utils.get_experience_level_colors() # will always be in order of Familiar, Novel 1, Novel >1

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
        ax3 = plt.subplot(gs1[2])
        allaxes.append(ax3)
            
        
        ####### set df for all experience levels of a given cre line   
        df = resp_amp_sum_df[resp_amp_sum_df['cre']==crenow] 
        
        mn = np.nanmin(df['shfl_av']-df['shfl_sd'])
        mx = np.nanmax(df['test_av']+df['test_sd'])            

        
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
            

        
        ####### plot the errorbars, showing for each cre line, the svm decoding accuracy for the 3 experience levels
        # ax1: plot test and shuffled
        for pos, y, err, c in zip(xnowall, df['test_av'], df['test_sd'], colors):    
            ax1.errorbar(pos, y, err, fmt=fmt_now, markersize=5, color = c) # capsize = 0, capthick = 4, lw = 2, 
        for pos, y, err, c in zip(xnowall, df['shfl_av'], df['shfl_sd'], colors):                
            ax1.errorbar(pos, y, err, fmt=fmt_now, markersize=5, color = 'gray')
            
        # ax2: plot test - shuffle
        for pos, y, err, c in zip(xnowall, df['test_av']-df['shfl_av'], df['test_sd']-df['shfl_sd'], colors):    
            ax2.errorbar(pos, y, err, fmt=fmt_now, markersize=5, color = c) 
        

            
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
                ax.set_xticks(np.array(xnowall).flatten())
                ax.set_xticklabels(exp_level_all, rotation=45)
                ax.set_xlim([-.5, xnowall[-1][-1]+.5]) # x[-1]+xgap+.5

            ax.tick_params(labelsize=10)            
#             ax.set_xlabel(xlabs, fontsize=12)
#             ax.set_ylim(ylims_now)
            if ax==ax1:
                ax.set_ylabel(ylabs, fontsize=12) #, labelpad=35) # , rotation=0
                ax.set_title(f'data\n{areasn[iax]}', y=1.1)
            if ax==ax2:
                ax.set_title(f'data-shuffle\n{areasn[iax]}', y=1.1)

        
        ####### add legend
        bb = (.97,.8)
        if project_codes_all == ['VisualBehaviorMultiscope']:
            ax = ax3
        else:
            ax = ax1
        if addleg:
            ax.legend(loc='center left', bbox_to_anchor=[bb[0]+xgapg, bb[1]], frameon=True, handlelength=1, fontsize=12, numpoints=1)

        seaborn.despine()        
        
        


        ############### add_tukey_lines: if a pariwaise tukey comparison is significant add a line and a start
        iax = -1
        for ax in [ax1, ax2]: # test, test-shfl
            iax = iax+1
            
            x_new = xnowall
            tukey = tukey_all[icre][iax]
#             print(tukey)

            if ax==ax1: # testing data
                y_new = df['test_av']+df['test_sd'] #top[inds_now, 1] + top_sd[inds_now, 1]
                mn = np.nanmin(df['shfl_av']-df['shfl_sd'])
                mx = np.nanmax(df['test_av']+df['test_sd'])                
            elif ax==ax2: # test-shfl
                y_new = (df['test_av']-df['shfl_av']) + df['test_sd']
                mn = np.nanmin((df['test_av']-df['shfl_av']) - df['test_sd'])
                mx = np.nanmax((df['test_av']-df['shfl_av']) + df['test_sd'])            
#             print(mn,mx)
            
                
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
    #                 print(group1_ind, group2_ind)
                    cnt = cnt+1
        #             x_new = xnowall[0] # ophys3

                    if tukey.data[cnt][-1] == False:
                        txtsig = "ns" 
                    else:
                        txtsig = "*"
                        cntr = cntr+1
                        r = cntr*((mx-mn)/10)

                        x1, x2 = x_new[group1_ind], x_new[group2_ind]   
                        y, h, col = np.max([y_new[group1_ind], y_new[group2_ind]]) + r, (mx-mn)/20, 'k'

                        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, clip_on=True) #, transform=trans)
                        ax.text((x1+x2)*.5, y+h, txtsig, ha='center', va='bottom', color=col)

                        # plot the line outside, but it didnt work:
        #                 https://stackoverflow.com/questions/47597534/how-to-add-horizontal-lines-as-annotations-outside-of-axes-in-matplotlib

            ylim = ax.get_ylim()
            print(ylim)




        

        ####
        if dosavefig:

            whatSess = f'_summaryExperienceLevels'

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
            
            word = word + '_anovaTukey'

                
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
                
            nam = f'{crenow[:3]}{whatSess}_{bln}_aveSessPooled{fgn}_{now}'
            
            fign = os.path.join(dir0, 'svm', dir_now, nam+fmt)
            print(fign)
            
            
            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    

        
        
        