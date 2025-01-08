#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define functions needed for UMAP analysis and plotting

Created on Tue Jul 21 11:52:25 2020
@author: farzaneh
"""


#%% Scatter plots for each cre line; 2 subplots, each colored for flash and omission responses.                

# import matplotlib.patches as mpatches

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')   

def scatter_umap(x, y, amp, clab, fig_ax='', lab_analysis='UMAP', cut_axes=0, same_norm_fo=[1,1], dosavefig=0, fign_subp='imageOmit_', pc_exp_var=np.nan, whatSess = '_AallBall', fgn='', dim=[2,2], cre='all'):

    # dim -> [2,2]: 2d plot ; [3,z]: 3d plot. (2nd element is the z values.)
    
#     low_dim_all_cre = embedding_all_cre

#     color_metric1 = peak_amp_eachN_traceMed_flash_all_cre # neurons will be color coded based on this metric in subplot 1
#     color_metric2 = peak_amp_eachN_traceMed_all_cre # neurons will be color coded based on this metric in subplot 2
#     color_metric = [color_metric1, color_metric2]

#     color_labs = ['image-evoked amplitude', 'omission-evoked amplitude']
    
#         x = low_dim_all_cre[icre][:, 0]
#         y = low_dim_all_cre[icre][:, 1]
#         amp = color_metric[isp][icre]    
#         clab = color_labs[isp]
            
    # from matplotlib import cm # colormap
    cmap = plt.cm.jet #viridis #bwr #or any other colormap 
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    
    if type(fig_ax)==np.str_: 
        fig = plt.figure(figsize=(14,5)) #(figsize=(10,3))
        ax = fig.add_subplot(1, len(color_metric), isp+1)     # ax = fig.add_subplot(111, projection='3d')
    else:
        fig = fig_ax[0]
        ax = fig_ax[1]
        
        
    ####################################################################################
    ############## set c_value (the colors of each point), and norm (how to scale the 3rd dimension of the plot) ##############
    ####################################################################################
    if type(amp[0])==np.str_ or type(amp[0])==str: # eg depth   #same_norm_fo[isp]==-1:
        norm = None

        u = np.unique(amp)    # u = np.unique(np.concatenate((color_metric[isp])))
        c_value = np.full((len(amp), 3), np.nan)
#         c_value = np.full((len(amp)), colors[0]) # when colors is strings, eg '#1f77b4'
        for ia in range(len(u)):
            c_value[amp==u[ia]] = colors[ia]
    else:
        c_value = amp

        if np.logical_and(same_norm_fo==0, type(c_value[0])!=np.str_): # use different cmap scales for image and omission
            mn = np.percentile(c_value, 1)
            mx = np.percentile(c_value, 99)
            norm = matplotlib.colors.Normalize(vmin=mn, vmax=mx)

            
    ######################################################################    
    ############## make the scatter plot ##############
    ######################################################################
    # each neuron is colored according to its omission or flash response, or depth/area, or cre line, etc
    if dim[0]==2:
        scatter = ax.scatter(x, y, s=1, label=cre[:3], marker='o', c=c_value, cmap=cmap, norm=norm) #cmap=cm.jet) 
    else:
        z = dim[1]
        scatter = ax.scatter(x, y, z, s=1, label=cre[:3], marker='o', c=c_value, cmap=cmap, norm=norm) #cmap=cm.jet) 



    ######################################################################
    ############## add legend, labels, colorbar, title, tick marks ##############
    ####################################################################################
    '''
    if type(amp[0])==np.str_: # eg depth    # same_norm_fo[isp]==-1:
        h = []
        for ia in range(len(u)):
            h.append(mpatches.Patch(color=colors[ia], label=u[ia]))
        plt.legend(handles=h, loc='center left', bbox_to_anchor=(-.1, 1.2, 1, .1), frameon=False, fontsize=12, ncol=2) # (1, .7)

    if type(pc_exp_var)==list:
        xle = f'\n{pc_exp_var[icre][0]:.02f} variance'
        yle = f'\n{pc_exp_var[icre][1]:.02f} variance'
    else:
        xle = ''
        yle = ''
    xlab = f'{lab_analysis} axis 1{xle}' # (arbitrary Units)')
    ylab = f'{lab_analysis} axis 2{yle}'
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)        #     ax.set_zlabel('UMAP Axis 3 ')    
    
    if type(amp[0])!=np.str_: # eg depth: dont add cmap
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=clab)
        cbar.set_label(clab, size=10)
        cbar.ax.tick_params(labelsize=10) 

    plt.title(clab, fontsize=12)
    ax.set_aspect('equal')

    if cut_axes:
        plt.xlim([mnx, mxx])
        plt.ylim([mny, mxy])

    else:
        xmj = np.unique(np.concatenate((np.arange(0, min(x), -5), np.arange(0, max(x), 5)))) # (max(x)-min(x))/5)
    #     xmn = np.arange(.25, x[-1], .5)
        ax.set_xticks(xmj)
        ax.set_yticks(xmj)
        ax.grid(True, which='both') # major
    '''
                
                
                
                
def plot_scatter_fo(cre_lines, low_dim_all_cre, color_metric, color_labs=['image_amp, omit_amp'], lab_analysis='UMAP', cut_axes=0, same_norm_fo=[1,1], dosavefig=0, fign_subp='imageOmit_', pc_exp_var=np.nan, whatSess='_AallBall', fgn='', dim=[2,2]):

#     low_dim_all_cre = embedding_all_cre
#     color_metric1 = peak_amp_eachN_traceMed_flash_all_cre # neurons will be color coded based on this metric in subplot 1
#     color_metric2 = peak_amp_eachN_traceMed_all_cre # neurons will be color coded based on this metric in subplot 2
#     color_metric = [color_metric1, color_metric2]
#     color_labs = ['image-evoked amplitude', 'omission-evoked amplitude']
#     cut_axes = 1 # if 1, show from 1 to 99 percentile of x and y values. helps when there are some outliers.
#     same_norm_fo = 1 # if 1, use the same cmap scale for both subplots (eg image and omission response amplitude); if 0, plot each using its own min and max. (NOTE: to avoid color saturation we use 1st and 99th percentiles of amplitude).


    for icre in range(len(cre_lines)): # icre = 2
        
        cre = cre_lines[icre]
#         all_sess_thisCre = all_sess_now.iloc[cre_all==cre]

        x = low_dim_all_cre[icre][:, 0]
        y = low_dim_all_cre[icre][:, 1]
    #     z = low_dim_all_cre[icre][:, 2]

    
        mnx = np.percentile(x, .5)
        mny = np.percentile(y, .5)    
        mxx = np.percentile(x, 99.5)
        mxy = np.percentile(y, 99.5)
        
        if same_norm_fo==1: # use the same cmap scale for image and omission
            amp = np.concatenate((color_metric[0][icre], color_metric[1][icre]))
            mn = np.percentile(amp, 1)
            mx = np.percentile(amp, 99)
            norm = matplotlib.colors.Normalize(vmin=mn, vmax=mx)
                        

        ##################################################################################
        ###### make different subplots, all show the same scatter plot ######
        ###### but color coded based on different metrics/ features #########
        ###### eg. image/omission response amplitude, area/depth ############
        ##################################################################################

        fig = plt.figure() #(figsize=(14,5)) #(figsize=(10,3))
        plt.suptitle(f'{cre[:3]}', y=.995, fontsize=22) 

        for isp in range(len(color_metric)): # isp = 0
            
            amp = color_metric[isp][icre]    
            clab = color_labs[isp]
            
            if dim[0]==2:
                ax = fig.add_subplot(1, len(color_metric), isp+1)

            else:
                ax = fig.add_subplot(1, len(color_metric), isp+1, projection='3d')
            
#             ax = fig.add_subplot(1, len(color_metric), isp+1)
            fig_ax = [fig, ax]
            
            # make the scatter plot of umap dimensions
            scatter_umap(x, y, amp, clab, fig_ax, lab_analysis, cut_axes, same_norm_fo, dosavefig, fign_subp, pc_exp_var, whatSess, fgn, dim, cre)

                
        plt.subplots_adjust(wspace=0.7)

        

        if dosavefig:
            cre_short = cre[:3]
            fgn_umap = f'{lab_analysis}_scatter_{fign_subp}allNeurSess'
    #         if useSDK:
    #             fgn_umap = f'{fgn_umap}_sdk_'
            nam = '%s_%s%s_%s_%s' %(cre_short, fgn_umap, whatSess, fgn, now)
            fign = os.path.join(dir0, dir_now, nam+fmt)     

            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    

