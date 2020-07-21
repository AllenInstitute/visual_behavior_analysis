#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vars needed here are set in umap_setVars_run.py

This script makes plots for umap and pca (which were ran on neurons x frames).


Created on Thu May 14 21:27:25 2020
@author: farzaneh
"""

import matplotlib.patches as mpatches


#%% Scatter plots for each cre line; 2 subplots, each colored for flash and omission responses.                

def scatter_umap(x, y, amp, clab, fig_ax='', lab_analysis='UMAP', cut_axes=0, same_norm_fo=[1,1], dosavefig=0, fign_subp='imageOmit_', pc_exp_var=np.nan, whatSess = '_AallBall', fgn='', dim=[2,2]):

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
    if type(amp[0])==np.str_: # eg depth   #same_norm_fo[isp]==-1:
        norm = None

        u = np.unique(amp)    # u = np.unique(np.concatenate((color_metric[isp])))
#         c_value = np.full((len(amp), 3), np.nan)
        c_value = np.full((len(amp)), colors[0]) # when colors is strings, eg '#1f77b4'
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
            scatter_umap(x, y, amp, clab, fig_ax, lab_analysis, cut_axes, same_norm_fo, dosavefig, fign_subp, pc_exp_var, whatSess, fgn, dim)

                
        plt.subplots_adjust(wspace=0.7)

        

        if dosavefig:
            cre_short = cre[:3]
            fgn_umap = f'{lab_analysis}_scatter_{fign_subp}allNeurSess'
    #         if useSDK:
    #             fgn_umap = f'{fgn_umap}_sdk_'
            nam = '%s_%s%s_%s_%s' %(cre_short, fgn_umap, whatSess, fgn, now)
            fign = os.path.join(dir0, dir_now, nam+fmt)     

            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    



            
            
##########################################################################################
###########################  Plots  ######################################################
##########################################################################################


################################################################################################                
# PCA plots
################################################################################################    

#%% Scatter plots for each cre line; 2 subplots, each colored for a distinct metric

lab_analysis = 'PCA'
cut_axes = 1 # if 1, show from 1 to 99 percentile of x and y values. helps when there are some outliers.
pc_exp_var = pca_variance_all_cre


# color by flash and omission responses.
same_norm_fo = 1 # if 1, use the same cmap scale for both subplots (eg image and omission response amplitude); if 0, plot each using its own min and max. (NOTE: to avoid color saturation we use 1st and 99th percentiles of amplitude).
color_labs = ['image-evoked amplitude', 'omission-evoked amplitude']
color_metric = [peak_amp_eachN_traceMed_flash_all_cre, peak_amp_eachN_traceMed_all_cre]
fign_subp = 'imageOmit_'
plot_scatter_fo(cre_lines, pc_all_cre, color_metric, color_labs, lab_analysis, cut_axes, same_norm_fo, dosavefig, fign_subp, pc_exp_var)


# color by area and depth.
same_norm_fo = 0
color_labs = ['area', 'depth']
color_metric = [area_all_cre, depth_all_cre]
fign_subp = 'areaDepth_'
plot_scatter_fo(cre_lines, pc_all_cre, color_metric, color_labs, lab_analysis, cut_axes, same_norm_fo, dosavefig, fign_subp, pc_exp_var)


# color by area and depth category (superficial, middle, deep).
same_norm_fo = 0
color_labs = ['area', 'depth']
# turn depth_categ_all_cre to str so plot_scatter_fo doesnt use colormap for it; otherwise bc vip doesnt have any deep neurons its colormap wont match the other cells colormap unless you add vmax to it (to be 2 and not 1)
depth_categ_all_cre = [depth_categ_all_cre[i].astype(str) for i in range(3)]
color_metric = [area_all_cre, depth_categ_all_cre]
fign_subp = 'areaDepth_'
plot_scatter_fo(cre_lines, pc_all_cre, color_metric, color_labs, lab_analysis, cut_axes, same_norm_fo, dosavefig, fign_subp, pc_exp_var)


    
################################################################################################                
# UMAP plots
################################################################################################                

#%% Scatter plots for each cre line; 2 subplots, each colored for a distinct metric

lab_analysis='UMAP'
cut_axes = 0
same_norm_fo = 1


# color by flash and omission responses.
color_labs = ['image-evoked amplitude', 'omission-evoked amplitude']
color_metric = [peak_amp_eachN_traceMed_flash_all_cre, peak_amp_eachN_traceMed_all_cre]
fign_subp = 'imageOmit_'
plot_scatter_fo(cre_lines, embedding_all_cre, color_metric, color_labs, lab_analysis, cut_axes, same_norm_fo, dosavefig, fign_subp)





#%% Make scatter plots; all cre lines superimposed 

color_cre_omit_flash = 3 # if 1: color neurons according to cre line; if 2, according to omission responses; if 3, according to flash responses.

cols_cre = ['b', 'r', 'g'] # slc, sst, vip

if color_cre_omit_flash==2:
    clab = 'omission-evoked amplitude'
    amp = peak_amp_eachN_traceMed_all_cre
elif color_cre_omit_flash==3:
    clab = 'image-evoked amplitude'
    amp = peak_amp_eachN_traceMed_flash_all_cre
mn = np.percentile(np.hstack(amp), 1)
mx = np.percentile(np.hstack(amp), 99)
norm = matplotlib.colors.Normalize(vmin=mn, vmax=mx)


fs = (5,5)
fig = plt.figure(figsize=fs) #(figsize=(10,3))
ax = fig.add_subplot(111)
# ax = fig.add_subplot(111, projection='3d')

for icre in range(len(cre_lines)): # icre = 2
    
    cre = cre_lines[icre]
    
    if color_cre_omit_flash==2:    
        amp = peak_amp_eachN_traceMed_all_cre[icre]
    elif color_cre_omit_flash==3:
        amp = peak_amp_eachN_traceMed_flash_all_cre[icre]    
#     mn = np.percentile(amp, 1)
#     mx = np.percentile(amp, 99)
#     norm = matplotlib.colors.Normalize(vmin=mn, vmax=mx)

    
    x = embedding_all_cre[icre][:, 0]
    y = embedding_all_cre[icre][:, 1]
#     z = embedding_all_cre[icre][:, 2]
    
#     ax = fig.add_subplot(1,3,icre+1, projection='3d')    
#     ax.set_title(cre)
#     ax.scatter(x, y, z, s=10, c=cols_cre[icre], label=cre[:3], marker='o') 
    '''
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette()[x] for x in penguins.species_short.map({"Adelie":0, "Chinstrap":1, "Gentoo":2})])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the Penguin dataset', fontsize=24)
    '''
    
    if color_cre_omit_flash==1:
        ax.scatter(x, y, s=10, c=cols_cre[icre], label=cre[:3], marker='o')
        ax.legend()

    else:     # each neuron is colored according to its omission or flash response
        ax.scatter(x, y, s=10, label=cre[:3], marker='o', c=amp, cmap=cmap, norm=norm) #cmap=cm.jet) 
    
    if icre==0:
        ax.set_xlabel('UMAP Axis 1 (Arbitrary Units)', )
        ax.set_ylabel('UMAP Axis 2')
    #     ax.set_zlabel('UMAP Axis 3 ')
        if color_cre_omit_flash!=1:
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=clab) 
    
    ax.set_aspect('equal')
    
    
   
                

    
    
########################################################################################################
########################################################################################################
#%% Manually choose subclasses for each cell line and plot their average traces.
########################################################################################################
########################################################################################################

############################################################
############################################################
#%% SLC subclasses
############################################################
############################################################
'''
slc:
    
large image; no omission ---> standard type
u2>10

small image; small omission --> this turned out to be the same as the standard type
u1<10; omit_resp<.015; img_resp<.015

small image; large omission --> vip type
u1>2; omit_resp>.02; omit_resp>(img_resp+.05)

large image; large omission ---> novel! --> this turned to be a subgroup of the above "vip type"
u1>10; omit_resp>.05; img_resp>.05
'''

icre = 0

cre = cre_lines[icre]
print(f'{cre}')
nr = 2; nc = 2

all_sess_ns_thisCre = all_sess_ns_all_cre[icre] # neurons_allExp_thisCre x 24(frames)
np.shape(all_sess_ns_thisCre)
#     all_sess_ns_fof_thisCre = all_sess_ns_fof_all_cre[icre] # neurons_allExp_thisCre x 24(frames)
#     np.shape(all_sess_ns_fof_thisCre)

x = embedding_all_cre[icre][:, 0]
y = embedding_all_cre[icre][:, 1]
#     z = embedding_all_cre[icre][:, 2]

amp_f = peak_amp_eachN_traceMed_flash_all_cre[icre]    
amp_o = peak_amp_eachN_traceMed_all_cre[icre]    

plt.figure()
plt.suptitle(f'{cre[:3]}', y=1.1, fontsize=14) 

############################################################
# large image; no omission ---> standard type
# y>10
############################################################
s = 1 # subplot

c1 = (y > 10)
c2 = (x < 10)
c = np.logical_or(c1, c2)
#     plt.plot(amp_f[c], 'b'); plt.plot(amp_o[c], 'r') # sanity check
st = f'{np.mean(c): .2f}\n{sum(c)}/{len(x)} neurons'
print(st)    

#     all_sess_ns_thisCre_thisCond = all_sess_ns_fof_thisCre[c] # neurons_thisCre_thisCondition x 24(frames)
all_sess_ns_thisCre_thisCond = all_sess_ns_thisCre[c] # neurons_thisCre_thisCondition x 40(frames)    
np.shape(all_sess_ns_thisCre_thisCond)

# average 
plt.subplot(nr,nc,s)
top = np.mean(all_sess_ns_thisCre_thisCond, axis=0)
h1 = plt.plot(time_trace, top)
plot_flashLines_ticks_legend([], h1, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, bbox_to_anchor=bb, ylab=ylabel, xmjn=xmjn)
# plt.vlines(samps_bef, min(top), max(top))
plt.title(st, fontsize=10)


############################################################
# small image; large omission --> vip type
# x>2; omit_resp>.02; omit_resp>(img_resp+.05)
############################################################
s = 2 # subplot

c1 = (x > 2)
c2 = (amp_o > .02)
c3 = (amp_o > (amp_f+.05))
c = np.logical_and(np.logical_and(c1,c2),c3)
#     plt.plot(amp_f[c]); plt.plot(amp_o[c]) # sanity check
st = f'{np.mean(c): .2f}\n{sum(c)}/{len(x)} neurons'
print(st)    

#     all_sess_ns_thisCre_thisCond = all_sess_ns_fof_thisCre[c] # neurons_thisCre_thisCondition x 24(frames)
all_sess_ns_thisCre_thisCond = all_sess_ns_thisCre[c] # neurons_thisCre_thisCondition x 40(frames)    
np.shape(all_sess_ns_thisCre_thisCond)

# average 
plt.subplot(nr,nc,s)
top = np.mean(all_sess_ns_thisCre_thisCond, axis=0)
h1 = plt.plot(time_trace, top)
plot_flashLines_ticks_legend([], h1, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, bbox_to_anchor=bb, ylab=ylabel, xmjn=xmjn)
plt.title(st, fontsize=10)

plt.subplots_adjust(wspace=0.5)

# scatter plot, colored by amp_f - amp_o
# norm=matplotlib.colors.Normalize(vmin=np.percentile(amp_f-amp_o, 1), vmax=np.percentile(amp_f-amp_o, 99)); fig=plt.figure(); plt.gca().scatter(x,y,marker='.', c=amp_f-amp_o, cmap=cmap, norm=norm); fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm))
    
if dosavefig:
    cre_short = cre[:3]
    nam = '%s_umap2D_manualClasses_allNeurSess%s_%s_%s' %(cre_short, whatSess, fgn, now)
    fign = os.path.join(dir0, dir_now, nam+fmt)        
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    


    
############################################################
############################################################    
#%% SST subclasses
############################################################
############################################################
'''
sst:

large image; no omission (falls into 2 parts of umap. why?) ---> standard response
1. 0<y<10
2. y<0
(combining the two; no difference noticed.)

small image; large omission --> vip type
y>10
amp_o>.05
amp_f < amp_o

small image; no omission
'''

icre = 1

cre = cre_lines[icre]
print(f'{cre}')
nr = 2; nc = 2

all_sess_ns_thisCre = all_sess_ns_all_cre[icre] # neurons_allExp_thisCre x 24(frames)
np.shape(all_sess_ns_thisCre)
#     all_sess_ns_fof_thisCre = all_sess_ns_fof_all_cre[icre] # neurons_allExp_thisCre x 24(frames)
#     np.shape(all_sess_ns_fof_thisCre)

x = embedding_all_cre[icre][:, 0]
y = embedding_all_cre[icre][:, 1]
#     z = embedding_all_cre[icre][:, 2]

amp_f = peak_amp_eachN_traceMed_flash_all_cre[icre]    
amp_o = peak_amp_eachN_traceMed_all_cre[icre]    

plt.figure()
plt.suptitle(f'{cre[:3]}', y=1.1, fontsize=14) 

############################################################
# large image; no omission (falls into 2 parts of umap. why?) ---> standard response
# 1. 0<y<10
# 2. y<0
############################################################
s = 1 # subplot

c1 = (y < 10)
c2 = np.logical_and((y > 10), amp_o <= .05)
c = np.logical_or(c1, c2)
# plt.plot(amp_f[c] - amp_o[c])
# plt.plot(amp_f[c], 'b'); plt.plot(amp_o[c], 'r') # sanity check
st = f'{np.mean(c): .2f}\n{sum(c)}/{len(x)} neurons'
print(st)    

all_sess_ns_thisCre_thisCond = all_sess_ns_thisCre[c] # neurons_thisCre_thisCondition x 40(frames)    
np.shape(all_sess_ns_thisCre_thisCond)

# average 
plt.subplot(nr,nc,s)
top = np.mean(all_sess_ns_thisCre_thisCond, axis=0)
h1 = plt.plot(time_trace, top)
plot_flashLines_ticks_legend([], h1, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, bbox_to_anchor=bb, ylab=ylabel, xmjn=xmjn)
plt.title(st, fontsize=10)


############################################################
# small image; large omission --> vip type
# y > 10
# amp_o > .05
# amp_f < amp_o
############################################################
s = 2 # subplot

c1 = (y > 10)
c2 = (amp_o > .05)
c3 = amp_f < amp_o
c = np.logical_and(np.logical_and(c1, c2), c3)
# plt.plot(amp_f[c] - amp_o[c])
# plt.plot(amp_f[c], 'b'); plt.plot(amp_o[c], 'r') # sanity check
st = f'{np.mean(c): .2f}\n{sum(c)}/{len(x)} neurons'
print(st)    

all_sess_ns_thisCre_thisCond = all_sess_ns_thisCre[c] # neurons_thisCre_thisCondition x 40(frames)    
np.shape(all_sess_ns_thisCre_thisCond)

# average 
plt.subplot(nr,nc,s)
top = np.mean(all_sess_ns_thisCre_thisCond, axis=0)
h1 = plt.plot(time_trace, top)
plot_flashLines_ticks_legend([], h1, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, bbox_to_anchor=bb, ylab=ylabel, xmjn=xmjn)
plt.title(st, fontsize=10)

plt.subplots_adjust(wspace=0.5)


if dosavefig:
    cre_short = cre[:3]
    nam = '%s_umap2D_manualClasses_allNeurSess%s_%s_%s' %(cre_short, whatSess, fgn, now)
    fign = os.path.join(dir0, dir_now, nam+fmt)        
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    

    
    
    
############################################################
############################################################
#%% VIP subclasses
############################################################
############################################################
'''
vip:
    
no image; large omission ---> standard response
x < 10; amp_o > amp_f

no image; small omission ---> standard response
x>10

large image; no omission ---> slc type 
x < 10; amp_f > amp_o
'''

icre = 2

cre = cre_lines[icre]
print(f'{cre}')
nr = 2; nc = 2

all_sess_ns_thisCre = all_sess_ns_all_cre[icre] # neurons_allExp_thisCre x 24(frames)
np.shape(all_sess_ns_thisCre)
#     all_sess_ns_fof_thisCre = all_sess_ns_fof_all_cre[icre] # neurons_allExp_thisCre x 24(frames)
#     np.shape(all_sess_ns_fof_thisCre)

x = embedding_all_cre[icre][:, 0]
y = embedding_all_cre[icre][:, 1]
#     z = embedding_all_cre[icre][:, 2]

amp_f = peak_amp_eachN_traceMed_flash_all_cre[icre]    
amp_o = peak_amp_eachN_traceMed_all_cre[icre]    

plt.figure()
plt.suptitle(f'{cre[:3]}', y=1.1, fontsize=14) 

############################################################
# no image; but omission ---> standard response
# x < 10; amp_o > amp_f
# no image; small omission ---> standard response
# x>10
############################################################
s = 1 # subplot

c1 = np.logical_and((x < 10), (amp_o > amp_f))
c2 = (x > 10)
c = np.logical_or(c1, c2)
# plt.plot(amp_f[c], 'b'); plt.plot(amp_o[c], 'r') # sanity check
st = f'{np.mean(c): .2f}\n{sum(c)}/{len(x)} neurons'
print(st)    

all_sess_ns_thisCre_thisCond = all_sess_ns_thisCre[c] # neurons_thisCre_thisCondition x 40(frames)    
np.shape(all_sess_ns_thisCre_thisCond)

# average 
plt.subplot(nr,nc,s)
top = np.mean(all_sess_ns_thisCre_thisCond, axis=0)
h1 = plt.plot(time_trace, top)
plot_flashLines_ticks_legend([], h1, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, bbox_to_anchor=bb, ylab=ylabel, xmjn=xmjn)
plt.title(st, fontsize=10)



############################################################
# large image; no omission ---> slc type 
# x < 10; amp_f > amp_o
############################################################
s = 2 # subplot

c1 = (x < 10)
c2 = (amp_f > amp_o)
c = np.logical_and(c1, c2)
# plt.plot(amp_f[c], 'b'); plt.plot(amp_o[c], 'r') # sanity check
st = f'{np.mean(c): .2f}\n{sum(c)}/{len(x)} neurons'
print(st)    

all_sess_ns_thisCre_thisCond = all_sess_ns_thisCre[c] # neurons_thisCre_thisCondition x 40(frames)    
np.shape(all_sess_ns_thisCre_thisCond)

# average 
plt.subplot(nr,nc,s)
top = np.mean(all_sess_ns_thisCre_thisCond, axis=0)
h1 = plt.plot(time_trace, top)
plot_flashLines_ticks_legend([], h1, flashes_win_trace_index_unq_time, grays_win_trace_index_unq_time, time_trace, bbox_to_anchor=bb, ylab=ylabel, xmjn=xmjn)
plt.title(st, fontsize=10)

plt.subplots_adjust(wspace=0.5)


if dosavefig:
    cre_short = cre[:3]
    nam = '%s_umap2D_manualClasses_allNeurSess%s_%s_%s' %(cre_short, whatSess, fgn, now)
    fign = os.path.join(dir0, dir_now, nam+fmt)        
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    

    
    
    
    
    