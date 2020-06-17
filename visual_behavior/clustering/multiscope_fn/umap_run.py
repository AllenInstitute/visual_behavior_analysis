#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vars needed here are set in umap_setVars.py

The main variable used here is all_sess_now.

This script runs umap/pca, and makes plots.


Created on Thu May 14 21:27:25 2020
@author: farzaneh
"""

    
################################################################################################    
################################################################################################        
#%% Run PCA
################################################################################################    
################################################################################################    

from sklearn.decomposition import PCA
varexpmax = .99 # 1 # .9

pc_all_cre = []

for icre in range(len(cre_lines)): # icre=0
    cre = cre_lines[icre]    
    all_sess_ns_fof_thisCre = all_sess_ns_fof_all_cre[icre] # neurons_allExp_thisCre x 24(frames)
    print(f'Running PCA on {cre}, matrix size: {np.shape(all_sess_ns_fof_thisCre)}')

    x_train_pc, pca = doPCA(all_sess_ns_fof_thisCre, varexpmax=varexpmax, doplot=1)
    x_train_pc.shape

    pc_all_cre.append(x_train_pc)



#%% Scatter plots for each cre line; 2 subplots, each colored for flash and omission responses.

cut_axes = 1 # if 1, show from 1 to 99 percentile of x and y values. helps when there are some outliers.
plot_scatter_fo(cre_lines, pc_all_cre, peak_amp_eachN_traceMed_flash_all_cre, peak_amp_eachN_traceMed_all_cre, lab_analysis='PCA', cut_axes=cut_axes, same_norm_fo=1, dosavefig=dosavefig)


    
################################################################################################    
################################################################################################    
#%% Run umap on all_sess_ns_fof_thisCre
################################################################################################
################################################################################################

import umap    
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

ncomp = 2 # 3 # number of umap components

embedding_all_cre = []

for icre in range(len(cre_lines)): # icre = 2    
    cre = cre_lines[icre]    
    all_sess_ns_fof_thisCre = all_sess_ns_fof_all_cre[icre] # neurons_allExp_thisCre x 24(frames)
    print(f'Running UMAP on {cre}')

    sp = 2
    neigh = 7
    embedding = umap.UMAP(spread= sp, n_neighbors = neigh, n_components = ncomp).fit_transform(all_sess_ns_fof_thisCre)
    print(f'embedding size: {embedding.shape}')
    embedding_all_cre.append(embedding)
    

# embedding_all_cre_3d = copy.deepcopy(embedding_all_cre)




###############################################################
###############################################################
###########################  Plots  ###########################
###############################################################
###############################################################
# from matplotlib import cm # colormap

#%% Scatter plots for each cre line; 2 subplots, each colored for flash and omission responses.                

def plot_scatter_fo(cre_lines, low_dim_all_cre, peak_amp_eachN_traceMed_flash_all_cre, peak_amp_eachN_traceMed_all_cre, lab_analysis='UMAP', cut_axes=0, same_norm_fo=1, dosavefig=0):
#     same_norm_fo = 1 # use the same cmap scale for image and omission; otherwise plot each using its own min and max
#     cut_axes = 1 # if 1, show from 1 to 99 percentile of x and y values. helps when there are some outliers.

    for icre in range(len(cre_lines)): # icre = 2

        cre = cre_lines[icre]

        x = low_dim_all_cre[icre][:, 0]
        y = low_dim_all_cre[icre][:, 1]
    #     z = low_dim_all_cre[icre][:, 2]

        mnx = np.percentile(x, .5)
        mny = np.percentile(y, .5)    
        mxx = np.percentile(x, 99.5)
        mxy = np.percentile(y, 99.5)
        
        if same_norm_fo: # use the same cmap scale for image and omission
            amp = np.concatenate((peak_amp_eachN_traceMed_flash_all_cre[icre], peak_amp_eachN_traceMed_all_cre[icre]))
            mn = np.percentile(amp, 1)
            mx = np.percentile(amp, 99)
            norm = matplotlib.colors.Normalize(vmin=mn, vmax=mx)


        fig = plt.figure(figsize=(14,5)) #(figsize=(10,3))
        plt.suptitle(f'{cre[:3]}', y=.995, fontsize=22) 

        #########################################
        ###### plot flash-evoked responses ######
        #########################################
        amp = peak_amp_eachN_traceMed_flash_all_cre[icre]    
        clab = 'image-evoked amplitude'

        ax3D = fig.add_subplot(121)
        # ax3D = fig.add_subplot(111, projection='3d')

        if same_norm_fo==0: # use different cmap scales for image and omission
            mn = np.percentile(amp, 1)
            mx = np.percentile(amp, 99)
            norm = matplotlib.colors.Normalize(vmin=mn, vmax=mx)

        # each neuron is colored according to its omission or flash response
        ax3D.scatter(x, y, s=10, label=cre[:3], marker='o', c=amp, cmap=cmap, norm=norm) #cmap=cm.jet) 

        ax3D.set_xlabel(f'{lab_analysis} axis 1 (arbitrary Units)')
        ax3D.set_ylabel(f'{lab_analysis} axis 2')
    #     ax3D.set_zlabel('UMAP Axis 3 ')    
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3D, label=clab) 
    #     ax3D.title(cre[:3])
        plt.title('Image')
        ax3D.set_aspect('equal')

        if cut_axes:
            plt.xlim([mnx, mxx])
            plt.ylim([mny, mxy])
        
        else:
            xmj = np.unique(np.concatenate((np.arange(0, min(x), -5), np.arange(0, max(x), 5)))) # (max(x)-min(x))/5)
        #     xmn = np.arange(.25, x[-1], .5)
            ax3D.set_xticks(xmj)
            ax3D.set_yticks(xmj)
            ax3D.grid(True, which='both') # major



        ############################################
        ###### plot omission-evoked responses ######
        ############################################
        amp = peak_amp_eachN_traceMed_all_cre[icre]    
        clab = 'omission-evoked amplitude'

        ax3D = fig.add_subplot(122)
        # ax3D = fig.add_subplot(111, projection='3d')

        if same_norm_fo==0: # use different cmap scales for image and omission
            mn = np.percentile(amp, 1)
            mx = np.percentile(amp, 99)
            norm = matplotlib.colors.Normalize(vmin=mn, vmax=mx)

        # each neuron is colored according to its omission or flash response
        ax3D.scatter(x, y, s=10, label=cre[:3], marker='o', c=amp, cmap=cmap, norm=norm) #cmap=cm.jet) 

        ax3D.set_xlabel(f'{lab_analysis} axis 1 (arbitrary Units)')
        ax3D.set_ylabel(f'{lab_analysis} axis 2')
    #     ax3D.set_zlabel('UMAP Axis 3 ')    
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3D, label=clab) 
    #     ax3D.title(cre[:3])    
        plt.title('Omission')
        ax3D.set_aspect('equal')

        if cut_axes:
            plt.xlim([mnx, mxx])
            plt.ylim([mny, mxy])
        
        else:        
            xmj = np.unique(np.concatenate((np.arange(0, min(x), -5), np.arange(0, max(x), 5)))) # (max(x)-min(x))/5)
        #     xmn = np.arange(.25, x[-1], .5)
            ax3D.set_xticks(xmj)
            ax3D.set_yticks(xmj)
            ax3D.grid(True, which='both') # major

        plt.subplots_adjust(wspace=0.5)

        

        if dosavefig:
            cre_short = cre[:3]
            fgn_umap = f'{lab_analysis}_scatter_imageOmit_allNeurSess'
    #         if useSDK:
    #             fgn_umap = f'{fgn_umap}_sdk_'
            nam = '%s_%s%s_%s_%s' %(cre_short, fgn_umap, whatSess, fgn, now)
            fign = os.path.join(dir0, dir_now, nam+fmt)     

            plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    



            
            
#%% Make scatter plots; all cre lines superimposed 

color_cre_omit_flash = 3 # if 1: color neurons according to cre line; if 2, according to omission responses; if 3, according to flash responses.

cols_cre = ['b', 'r', 'g'] # slc, sst, vip
cmap = plt.cm.jet #bwr #or any other colormap 

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
ax3D = fig.add_subplot(111)
# ax3D = fig.add_subplot(111, projection='3d')

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
    
#     ax3D = fig.add_subplot(1,3,icre+1, projection='3d')    
#     ax3D.set_title(cre)
#     ax3D.scatter(x, y, z, s=10, c=cols_cre[icre], label=cre[:3], marker='o') 

    if color_cre_omit_flash==1:
        ax3D.scatter(x, y, s=10, c=cols_cre[icre], label=cre[:3], marker='o')
        ax3D.legend()

    else:     # each neuron is colored according to its omission or flash response
        ax3D.scatter(x, y, s=10, label=cre[:3], marker='o', c=amp, cmap=cmap, norm=norm) #cmap=cm.jet) 
    
    if icre==0:
        ax3D.set_xlabel('UMAP Axis 1 (Arbitrary Units)', )
        ax3D.set_ylabel('UMAP Axis 2')
    #     ax3D.set_zlabel('UMAP Axis 3 ')
        if color_cre_omit_flash!=1:
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3D, label=clab) 
    
    ax3D.set_aspect('equal')
    
    
   

#%% Scatter plots for each cre line; 2 subplots, each colored for flash and omission responses.

plot_scatter_fo(cre_lines, embedding_all_cre, peak_amp_eachN_traceMed_flash_all_cre, peak_amp_eachN_traceMed_all_cre, lab_analysis='UMAP', cut_axes=0, same_norm_fo=1, dosavefig=dosavefig)

                

    
    
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

    
    
    
    
    