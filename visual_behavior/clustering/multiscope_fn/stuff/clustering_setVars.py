#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vars needed here are set in omissions_traces_peaks_plots_setVars.py

Created on Thu May 14 21:27:25 2020
@author: farzaneh
"""

flash_ind = abs(flashes_win_trace_index_unq[flashes_win_trace_index_unq<0][-1].astype(int))

# take one image before and one image after omission
samps_bef_now = samps_bef - flash_ind # image before omission
samps_aft_now = samps_bef + flash_ind*2 # omission and the image after

bl_index_pre_omit = np.arange(0,samps_bef) # we will need it for response amplitude quantification.
    
    
#%% Remove sessions without any omissions

# identify sessions with no omissions
o = all_sess_2an['n_omissions'].values.astype(float) # if n_omissions is nan, that's because the experiment is invalid
sess_no_omit = np.unique(all_sess_2an[o==0]['session_id'].values)

# remove sessions with no omissions
all_sess_no_omit = all_sess_2an[~np.in1d(all_sess_2an['session_id'].values, sess_no_omit)]
all_sess_no_omit.shape[0]/8.


# remove invalid experiments
o = all_sess_no_omit['valid'].values # if valid is 0, that's because the experiment is invalid or there are no neurons in that plane.
print(f'Removing {sum(o==0)} invalid experiments!')
all_sess_now = all_sess_no_omit[o==1]
print(len(all_sess_now))


#%% Get traces for all mice (all sessions, all neurons concatenated) of a specific cre line

cre_all = all_sess_now['cre'].values # len_mice_withData
cre_lines = np.unique(cre_all)

# loop through each cre line
all_sess_ns_fof_all_cre = [] # size: number of distinct cre lines; # includes image before omission, omission, and image after omission (24 frames) # use samps_bef_now to find omission time.
bl_preOmit_all_cre = []
all_sess_ns_all_cre = [] # 40 frames; use samps_bef to find omission time.

for icre in range(len(cre_lines)): # icre = 0
    
    cre = cre_lines[icre]    
    thisCre_numMice = sum(cre_all==cre) # number of mice that are of line cre
    print(f'\n~{thisCre_numMice/8.} sessions for {cre} mice')

    all_sess_thisCre = all_sess_now.iloc[cre_all==cre]    
#     print(all_sess_thisCre.shape)

    # note: local_fluo_allOmitt for invalid experiments will be a nan trace as if they had one neuron
    a = all_sess_thisCre['local_fluo_allOmitt'].values # each element is frames x units x trials
#     a.shape

    ############### take the average of omission-aligned traces across trials; also transpose so the traces are neurons x frames
    aa = [np.nanmean(a[i], axis=2).T for i in range(len(a))] # each element is neurons x frames
#     np.shape(aa)

    ############### concantenate trial-averaged neuron traces from all sessions
    all_sess_ns = np.vstack(aa) # neurons_allSessions_thisCre x 24(frames)
#     all_sess_ns.shape
#     plt.plot(np.nanmean(all_sess_ns, axis=0))

    ############### compute baseline (on trial-averaged traces); we need it for quantifying flash/omission evoked responses.
#     bl_index_pre_flash = np.arange(0,samps_bef) # we are computing flash responses on flash aligned traces.
    bl_preOmit = np.percentile(all_sess_ns.T[bl_index_pre_omit,:], bl_percentile, axis=0) # neurons # use the 10th percentile
#     bl_preFlash = np.percentile(all_sess_ns.T[bl_index_pre_flash,:], bl_percentile, axis=0) # neurons

    ############### take one image before and one image after omission
    all_sess_ns_fof_thisCre = all_sess_ns[:, samps_bef_now:samps_aft_now] # df/f contains a sequence of image, omission and image
    print(f'{all_sess_ns_fof_thisCre.shape}: size of neurons_allSessions_thisCre') # neurons_allSessions_thisCre x 24(frames)
#     plt.plot(np.nanmean(all_sess_ns_fof_thisCre, axis=0))
    
    # keep arrays for all cres
    all_sess_ns_fof_all_cre.append(all_sess_ns_fof_thisCre) # each element is # neurons_allSessions_thisCre x 24(frames)
    bl_preOmit_all_cre.append(bl_preOmit) # each element is # neurons_allSessions_thisCre
    all_sess_ns_all_cre.append(all_sess_ns)
    
    
 
#%% Compute flash and omission-evoked responses on trial-averaged traces in all_sess_ns_fof_thisCre (relative to baseline)

from omissions_traces_peaks_quantify import *

peak_win = [0, .75] # this should be named omit_win
flash_win = [-.75, -.25] # flash responses are computed on omission-aligned traces; # [-.75, 0] 

flash_index = samps_bef_now + np.round(-.75 / frame_dur).astype(int)

peak_amp_eachN_traceMed_all_cre = []
peak_timing_eachN_traceMed_all_cre = []
peak_amp_eachN_traceMed_flash_all_cre = []
peak_timing_eachN_traceMed_flash_all_cre = []

for icre in range(len(cre_lines)): # icre = 0

    cre = cre_lines[icre]    
    all_sess_ns_fof_thisCre = all_sess_ns_fof_all_cre[icre] # neurons_allSessions_thisCre x 24(frames)
    bl_preOmit = bl_preOmit_all_cre[icre] # neurons_allSessions_thisCre
    bl_preFlash = bl_preOmit

    # the input traces below need to be frames x neurons
    # output size: # neurons_allSessions_thisCre
    # flash-evoked responses are computed on omission aligned traces.
    peak_amp_eachN_traceMed, peak_timing_eachN_traceMed, peak_amp_eachN_traceMed_flash, peak_timing_eachN_traceMed_flash, \
        peak_allTrsNs, peak_timing_allTrsNs, peak_om_av_h1, peak_om_av_h2, auc_peak_h1_h2 = \
                omissions_traces_peaks_quantify \
    (all_sess_ns_fof_thisCre.T, bl_preOmit, bl_preFlash, mean_notPeak, cre, peak_win, flash_win, flash_win_timing, flash_index, samps_bef_now, frame_dur, doShift_again, doPeakPerTrial=0, doROC=0, doPlots=0, index=0, local_fluo_allOmitt=0, num_omissions=0)

    
    peak_amp_eachN_traceMed_all_cre.append(peak_amp_eachN_traceMed)
    peak_timing_eachN_traceMed_all_cre.append(peak_timing_eachN_traceMed)
    peak_amp_eachN_traceMed_flash_all_cre.append(peak_amp_eachN_traceMed_flash)
    peak_timing_eachN_traceMed_flash_all_cre.append(peak_timing_eachN_traceMed_flash)

    

#%% Run umap on all_sess_ns_fof_thisCre

import umap    
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

ncomp = 2 # 3 # number of umap components

embedding_all_cre = []
for icre in range(len(cre_lines)): # icre = 2    
    cre = cre_lines[icre]    
    all_sess_ns_fof_thisCre = all_sess_ns_fof_all_cre[icre] # neurons_allSessions_thisCre x 24(frames)
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
    
    
   

#%% Scatter plots for each cre lines; 2 subplots: flash and omission responses

same_norm_fo = 1 # use the same cmap scale for image and omission; otherwise plot each using its own min and max
    
for icre in range(len(cre_lines)): # icre = 0
    
    cre = cre_lines[icre]

    x = embedding_all_cre[icre][:, 0]
    y = embedding_all_cre[icre][:, 1]
#     z = embedding_all_cre[icre][:, 2]

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
    
    ax3D.set_xlabel('UMAP Axis 1 (Arbitrary Units)', )
    ax3D.set_ylabel('UMAP Axis 2')
#     ax3D.set_zlabel('UMAP Axis 3 ')    
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3D, label=clab) 
#     ax3D.title(cre[:3])
    plt.title('Image')
    ax3D.set_aspect('equal')
    
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
    
    ax3D.set_xlabel('UMAP Axis 1 (Arbitrary Units)', )
    ax3D.set_ylabel('UMAP Axis 2')    
#     ax3D.set_zlabel('UMAP Axis 3 ')    
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3D, label=clab) 
#     ax3D.title(cre[:3])    
    plt.title('Omission')
    ax3D.set_aspect('equal')
    
    xmj = np.unique(np.concatenate((np.arange(0, min(x), -5), np.arange(0, max(x), 5)))) # (max(x)-min(x))/5)
#     xmn = np.arange(.25, x[-1], .5)
    ax3D.set_xticks(xmj)
    ax3D.set_yticks(xmj)
    ax3D.grid(True, which='both') # major
    
    plt.subplots_adjust(wspace=0.5)

    
    if dosavefig:
        cre_short = cre[:3]
        nam = '%s_umap2D_scatter_imageOmit_allNeurSess%s_%s_%s' %(cre_short, whatSess, fgn, now)
        fign = os.path.join(dir0, dir_now, nam+fmt)        
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    


    
#%% Manually choose subclasses for each cell line and plot their average traces.

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

all_sess_ns_thisCre = all_sess_ns_all_cre[icre] # neurons_allSessions_thisCre x 24(frames)
np.shape(all_sess_ns_thisCre)
#     all_sess_ns_fof_thisCre = all_sess_ns_fof_all_cre[icre] # neurons_allSessions_thisCre x 24(frames)
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

all_sess_ns_thisCre = all_sess_ns_all_cre[icre] # neurons_allSessions_thisCre x 24(frames)
np.shape(all_sess_ns_thisCre)
#     all_sess_ns_fof_thisCre = all_sess_ns_fof_all_cre[icre] # neurons_allSessions_thisCre x 24(frames)
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

all_sess_ns_thisCre = all_sess_ns_all_cre[icre] # neurons_allSessions_thisCre x 24(frames)
np.shape(all_sess_ns_thisCre)
#     all_sess_ns_fof_thisCre = all_sess_ns_fof_all_cre[icre] # neurons_allSessions_thisCre x 24(frames)
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

    
    
    
    
    