#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run "omissions_traces_peaks_plots_setVars.py" to set var "all_sess_2an" needed here.

Plot histogram of number of neurons per area and depth, for each cre line.

Created on Fri Aug 21 12:07:29 2020
@author: farzaneh
"""

dir_now = 'num_neurons'

#%% Set all_sess_2an for each cre line

cres = np.unique(all_sess_2an['cre'].values)
all_sess_2an_allcre = []
for icre in range(len(cres)):
    all_sess_2an_allcre.append(all_sess_2an[all_sess_2an['cre']==cres[icre]])

n_sess_pre_cre = [len(all_sess_2an_allcre[icre]) for icre in range(len(cres))]    
print(f'n_session_pre_cre: {n_sess_pre_cre}')



#%% Set number of neurons for each cre line

nn_plane_allcre = []
dd_plane_allcre = []
for icre in range(len(cres)): # icre = 0
    nn = all_sess_2an_allcre[icre]['n_neurons'].values
    dd = all_sess_2an_allcre[icre]['depth'].values
    aa = all_sess_2an_allcre[icre]['area'].values

    a = all_sess_2an_allcre[icre]
    
    aa = a['n_neurons'].values
    nn_plane = np.reshape(aa, (num_planes, int(len(aa)/num_planes)), order='F').astype(float) # 8 x num_sess
    
    aa = a['depth'].values
    dd_plane = np.reshape(aa, (num_planes, int(len(aa)/num_planes)), order='F').astype(float) # 8 x num_sess
    
    # below doesnt work; have no idea how it worked at some point!
    '''
    nn_plane = np.array([None] * nn[0].shape[0])
    dd_plane = np.array([None] * nn[0].shape[0])
    for ip in range(nn[0].shape[0]):
        nn_plane[ip] = np.array([nn[isess][ip] for isess in range(nn.shape[0])])
        dd_plane[ip] = np.array([dd[isess][ip] for isess in range(nn.shape[0])])
    '''
    nn_plane_allcre.append(nn_plane)
    dd_plane_allcre.append(dd_plane)



    
#%% Plot a histogram of number of neurons for each plane, of each cre line

for icre in range(len(cres)): # icre=0
    
    cre = cres[icre][:3]    
    nn_plane = nn_plane_allcre[icre]
#     mn = np.nanmin(nn_plane_allcre[icre])
#     mx = np.nanmax(nn_plane_allcre[icre])    
    
    plt.figure(figsize=(7,4))
    plt.suptitle(f'{cre}', y=1.1)
    
    # V1
    nn_plane_now = nn_plane[inds_v1] # 4 x num_sessions
    arean = 'V1'
    for ip in range(4): # ip=0     
        avn = np.nanmean(nn_plane_now[ip])
        avnt = avn.astype(int)
        
        mn = np.nanmin(nn_plane_allcre[icre][[ip, ip+num_depth],:])
        mx = np.nanmax(nn_plane_allcre[icre][[ip, ip+num_depth],:])
        
        plt.subplot(2,4,ip+1)
        plt.hist(nn_plane_now[ip]) # nn_plane_now
        
#         plt.title(f'{arean}, depth {ip+1}', fontsize=10)        
        plt.title(f'{arean}, depth {ip+1}\n{avnt} neurons', fontsize=10)        
#         plt.vlines(avn, 0, 10, 'r',':')
        plt.plot(avn, 0, color='r', marker='*', markersize=10)
        if ip==0:
            plt.ylabel('num sessions', fontsize=10)
        makeNicePlots(plt.gca())
        plt.gca().tick_params(labelsize=10)
        plt.xlim([mn, mx])
        
        
    # LM
    nn_plane_now = nn_plane[inds_lm]
    arean = 'LM'
    for ip in range(4):    
        avn = np.nanmean(nn_plane_now[ip])
        avnt = avn.astype(int)
        
        mn = np.nanmin(nn_plane_allcre[icre][[ip, ip+num_depth],:])
        mx = np.nanmax(nn_plane_allcre[icre][[ip, ip+num_depth],:])
        
        plt.subplot(2,4,4+ip+1)
        plt.hist(nn_plane_now[ip]) # nn_plane_now
        
#         plt.title(f'{arean}, depth {ip+1}', fontsize=10)        
        plt.title(f'{arean}, depth {ip+1}\n{avnt} neurons', fontsize=10)        
#         plt.vlines(avn, 0, 10, 'r',':')
        plt.plot(avn, 0, color='r', marker='*', markersize=10)
        if ip==0:
            plt.ylabel('num sessions', fontsize=10)
#         if ip==0:
        plt.xlabel('num neurons', fontsize=10)            
        makeNicePlots(plt.gca())
        plt.gca().tick_params(labelsize=10)
        plt.xlim([mn, mx])
        
    plt.subplots_adjust(wspace=.5, hspace=.8)
    
    
    if dosavefig:
        nam = 'numNeurons_%s%s_hist_%s' %(cre, whatSess, now)        
        fign = os.path.join(dir0, dir_now, nam+fmt)        
        plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    

        
        
#%% Plot a histogram of depth for each plane, of each cre line

plt.figure(figsize=(7,7))
for icre in range(len(cres)):
    
    cre = cres[icre][:3]
#     plt.suptitle(f'{cre}', y=1.1)
    nn_plane = dd_plane_allcre[icre]
    
    # V1: V1 and LM are the same in terms of depth, so we just plot V1 depth.
    nn_plane_now = nn_plane[inds_v1]
#     arean = 'V1' # 
    for ip in range(4):    
        avn = np.nanmean(nn_plane_now[ip])
        avnt = avn.astype(int)
        
        plt.subplot(3,4, icre*4 + ip+1)
        plt.hist(nn_plane_now[ip]) # nn_plane_now
        
        plt.title(f'{cre}\nplane {ip+1}, {avnt} μm', fontsize=10)   
#         plt.title(f'{arean}, depth {ip+1}\n{avnt} neurons', fontsize=10)        
#         plt.vlines(avn, 0, 10, 'r',':')
        plt.plot(avn, 0, color='r', marker='*', markersize=10)
        if ip==0:
            plt.ylabel('num sessions', fontsize=10)
        plt.xlabel('Depth (μm)', fontsize=10)
        makeNicePlots(plt.gca())
        plt.gca().tick_params(labelsize=10)
        
plt.subplots_adjust(wspace=.5, hspace=.8)


if dosavefig:
    nam = 'depth_allCre%s_hist_%s' %(whatSess, now)        
    fign = os.path.join(dir0, dir_now, nam+fmt)        
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    
        
        