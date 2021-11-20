'''
Plot histograms of cc across all neurons pairs.
Vars are set in omissions_traces_peaks_plots_setVars_corr.py

Created on Fri Nov 19 11:41:29 2021
@author: farzaneh

'''

#### Plot histogram of correlation coefficients across all neuron pairs (pooled experiments) for each mouse

# V1-LM corrs
toplot = 'cc12_peak_amp_omit_allPairs_allSess'
toplotf = 'cc12_peak_amp_flash_allPairs_allSess'
ll = len(layerPairs_a12)
ctype = 'V1-LM'

# V1-V1 corrs
toplot = 'cc22_peak_amp_omit_allPairs_allSess'
toplotf = 'cc22_peak_amp_flash_allPairs_allSess'
ll = len(layerPairs_aSame)
ctype = 'V1-V1'


# LM-LM corrs
toplot = 'cc11_peak_amp_omit_allPairs_allSess'
toplotf = 'cc11_peak_amp_flash_allPairs_allSess'
ll = len(layerPairs_aSame)
ctype = 'LM-LM'


plt.figure(figsize=(10, 18))

nmice = len(corr_trace_peak_allMice)

for im in range(nmice):
    cre = corr_trace_peak_allMice['cre'].iloc[im]
    nsess = len(corr_trace_peak_allMice[toplot].iloc[im])    
#     for isess in range(nsess):
#         for ilc in range(16):
#             print(np.shape(corr_trace_peak_allMice[toplot].iloc[im][isess][ilc]))

    # pool neurons pairs across all layer pairs for all sessions of a given mouse    
    np_allLp_allSess = np.hstack(np.hstack([corr_trace_peak_allMice[toplot].iloc[im][isess][ilc] for ilc in range(ll)]) for isess in range(nsess))
    np_allLp_allSess_f = np.hstack(np.hstack([corr_trace_peak_allMice[toplotf].iloc[im][isess][ilc] for ilc in range(ll)]) for isess in range(nsess))

    o = np.nanmean(np_allLp_allSess)
    f = np.nanmean(np_allLp_allSess_f)
    
    t = f'{cre[:3]}\n{o/f:.3f}, {o:.3f}, {f:.3f}\n'
    print(t)
    
    if len(np_allLp_allSess)>0:
        plt.subplot(nmice/3, 3, im+1)
        
        n1,bins,patches = plt.hist(np_allLp_allSess, facecolor='r')
        n2,bins,patches = plt.hist(np_allLp_allSess_f, facecolor='b')
        
        n = max(max(n1), max(n2))
        m = n+n/10
        
        plt.plot(o, m, 'rX')
        plt.plot(f, m, 'bX')
        
        plt.xlabel('$\Delta$ cc')
        if np.remainder(im+1,3)==1:
            plt.ylabel('# neuron pairs')
        plt.title(cre[:3]);            
#         plt.title(t);
    else:
        print(f"mouse {corr_trace_peak_allMice['mouse_id'].iloc[im]} has empty array of neuron pairs")
    
plt.suptitle(ctype, fontsize=20, y=.93)
plt.subplots_adjust(hspace=1.5, wspace=.8)
    
    
if dosavefig:
    nam = f'allMice_{ctype}_ccDist_allExpPooled_{fgn}{whatSexx}_{now}'
    fign = os.path.join(dir0, dir_now, nam+fmt)        
    
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,) 
    
    
    
    
    
###############################################################
#%% 3 plots for each cre line; pool mice for each cre line

# V1-LM corrs
toplot = 'cc12_peak_amp_omit_allPairs_allSess'
toplotf = 'cc12_peak_amp_flash_allPairs_allSess'
ll = len(layerPairs_a12)
ctype = 'V1-LM'


# V1-V1 corrs
toplot = 'cc22_peak_amp_omit_allPairs_allSess'
toplotf = 'cc22_peak_amp_flash_allPairs_allSess'
ll = len(layerPairs_aSame)
ctype = 'V1-V1'


# LM-LM corrs
toplot = 'cc11_peak_amp_omit_allPairs_allSess'
toplotf = 'cc11_peak_amp_flash_allPairs_allSess'
ll = len(layerPairs_aSame)
ctype = 'LM-LM'


plt.figure(figsize=(3, 10))

icre = 0
for cre in corr_trace_peak_allMice['cre'].unique(): # cre = 'Vip' # ['Vip', 'Sst', 'Slc17a7']
    icre = icre+1
    dfn = corr_trace_peak_allMice[corr_trace_peak_allMice['cre']==cre]
    nmice = len(dfn)
    print(nmice)

    np_allLp_allSess_all = []
    np_allLp_allSess_f_all = []
    for im in range(nmice):
        nsess = len(dfn[toplot].iloc[im])    
        
        # pool neurons pairs across all layer pairs for all sessions of a given mouse    
        np_allLp_allSess = np.hstack(np.hstack([dfn[toplot].iloc[im][isess][ilc] for ilc in range(ll)]) for isess in range(nsess))
        np_allLp_allSess_f = np.hstack(np.hstack([dfn[toplotf].iloc[im][isess][ilc] for ilc in range(ll)]) for isess in range(nsess))

        np_allLp_allSess_all.append(np_allLp_allSess)
        np_allLp_allSess_f_all.append(np_allLp_allSess_f)
    
    oo = np.concatenate((np_allLp_allSess_all))    
    ff = np.concatenate((np_allLp_allSess_f_all))    
    
    o = np.nanmean(oo)
    f = np.nanmean(ff)
    
#     t = f'{cre[:3]}\n{o/f:.3f}, {o:.3f}, {f:.3f}\n'
    t = f'{cre[:3]}\n{o-f:.3f}, {o:.3f}, {f:.3f}\n'
    
    ##### plot
    plt.subplot(3,1,icre)

#     n1,bins,patches = plt.hist(oo, facecolor='r')
#     n2,bins,patches = plt.hist(ff, facecolor='b')
#     n = max(max(n1), max(n2))

    # plot the difference in cc: omission - image
    n0,bins,patches = plt.hist(oo-ff, facecolor='b')
    n = max(n0)
    
    m = n+n/10

    plt.plot(o, m, 'rX')
    plt.plot(f, m, 'bX')

    plt.xlabel('$\Delta$ cc')        
    plt.ylabel('# neuron pairs')
#     plt.title(cre[:3]);            
    plt.title(t);
    
plt.suptitle(ctype, fontsize=20, y=1)    
plt.subplots_adjust(hspace=1.2)
    
    
# add to figure name, which area-area interactions it is, and if it's omission or image or their difference.    
if dosavefig:
    nam = f'allCre_{ctype}_ccDist_allMiceExpPooled_{fgn}{whatSexx}_{now}'
    fign = os.path.join(dir0, dir_now, nam+fmt)        
    
    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,) 



#%% Sanity check for how _allPairs and _sessAv vars relate to each other:
# the following 2 are identical
# 1. get average of neuron pairs, per session (s), compute each for each layer(l)
# np.reshape([np.nanmean([np.nanmean(corr_trace_peak_allMice[toplot].iloc[6][s][l]) for s in range(5)]) for l in range(16)], (4,4), order='C')
# 2. get session-averaged neuron-pair-averaged values for each layer
# corr_trace_peak_allMice['cc12_peak_amp_flash_sessAv'].iloc[6]

    
'''
im = 0
np.concatenate((corr_trace_peak_allMice['cc12_peak_amp_flash_allPairs_allSess'].iloc[im])).shape

np.shape(corr_trace_peak_allMice['cc12_peak_amp_flash_allPairs_allSess'].iloc[im]) # sessions x layer_pairs

isess = 0
for ilc in range(16):
    a = corr_trace_peak_allMice['cc12_peak_amp_flash_allPairs_allSess'].iloc[im][isess][ilc] # neuron pairs

    plt.hist(a)
    plt.xlabel('$\Delta$ correlation coefficient')
    plt.ylabel('# neuron pairs')


for isess in range(len(corr_trace_peak_allMice['cc12_peak_amp_flash_allPairs_allSess'].iloc[im])):
    # pool neurons pairs across all layer pairs for a given session    
    np_allLp_eachSess = np.hstack([corr_trace_peak_allMice['cc12_peak_amp_flash_allPairs_allSess'].iloc[im][isess][ilc] for ilc in range(16)])

    plt.hist(np_allLp_eachSess)
    plt.xlabel('$\Delta$ correlation coefficient')
    plt.ylabel('# neuron pairs')

'''    