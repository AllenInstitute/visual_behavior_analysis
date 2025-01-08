'''
# Define functions to set an "active trace" (referred to as "traces_evs" below), i.e. 
# a trace made by extracting and concatenating the active parts of the input trace.
# Farzaneh Najafi
# March 2020

# Example call to the function:

## set the input trace (neurons x frames)
plane_ind = 0
traces_y0 = this_sess.iloc[plane_ind]['local_fluo_traces']
traces_y0.shape # neurons x frames

len_ne = 20 # number of frames before and after each event that are taken to create traces_events.
th_ag = 10 #8 # threshold to apply on erfc (output of evaluate_components) to find events on the trace; the higher the more strict on what we call an event.
doPlots = 1 # set to 1 to see an example neuron

## call the function to set traces_evs (for the y trace), ie traces that are made by extracting the active parts of the input trace
# the idea is that to help with learning the events, take parts of the trace that have events        
[traces_y0_evs, inds_final_all] = set_traces_evs(traces_y0, th_ag, len_ne, doPlots)

'''

#%% Set traces_evs (for the y trace), ie traces that are made by extracting the active parts of the input trace
# the idea is that to help with learning the events, take parts of the trace that have events

def set_traces_evs(traces_y0, th_ag, len_ne, doPlots=1):
    
#    len_ne = len_win #20 # number of frames before and after each event that are taken to create traces_events.    
#    th_ag = 10 #8 # the higher the more strict on what we call an event.
    
    ###############################################################################
    ############# Andrea's method of identifying "exceptional" events. ############
    ###############################################################################
    [idx_components, fitness, erfc] = evaluate_components(traces_y0, N=5, robust_std=False)
    erfc = -erfc
    
    #erfc.shape
    
    '''
    p1 = traces_y0
    p2 = -erfc

#    p1 = supimpose(traces_y0)
#    p2 = supimpose(-erfc)

    plt.figure()
    plt.plot(p1,'b')
    plt.plot(p2,'r')
    '''
    
    # seems like I can apply a threshold of ~10 on erfc to find "events"
    evs = (erfc >= th_ag) # neurons x frames
    #traces_y0_evs = np.full(evs.shape, np.nan)
    #traces_y0_evs[evs] = traces_y0[evs]
    
    # take a look at the fraction of time points that have calcium events.
    evs_num = np.sum(evs,axis=1) # number of time points with high df/f values for each neuron
    evs_fract = np.mean(evs,axis=1)
    
    if doPlots:
        get_ipython().magic(u'matplotlib inline')
        
        plt.figure(figsize=(4,3)); 
        plt.plot(evs_fract); plt.xlabel('Neuron in Y trace'); plt.title('Fraction of time points with high ("active") DF/F')
        makeNicePlots(plt.gca())
    
    ##################################################################
    ############ find gaps between events for each neuron ############
    ##################################################################    
    [gap_evs_all, begs_evs, ends_evs, gap_evs, bgap_evs, egap_evs, begs, ends] = find_event_gaps(evs)
    # np.equal(begs_evs[neuron_y],   ends_evs[neuron_y] + gap_evs[neuron_y])
    # begs[1,begs[0]==neuron_y]

    ##### Note: begs_evs does not include the first event; so index i in gap_evs corresponds to index i in begs_evs and ends_evs.
    ##### however, gaps are truely gaps, ie number of frames without an event that span the interval between two frames.
#    gap_evs_all # includes the gap before the 1st event and the gap before the last event too, in addition to inter-event gaps        
#    gap_evs # includes only gap between events 
#    begs_evs # index of event onsets, excluding the 1st events. (in fact these are 1 index before the actual event onset; since they are computed on the difference trace ('d'))
#    ends_evs # index of event offsets, excluding the last event. 
#    bgap_evs # number of frames before the first event
#    egap_evs # number of frames after the last event    
    
    '''
    # old stuff
    # if the gap between two "events" is < 1 sec, extend the events in traces_y0_evs 
    # this is to take care of the rise of the ca events bc erfc during rise is low, and it is only high after the rise  
    
    # find the index of event offset that preceded the gaps shorter than 10 frames on the trace
    # Note: begs_evs does not include the first event; so index i in gap_evs corresponds to index i in begs_evs and ends_evs.
    # the following two are the same:
    ends_evs[neuron_y][gap_evs[neuron_y] <= 10] + gap_evs[neuron_y][gap_evs[neuron_y] <= 10]
    begs_evs[neuron_y][gap_evs[neuron_y] <= 10]
    '''
        
    #############################################################################################################
    ############ set traces_evs, ie a trace that contains mostly the active parts of the input trace ############
    #############################################################################################################    
    traces_y0_evs = []
    inds_final_all = []
    
    for iu in range(traces_y0.shape[0]):
    
        # loop through each event, and take 20 frames before until 20 frames after the event ... we want to have a good sample of non-events too for training the model.
        #bnow = begs[1, begs[0]==iu]
        #enow = ends[1, ends[0]==iu]
        enow = ends_evs[iu]
        bnow = begs_evs[iu]
        e_aft = []
        b_bef = []
        for ig in range(len(bnow)):
        #    ev_sur.append(np.arange(bnow[ig]-len_ne, enow[ig]+len_ne+1))
            e_aft.append(np.arange(enow[ig], min(evs.shape[1], enow[ig]+len_ne)))
            b_bef.append(np.arange(bnow[ig]+1-len_ne, min(evs.shape[1], bnow[ig]+2))) # +2 because bnow[ig] is actually the frame preceding the event onset, so +1 here, and another +1 because range doesn't include the last element. # anyway the details dont matter bc we will do unique later.
        
        e_aft = np.array(e_aft) # n_gap x len_ne
        b_bef = np.array(b_bef)
        
        if len(e_aft)>1:
            e_aft_u = np.hstack(e_aft)
        else:
            e_aft_u = []
            
        if len(b_bef)>1:            
            b_bef_u = np.hstack(b_bef)
        else:
            b_bef_u = []
        #e_aft.shape, b_bef_u.shape
        
        # below sets frames that cover the duration of all events, but excludes the first and last event
        ev_dur = []
        for ig in range(len(bnow)-1):
            ev_dur.append(np.arange(bnow[ig], enow[ig+1]))
        
        ev_dur = np.array(ev_dur)
        
        if len(ev_dur)>1:            
            ev_dur_u = np.hstack(ev_dur)
        else:
            ev_dur_u = []
        #ev_dur_u.shape
        
        
        evs_inds = np.argwhere(evs[iu]).flatten() # includes ALL events.   
        # now take care of the 1st and the last event
        #bgap = [begs_this_n[0] + 1] # after how many frames the first event happened
        #egap = [evs.shape[1] - ends_this_n[-1] - 1] # how many frames with no event exist after the last event
        if len(bgap_evs[iu])>0:
            # get len_ne frames before the 1st event
            ind1 = np.arange(np.array(bgap_evs[iu])-len_ne, bgap_evs[iu])
            # if the 1st event is immediately followed by more events, add those to ind1, because they dont appear in any of the other vars that we are concatenating below.
            if len(ends_evs[iu])>1:
                ii = np.argwhere(np.in1d(evs_inds, ends_evs[iu][0])).squeeze()
                ind1 = np.concatenate((ind1, evs_inds[:ii]))            
        else: # first event was already going when the recording started; add these events to ind1
            jj = np.argwhere(np.in1d(evs_inds, ends_evs[iu][0])).squeeze()
#            jj = ends_evs[iu][0]
            ind1 = evs_inds[:jj+1]
            
            
        if len(egap_evs[iu])>0:
            # get len_ne frames after the last event
            indl = np.arange(evs.shape[1]-np.array(egap_evs[iu])-1, min(evs.shape[1], evs.shape[1]-np.array(egap_evs[iu])+len_ne))
            # if the last event is immediately preceded by more events, add those to indl, because they dont appear in any of the other vars that we are concatenating below.
            if len(begs_evs[iu])>1:
                ii = np.argwhere(np.in1d(evs_inds, 1+begs_evs[iu][-1])).squeeze() # find the fist event of the last event bout
                indl = np.concatenate((evs_inds[ii:], indl))
        else: # last event was already going when the recording ended; add these events to ind1
            jj = np.argwhere(np.in1d(evs_inds, begs_evs[iu][-1]+1)).squeeze()
            indl = evs_inds[jj:]
        
        #ind1, indl
        
        inds_final = np.unique(np.concatenate((e_aft_u, b_bef_u, ev_dur_u, ind1, indl))).astype(int)
        
        if np.in1d(evs_inds, inds_final).all()==False: # all evs_inds must exist in inds_final, otherwise something is wrong!
            if np.array([len(e_aft)>1, len(b_bef)>1, len(ev_dur)>1]).all()==False: # there was only one event bout in the trace
                inds_final = np.unique(np.concatenate((inds_final, evs_inds))).astype(int)
            else:
                print(np.in1d(evs_inds, inds_final))
                sys.exit('error in neuron %d! some of the events dont exist in inds_final! all events must exist in inds_final!' %iu)
    #    inds_final.shape
        inds_final = inds_final[inds_final>=0] # to avoid the negative values that can happen due to taking 20 frames before an event.
        
        traces_y0_evs_now = traces_y0[iu][inds_final]
        
        #plt.figure(); plt.plot(inds_final)
        inds_final_all.append(inds_final)
        traces_y0_evs.append(traces_y0_evs_now) # neurons


    inds_final_all = np.array(inds_final_all)
    traces_y0_evs = np.array(traces_y0_evs) # neurons

    
    
    #################################################################################################
    ##################### make plots of traces_events for a random y_neuron #########################
    #################################################################################################
    if doPlots:
        get_ipython().magic(u'matplotlib inline')
        neuron_y = rnd.permutation(traces_y0.shape[0])[0] # 10 # 4
#        print(neuron_y)
        evs_inds = np.argwhere(evs[neuron_y]).flatten()

        
        # plot the entire trace and mark the extracted events
        plt.figure(); plt.suptitle('Y Neuron, %d' %neuron_y)
        plt.subplot(211)
        plt.plot(traces_y0[neuron_y],'b', label='df/f'); 
        plt.plot(evs_inds, np.full(evs_inds.shape, 1),'g.', label='events'); # max(traces_y0[neuron_y]) 
        plt.plot(inds_final_all[neuron_y], np.ones(inds_final_all[neuron_y].shape)*.9, 'r.', label='extracted frames')
        plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False, fontsize=12)
        makeNicePlots(plt.gca())
        
        plt.subplot(212)
        plt.plot(erfc[neuron_y],'r', label='-erfc'); 
        plt.plot(evs[neuron_y].astype(int)*10,'g', label='events'); 
        plt.plot(traces_y0[neuron_y],'b', label='df/f'); 
        plt.hlines(th_ag, 0, erfc.shape[1])
        plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False, fontsize=12)
        makeNicePlots(plt.gca())
        
        # now plot the extracted chuch of trace which includes events!!
        #iu = 10
        evs_inds_evs = np.argwhere(evs[neuron_y][inds_final_all[neuron_y]])
        plt.figure(); plt.suptitle('Y Neuron, %d' %neuron_y)
        plt.subplot(211)
        plt.plot(traces_y0_evs[neuron_y],'b', label='df/f'); 
        plt.plot(evs_inds_evs, np.full(evs_inds_evs.shape, 1),'g.', label='events'); # max(traces_y0[neuron_y]) 
        makeNicePlots(plt.gca())
        
        plt.subplot(212)
        plt.plot(traces_y0_evs[neuron_y],'b', label='df/f'); 
        plt.plot(erfc[neuron_y][inds_final_all[neuron_y]],'r', label='-erfc'); 
        plt.plot(evs[neuron_y][inds_final_all[neuron_y]].astype(int)*10,'g', label='events'); 
        plt.hlines(th_ag, 0, traces_y0_evs[neuron_y].shape)
        plt.legend(loc='center left', bbox_to_anchor=(1, .7), frameon=False, fontsize=12)
        makeNicePlots(plt.gca())
    
    return traces_y0_evs, inds_final_all


#%%
def evaluate_components(traces, N=5, robust_std=False):
    
    # traces: neurons x frames
    
    """ Define a metric and order components according to the probabilty if some "exceptional events" (like a spike). Suvh probability is defined as the likeihood of observing the actual trace value over N samples given an estimated noise distribution. 
    The function first estimates the noise distribution by considering the dispersion around the mode. This is done only using values lower than the mode. The estimation of the noise std is made robust by using the approximation std=iqr/1.349. 
    Then, the probavility of having N consecutive eventsis estimated. This probability is used to order the components. 

    Parameters
    ----------
    traces: ndarray
        Fluorescence traces 

    N: int
        N number of consecutive events


    Returns
    -------
    idx_components: ndarray
        the components ordered according to the fitness

    fitness: ndarray


    erfc: ndarray
        probability at each time step of observing the N consequtive actual trace values given the distribution of noise


    Created on Tue Aug 23 09:40:37 2016    
    @author: Andrea G with small modifications from farznaj

    """

#    import scipy  #    import numpy   #    from scipy.stats import norm   

#    import numpy as np
#    import scipy.stats as st

    
    T=np.shape(traces)[-1]
   # import pdb
   # pdb.set_trace()
    md = mode_robust(traces, axis=1)
    ff1 = traces - md[:, None]
    # only consider values under the mode to determine the noise standard deviation
    ff1 = -ff1 * (ff1 < 0)
    if robust_std:
        # compute 25 percentile
        ff1 = np.sort(ff1, axis=1)
        ff1[ff1 == 0] = np.nan
        Ns = np.round(np.sum(ff1 > 0, 1) * .5)
        iqr_h = np.zeros(traces.shape[0])

        for idx, el in enumerate(ff1):
            iqr_h[idx] = ff1[idx, -Ns[idx]]

        # approximate standard deviation as iqr/1.349
        sd_r = 2 * iqr_h / 1.349

    else:
        Ns = np.sum(ff1 > 0, 1)
        sd_r = np.sqrt(np.sum(ff1**2, 1) / Ns)
#


    # compute z value
    z = (traces - md[:, None]) / (3 * sd_r[:, None])
    # probability of observing values larger or equal to z given notmal
    # distribution with mean md and std sd_r
    erf = 1 - st.norm.cdf(z)
    # use logarithm so that multiplication becomes sum
    erf = np.log(erf)
    filt = np.ones(N)
    # moving sum
    erfc = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='full'), axis=1, arr=erf)
    erfc = erfc[:,:T]

    # select the maximum value of such probability for each trace
    fitness = np.min(erfc, 1)

    ordered = np.argsort(fitness)

    idx_components = ordered  # [::-1]# selec only portion of components
#    fitness = fitness[idx_components] % FN commented bc we want the indexing to match C and YrA.
#    erfc = erfc[idx_components] % FN commented bc we want the indexing to match C and YrA.

    return idx_components, fitness, erfc




def mode_robust(inputData, axis=None, dtype=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3
    """
    if axis is not None:
        fnc = lambda x: mode_robust(x, dtype=dtype)
        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        def _hsm(data):
            if data.size == 1:
                return data[0]
            elif data.size == 2:
                return data.mean()
            elif data.size == 3:
                i1 = data[1] - data[0]
                i2 = data[2] - data[1]
                if i1 < i2:
                    return data[:2].mean()
                elif i2 > i1:
                    return data[1:].mean()
                else:
                    return data[1]
            else:
                
                #            wMin = data[-1] - data[0]
                wMin = np.inf
                N = data.size / 2 + data.size % 2
                N = int(N)
                for i in range(0, N):
                    w = data[i + N - 1] - data[i]
                    if w < wMin:
                        wMin = w
                        j = i

                return _hsm(data[j:j + N])

        data = inputData.ravel()
        if type(data).__name__ == "MaskedArray":
            data = data.compressed()
        if dtype is not None:
            data = data.astype(dtype)

        # The data need to be sorted for this to work
        data = np.sort(data)

        # Find the mode
        dataMode = _hsm(data)

    return dataMode


#%% Find length of gaps between events

def find_event_gaps(evs):
    # evs: boolean; neurons x frames
    # evs indicates if there was an event. (it can be 1 for several continuous frames too)
    # eg: evs = (traces >= th_ag) # neurons x frames
    
    d = np.diff(evs.astype(int), axis=1) # neurons x frames
    begs = np.array(np.nonzero(d==1)) # 2 x sum(d==1) # first row is row index (ie neuron) in d; second row is column index (ie frame) in d.
    ends = np.array(np.nonzero(d==-1)) # 2 x sum(d==-1)
    #np.shape(begs)
    #np.shape(ends)
    
    ##### Note: begs_evs does not include the first event; so index i in gap_evs corresponds to index i in begs_evs and ends_evs.
    ##### however, gaps are truely gaps, ie number of frames without an event that span the interval between two frames.
    gap_evs_all = [] # # includes the gap before the 1st event and the gap before the last event too, in addition to inter-event gaps        
    gap_evs = [] # includes only gap between events 
    begs_evs = [] # index of event onsets, excluding the 1st events. (in fact these are 1 index before the actual event onset; since they are computed on the difference trace ('d'))
    ends_evs = [] # index of event offsets, excluding the last event. 
    bgap_evs = [] # number of frames before the first event
    egap_evs = [] # number of frames after the last event

    for iu in range(evs.shape[0]): 
#        print(iu)
        # sum(begs[0]==0)
        begs_this_n = begs[1,begs[0]==iu] # indeces belong to "d" (the difference trace)
        ends_this_n = ends[1,ends[0]==iu]

        # gap between event onsets will be begs(next event) - ends(current event)
        
        if evs[iu, 0]==False and evs[iu, -1]==False: # normal case
            #len(begs_this_n) == len(ends_this_n): 
            b = begs_this_n[1:]
            e = ends_this_n[:-1]
            
            bgap = [begs_this_n[0] + 1] # after how many frames the first event happened
            egap = [evs.shape[1] - ends_this_n[-1] - 1] # how many frames with no event exist after the last event
            
        elif evs[iu, 0]==True and evs[iu, -1]==False: # first event was already going when the recording started.
            #len(begs_this_n)+1 == len(ends_this_n): 
            b = begs_this_n
            e = ends_this_n[:-1]
            
            bgap = []
            egap = [evs.shape[1] - ends_this_n[-1] - 1]
        
        elif evs[iu, 0]==False and evs[iu, -1]==True: # last event was still going on when the recording ended.
            #len(begs_this_n) == len(ends_this_n)+1:
            b = begs_this_n[1:]
            e = ends_this_n
            
            bgap = [begs_this_n[0] + 1]
            egap = []
            
        elif evs[iu, 0]==True and evs[iu, -1]==True: # first event and last event were happening when the recording started and ended.
#            print('cool :)')
            b = begs_this_n
            e = ends_this_n
            
            bgap = []
            egap = []
    
        else:
            sys.exit('doesnt make sense! plot d to debug')
        
        
        gap_this_n = b - e
        gap_this = np.concatenate((bgap, gap_this_n, egap)).astype(int) # includes all gaps, before the 1st event, between events, and after the last event.
        
        
        gap_evs_all.append(gap_this) # includes the gap before the 1st event and the gap before the last event too.        
        gap_evs.append(gap_this_n) # only includes gaps between events: b - e      
        begs_evs.append(b)
        ends_evs.append(e)
        bgap_evs.append(bgap)
        egap_evs.append(egap)

    
    gap_evs_all = np.array(gap_evs_all) # size: number of neurons; each element shows the gap between events for a given neuron
    gap_evs = np.array(gap_evs)
    begs_evs = np.array(begs_evs)
    ends_evs = np.array(ends_evs)
    bgap_evs = np.array(bgap_evs)
    egap_evs = np.array(egap_evs)
    
    return gap_evs_all, begs_evs, ends_evs, gap_evs, bgap_evs, egap_evs, begs, ends




