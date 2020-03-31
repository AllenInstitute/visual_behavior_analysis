
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy.random as rnd
import sys


def get_traces_evs(traces_y0, th_ag, len_ne, do_plots=1):
    """
    Function to get an "active trace" i.e. a trace made by extracting and concatenating the active parts of the input trace

    example use:
        len_ne = 20
        th_ag = 10
        doPlots = 1
        [traces_y0_evs, inds_final_all] = get_traces_evs(traces_y0, th_ag, len_ne, doPlots)

        or, if need to re-apply to a different input vector:
        traces_active[neuron_y] = traces[neuron_y][inds_final_all[neuron_y]]

    Farzaneh Najafi
    March 2020

    :param traces_y0: numpy.array of size NxM where N : number of neurons (rois), M: number of timestamps
    :param th_ag: scalar : threshold to find events on the trace; the higher the more strict on what we call an event.
    :param len_ne: scalar; number of frames before and after each event that are taken to create traces_events
    :param do_plots: bool, flag to control whether to plot
    :return:
        traces_y0_evs: ndarray, size N (number of neurons); each neuron has size n, which is the size of the "active trace" for that neuron
        inds_final_all: ndarray, size number_of_neurons; indeces to apply on traces_y0 to get traces_y0_evs:
    """

    #  Andrea Giovannucci's method of identifying "exceptional" events
    [_, _, erfc] = evaluate_components(traces_y0, N=5, robust_std=False)
    erfc = -erfc

    # applying threshold
    evs = (erfc >= th_ag)  # neurons x frames

    evs_fract = np.mean(evs, axis=1)

    if do_plots:

        plt.figure(figsize=(4, 3))
        plt.plot(evs_fract)
        plt.xlabel('Neuron in Y trace')
        plt.title('Fraction of time points with high ("active") DF/F')
        make_nice_plots(plt.gca())

    # find gaps between events for each neuron
    [_, begs_evs, ends_evs, _, bgap_evs, egap_evs, _, _] = find_event_gaps(evs)

    # set traces_evs, ie a trace that contains mostly the active parts of the input trace #
    traces_y0_evs = []
    inds_final_all = []

    for iu in range(traces_y0.shape[0]):

        if sum(evs[iu]) > 0:

            enow = ends_evs[iu]
            bnow = begs_evs[iu]
            e_aft = []
            b_bef = []
            for ig in range(len(bnow)):

                e_aft.append(np.arange(enow[ig], min(evs.shape[1], enow[ig] + len_ne)))
                b_bef.append(np.arange(bnow[ig] + 1 - len_ne, min(evs.shape[1], bnow[ig] + 2)))

            e_aft = np.array(e_aft)
            b_bef = np.array(b_bef)

            if len(e_aft) > 1:
                e_aft_u = np.hstack(e_aft)
            else:
                e_aft_u = []

            if len(b_bef) > 1:
                b_bef_u = np.hstack(b_bef)
            else:
                b_bef_u = []

            # below sets frames that cover the duration of all events, but excludes the first and last event
            ev_dur = []
            for ig in range(len(bnow) - 1):
                ev_dur.append(np.arange(bnow[ig], enow[ig + 1]))

            ev_dur = np.array(ev_dur)

            if len(ev_dur) > 1:
                ev_dur_u = np.hstack(ev_dur)
            else:
                ev_dur_u = []
            # ev_dur_u.shape

            evs_inds = np.argwhere(evs[iu]).flatten()  # includes ALL events.

            if len(bgap_evs[iu]) > 0:
                # get len_ne frames before the 1st event
                ind1 = np.arange(np.array(bgap_evs[iu]) - len_ne, bgap_evs[iu])
                # if the 1st event is immediately followed by more events, add those to ind1, because they dont appear in any of the other vars that we are concatenating below.
                if len(ends_evs[iu]) > 1:
                    ii = np.argwhere(
                        np.in1d(evs_inds, ends_evs[iu][0])).squeeze()
                    ind1 = np.concatenate((ind1, evs_inds[:ii]))
            else:  # first event was already going when the recording started; add these events to ind1
                jj = np.argwhere(np.in1d(evs_inds, ends_evs[iu][0])).squeeze()
    #            jj = ends_evs[iu][0]
                ind1 = evs_inds[:jj + 1]

            if len(egap_evs[iu]) > 0:
                # get len_ne frames after the last event
                indl = np.arange(evs.shape[1] - np.array(egap_evs[iu]) - 1, min(
                    evs.shape[1], evs.shape[1] - np.array(egap_evs[iu]) + len_ne))
                # if the last event is immediately preceded by more events, add those to indl, because they dont appear in any of the other vars that we are concatenating below.
                if len(begs_evs[iu]) > 1:
                    # find the fist event of the last event bout
                    ii = np.argwhere(
                        np.in1d(evs_inds, 1 + begs_evs[iu][-1])).squeeze()
                    indl = np.concatenate((evs_inds[ii:], indl))
            else:  # last event was already going when the recording ended; add these events to ind1
                jj = np.argwhere(
                    np.in1d(evs_inds, begs_evs[iu][-1] + 1)).squeeze()
                indl = evs_inds[jj:]

            inds_final = np.unique(np.concatenate(
                (e_aft_u, b_bef_u, ev_dur_u, ind1, indl))).astype(int)

            # all evs_inds must exist in inds_final, otherwise something is wrong!
            if not np.in1d(evs_inds, inds_final).all():
                # there was only one event bout in the trace
                if not np.array([len(e_aft) > 1, len(b_bef) > 1, len(ev_dur) > 1]).all():
                    inds_final = np.unique(np.concatenate(
                        (inds_final, evs_inds))).astype(int)
                else:
                    print(np.in1d(evs_inds, inds_final))
                    sys.exit(
                        'error in neuron %d! some of the events dont exist in inds_final! all events must exist in inds_final!' % iu)

            traces_y0_evs_now = traces_y0[iu][inds_final]

        else:  # there are no events in the neuron; assign a nan vector of length 10 to the following vars
            inds_final = np.full((10,), np.nan)
            traces_y0_evs_now = np.full((10,), np.nan)

        inds_final_all.append(inds_final)
        traces_y0_evs.append(traces_y0_evs_now)  # neurons

    inds_final_all = np.array(inds_final_all)
    traces_y0_evs = np.array(traces_y0_evs)  # neurons

    # make plots of traces_events for a random y_neuron #
    if do_plots:
        neuron_y = rnd.permutation(traces_y0.shape[0])[0]
        if sum(evs[neuron_y]) == 0:
            neuron_y = rnd.permutation(traces_y0.shape[0])[0]

        doscl = 1  # set to 1 if not dealing with df/f , so you can better see the plots
        if doscl:
            traces_y0c = traces_y0 / np.max(traces_y0, axis=0)
            traces_y0_evsc = [
                traces_y0_evs[iu] / np.max(traces_y0_evs[iu]) for iu in range(len(traces_y0_evs))]

        evs_inds = np.argwhere(evs[neuron_y]).flatten()

        # plot the entire trace and mark the extracted events
        plt.figure()
        plt.suptitle('Y Neuron, %d' % neuron_y)
        plt.subplot(211)
        plt.plot(traces_y0[neuron_y], 'b', label='df/f')

        # max(traces_y0[neuron_y])
        plt.plot(evs_inds, np.full(evs_inds.shape, max(traces_y0[neuron_y])), 'g.', label='events')
        plt.plot(inds_final_all[neuron_y], np.ones(inds_final_all[neuron_y].shape)
                 * (max(traces_y0[neuron_y]) * .9), 'r.', label='extracted frames')
        plt.legend(loc='center left', bbox_to_anchor=(
            1, .7), frameon=False, fontsize=12)
        make_nice_plots(plt.gca())

        plt.subplot(212)
        plt.plot(erfc[neuron_y], 'r', label='-erfc')
        plt.plot(evs[neuron_y].astype(int) * 10, 'g', label='events')
        plt.plot(traces_y0c[neuron_y], 'b', label='df/f')
        plt.hlines(th_ag, 0, erfc.shape[1])
        plt.legend(loc='center left', bbox_to_anchor=(
            1, .7), frameon=False, fontsize=12)
        make_nice_plots(plt.gca())

        # now plot the extracted chunk of trace which includes events!!
        # iu = 10
        evs_inds_evs = np.argwhere(evs[neuron_y][inds_final_all[neuron_y]])
        plt.figure()
        plt.suptitle('Y Neuron, %d' % neuron_y)
        plt.subplot(211)
        plt.plot(traces_y0_evsc[neuron_y], 'b', label='df/f')
        plt.plot(evs_inds_evs, np.full(evs_inds_evs.shape, 1),
                 'g.', label='events')  # max(traces_y0[neuron_y])
        make_nice_plots(plt.gca())

        plt.subplot(212)
        plt.plot(traces_y0_evsc[neuron_y], 'b', label='df/f')
        plt.plot(erfc[neuron_y][inds_final_all[neuron_y]], 'r', label='-erfc')
        plt.plot(evs[neuron_y][inds_final_all[neuron_y]].astype(
            int) * 10, 'g', label='events')
        plt.hlines(th_ag, 0, traces_y0_evsc[neuron_y].shape)
        plt.legend(loc='center left', bbox_to_anchor=(
            1, .7), frameon=False, fontsize=12)
        make_nice_plots(plt.gca())

    return traces_y0_evs, inds_final_all


def evaluate_components(traces, n=5, robust_std=False):

    """ Define a metric and order components according to the probability if some "exceptional events" (like a spike).
    Suvh probability is defined as the likelihood of observing the actual trace value over N samples given an estimated
    noise distribution. The function first estimates the noise distribution by considering the dispersion around the
    mode. This is done only using values lower than the mode. The estimation of the noise std is made robust by using
    the approximation std=iqr/1.349. Then, the probability of having N consecutive events is estimated.
    This probability is used to order the components.

    Created on Tue Aug 23 09:40:37 2016
    @author: Andrea G with small modifications from farznaj

    :param n: int, number of consecutive events
    :param traces: numpy.array, Fluorescence traces
    :param robust_std:
    :return
        idx_components: numpy.array; the components ordered according to the fitness
        fitness: numpy.array;
        erfc: numpy.array; probability at each time step of observing the N consecutive actual trace values given the distribution of noise

    """

    t = np.shape(traces)[-1]
    md = mode_robust(traces, axis=1)
    ff1 = traces - md[:, None]

    # only consider values under the mode to determine the noise standard deviation
    ff1 = -ff1 * (ff1 < 0)
    if robust_std:
        # compute 25 percentile
        ff1 = np.sort(ff1, axis=1)
        ff1[ff1 == 0] = np.nan
        ns = np.round(np.sum(ff1 > 0, 1) * .5)
        iqr_h = np.zeros(traces.shape[0])

        for idx, el in enumerate(ff1):
            iqr_h[idx] = ff1[idx, -ns[idx]]

        # approximate standard deviation as iqr/1.349
        sd_r = 2 * iqr_h / 1.349

    else:
        ns = np.sum(ff1 > 0, 1)
        sd_r = np.sqrt(np.sum(ff1**2, 1) / ns)
#

    # compute z value
    z = (traces - md[:, None]) / (3 * sd_r[:, None])
    # probability of observing values larger or equal to z given notmal
    # distribution with mean md and std sd_r
    erf = 1 - st.norm.cdf(z)
    # use logarithm so that multiplication becomes sum
    erf = np.log(erf)
    filt = np.ones(n)
    # moving sum
    erfc = np.apply_along_axis(lambda m: np.convolve(
        m, filt, mode='full'), axis=1, arr=erf)
    erfc = erfc[:, :t]

    # select the maximum value of such probability for each trace
    fitness = np.min(erfc, 1)

    ordered = np.argsort(fitness)

    idx_components = ordered  # [::-1]# selec only portion of components
#    fitness = fitness[idx_components] % FN commented bc we want the indexing to match C and YrA.
#    erfc = erfc[idx_components] % FN commented bc we want the indexing to match C and YrA.

    return idx_components, fitness, erfc


def mode_robust(input_data, axis=None, d_type=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3
    """
    if axis is not None:
        def fnc(x):
            return mode_robust(x, d_type=d_type)
        data_mode = np.apply_along_axis(fnc, axis, input_data)
    else:
        # Create the function that we can use for the half-sample mode
        def _hsm(dt):
            if dt.size == 1:
                return dt[0]
            elif dt.size == 2:
                return dt.mean()
            elif dt.size == 3:
                i1 = dt[1] - dt[0]
                i2 = dt[2] - dt[1]
                if i1 < i2:
                    return dt[:2].mean()
                elif i2 > i1:
                    return dt[1:].mean()
                else:
                    return dt[1]
            else:

                w_min = np.inf
                n = dt.size / 2 + dt.size % 2
                n = int(n)
                for i in range(0, n):
                    w = dt[i + n - 1] - dt[i]
                    if w < w_min:
                        w_min = w
                        j = i

                return _hsm(dt[j:j + n])

        data = input_data.ravel()  # flatten all dimensions
        if type(data).__name__ == "MaskedArray":
            data = data.compressed()
        if d_type is not None:
            data = data.astype(d_type)

        # The data need to be sorted for this to work
        data = np.sort(data)

        # Find the mode
        data_mode = _hsm(data)

    return data_mode


def find_event_gaps(evs):
    """
    function to find gaps between events
    :param evs: boolean; neurons x frames, indicates if there was an event. (it can be 1 for several continuous frames too)
    :return:
        gap_evs_all: includes the gap before the 1st event and the gap before the last event too, in addition to inter-event gaps
        begs_evs: index of event onsets, excluding the 1st events. (in fact these are 1 index before the actual event onset;
                  since they are computed on the difference trace ('d'))
        ends_evs: index of event offsets, excluding the last event
        gap_evs: includes only gap between events
        bgap_evs: number of frames before the first event
        egap_evs: number of frames after the last event
        begs: beginings of the events
        ends: ends of the events
    """
    d = np.diff(evs.astype(int), axis=1)
    begs = np.array(np.nonzero(d == 1))
    ends = np.array(np.nonzero(d == -1))
    gap_evs_all = []
    gap_evs = []
    begs_evs = []
    ends_evs = []
    bgap_evs = []
    egap_evs = []
    for iu in range(evs.shape[0]):
        # make sure there are events in the trace of unit iu
        if sum(evs[iu]) > 0:
            # indeces belong to "d" (the difference trace)
            begs_this_n = begs[1, begs[0] == iu]
            ends_this_n = ends[1, ends[0] == iu]

            # gap between event onsets will be begs(next event) - ends(current event)
            if not evs[iu, 0] and not evs[iu, -1]:  # normal case
                b = begs_this_n[1:]
                e = ends_this_n[:-1]
                # after how many frames the first event happened
                bgap = [begs_this_n[0] + 1]
                # how many frames with no event exist after the last event
                egap = [evs.shape[1] - ends_this_n[-1] - 1]

            # first event was already going when the recording started.
            elif evs[iu, 0] and not evs[iu, -1]:
                # len(begs_this_n)+1 == len(ends_this_n):
                b = begs_this_n
                e = ends_this_n[:-1]

                bgap = []
                egap = [evs.shape[1] - ends_this_n[-1] - 1]

            # last event was still going on when the recording ended.
            elif not evs[iu, 0] and evs[iu, -1]:
                # len(begs_this_n) == len(ends_this_n)+1:
                b = begs_this_n[1:]
                e = ends_this_n

                bgap = [begs_this_n[0] + 1]
                egap = []

            # first event and last event were happening when the recording started and ended.
            elif evs[iu, 0] and evs[iu, -1]:
                b = begs_this_n
                e = ends_this_n

                bgap = []
                egap = []

            else:
                sys.exit('doesnt make sense! plot d to debug')

            gap_this_n = b - e
            # includes all gaps, before the 1st event, between events, and after the last event.
            gap_this = np.concatenate((bgap, gap_this_n, egap)).astype(int)

        else:  # there are no events in this neuron; set everything to nan
            gap_this = np.nan
            gap_this_n = np.nan
            b = np.nan
            e = np.nan
            bgap = np.nan
            egap = np.nan

        # includes the gap before the 1st event and the gap before the last event too.
        gap_evs_all.append(gap_this)
        gap_evs.append(gap_this_n)  # only includes gaps between events: b - e
        begs_evs.append(b)
        ends_evs.append(e)
        bgap_evs.append(bgap)
        egap_evs.append(egap)

    # size: number of neurons; each element shows the gap between events for a given neuron
    gap_evs_all = np.array(gap_evs_all)
    gap_evs = np.array(gap_evs)
    begs_evs = np.array(begs_evs)
    ends_evs = np.array(ends_evs)
    bgap_evs = np.array(bgap_evs)
    egap_evs = np.array(egap_evs)

    return gap_evs_all, begs_evs, ends_evs, gap_evs, bgap_evs, egap_evs, begs, ends


def make_nice_plots(ax, rmv2nd_xtick_label=0, rmv2nd_ytick_label=0):
    """
    Function to only show left and bottom axes of plots, make tick directions outward, remove every other tick label if requested.
    :param ax:
    :param rmv2nd_xtick_label:
    :param rmv2nd_ytick_label:
    :return:
    """
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(direction='out')
    if rmv2nd_xtick_label:
        [label.set_visible(False) for label in ax.xaxis.get_ticklabels()[::2]]

    if rmv2nd_ytick_label:
        [label.set_visible(False) for label in ax.yaxis.get_ticklabels()[::2]]
    plt.grid(False)

    ax.tick_params(labelsize=12)
