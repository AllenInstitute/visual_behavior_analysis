
##########################################
# codes summary that gives us cc traces:
# for each session, cc traces are averaged across all neurons pairs
# for each mouse, an average is computed across sessions
# a final average is computed across mice

# remember: baselines are not subtracted from the traces; but when we quantify the peak we subtract the baseline if sameBl_allLayerPairs is set to 1.


cc11_sessAv_avMice


cc11_sessAv_avMice = np.nanmean(cc11_sessAv_allMice_thisCre, axis=0) # 80x4x4

cc11_sessAv_allMice_thisCre[icn] = corr_trace_peak_allMice['cc11_sessAv_44'].iloc[im]

cc11_sessAv_44 = traces_44_layerSame(cc11_sessAv, num_depth) # 80 x 4 x 4

cc11_sessAv = np.nanmean(cc11_aveNPairs_allSess, axis=0) # num_frames x 10

cc11_aveNPairs_allSess[ise] = cc11_aveNPairs

cc11_aveNPairs[:,ilc] = np.nanmean(cc_a11[ilc], axis=1) # num_frames # ave cc across pairs


##########################################
# codes summary that gives us quantification of cc:
# average of cc traces are computed across all neuron pairs
# for each session cc is quantified (note: baseline will be subtracted in quant_cc if sameBl_allLayerPairs=1)
# for each mouse, average cc_peak_amp is computed across sessions
# a final average is computed across mice


cc11_aveNPairs_allSess
cc11_peak_avMice_flash
cc11_peak_avMice_omit


cc11_peak_avMice_omit = np.nanmean(cc11_peak_sessAv_allMice_thisCre_omit, axis=0) # 4x4 

cc11_peak_sessAv_allMice_thisCre_omit[icn] = corr_trace_peak_allMice['cc11_peak_amp_omit_sessAv'].iloc[im]

corr_trace_peak_allMice.at[im, 'cc11_peak_amp_omit_sessAv'] = cc11_peak_amp_omit_sessAv

cc11_peak_amp_omit_sessAv = np.nanmean(cc11_peak_amp_omit_eachSess, axis=0)

# note: baseline will be subtracted in quant_cc if sameBl_allLayerPairs=1
cc11_peak_amp_omit_eachSess[ise], cc11_peak_amp_flash_eachSess[ise] = quant_cc(cc11_aveNPairs_allSess[ise], fo_dur, list_times, list_times_flash_final, samps_bef, bl_percentile, frame_dur, num_depth, sameBl_allLayerPairs)         

cc11_aveNPairs_allSess[ise] = cc11_aveNPairs

cc11_aveNPairs[:,ilc] = np.nanmean(cc_a11[ilc], axis=1) # num_frames

cc_a11 = all_sess_2an_this_mouse['cc_a11'].iloc[ise]




