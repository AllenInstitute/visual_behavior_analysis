from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
import visual_behavior.ophys.response_analysis.utilities as ut


from visual_behavior.visualization.ophys import experiment_summary_figures as esf
from visual_behavior.visualization.ophys import summary_figures as sf

# import logging
import os
import numpy as np

# logger = logging.getLogger(__name__)


def create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=True):
    # logger.info(experiment_id)
    print(experiment_id)
    print('saving ', str(experiment_id), 'to', cache_dir)
    dataset = VisualBehaviorOphysDataset(experiment_id, cache_dir)

    use_events = False
    # analysis = ResponseAnalysis(dataset, overwrite_analysis_files, use_events=use_events)

    # print('plotting experiment summary figure')
    # esf.plot_experiment_summary_figure(analysis, save_dir=cache_dir, use_events=use_events)
    # esf.plot_experiment_summary_figure(analysis, save_dir=dataset.analysis_dir, use_events=use_events)
    # esf.plot_roi_masks(dataset, save_dir=cache_dir)

    print('plotting example traces')
    snr_values = []
    for i, trace in enumerate(dataset.dff_traces):
        mean = np.mean(trace, axis=0)
        std = np.std(trace, axis=0)
        snr = std / mean
        snr_values.append(snr)
    active_cell_indices = np.argsort(snr_values)[-10:]
    length_mins = 1
    for xmin_seconds in np.arange(0, 3000, length_mins * 60):
        sf.plot_example_traces_and_behavior(dataset, active_cell_indices, xmin_seconds, length_mins, save=True,
                                         cell_label=False, include_running=True, use_events=use_events)

    # print('plotting cell responses')
    # save_dir = os.path.join(cache_dir, 'summary_figures')
    # for cell in dataset.get_cell_indices():
    #     # sf.plot_image_response_for_trial_types(analysis, cell, save=True, use_events=use_events)
    #     sf.plot_cell_summary_figure(analysis, cell, save=True, show=False, cache_dir=cache_dir, use_events=use_events)

    if dataset.events is not None:
        use_events = True
        # analysis = ResponseAnalysis(dataset, overwrite_analysis_files, use_events=use_events)

        # print('plotting experiment summary figure')
        # esf.plot_experiment_summary_figure(analysis, save_dir=cache_dir, use_events=use_events)
        # esf.plot_experiment_summary_figure(analysis, save_dir=dataset.analysis_dir, use_events=use_events)

        print('plotting example traces')
        for xmin_seconds in np.arange(0, 3000, length_mins * 60):
            sf.plot_example_traces_and_behavior(dataset, active_cell_indices, xmin_seconds, length_mins, save=True,
                                             cell_label=False, include_running=True, use_events=use_events)

        # print('plotting cell responses')
        # save_dir = os.path.join(cache_dir, 'summary_figures')
        # for cell in dataset.get_cell_indices():
        #     # sf.plot_image_response_for_trial_types(analysis, cell, save=True, use_events=use_events)
        #     sf.plot_cell_summary_figure(analysis, cell, save=True, show=False, cache_dir=cache_dir, use_events=use_events)
    else:
        print('no events for',experiment_id)



if __name__ == '__main__':
    lims_ids = [639253368, 639438856, 639769395, 639932228, 644942849, 645035903,
                645086795, 645362806, 646922970, 647108734, 647551128, 647887770,
                648647430, 649118720, 649318212, 661423848, 663771245, 663773621,
                664886336, 665285900, 665286182, 670396087, 671152642, 672185644,
                672584839, 673139359, 673460976, 685744008, 686726085, 692342909,
                692841424, 693272975, 693862238, 695471168, 696136550, 698244621,
                698724265, 700914412, 701325132, 702134928, 702723649, 712178916,
                712860764, 713525580, 714126693, 715161256, 715228642, 715887471,
                715887497, 716327871, 716337289, 716600289, 716602547, 719321260,
                719996589, 720001924, 720793118, 723037901, 723064523, 723748162,
                723750115, 729951441, 730863840, 731936595, 732911072, 733691636,
                736490031, 736927574, 737471012, 745353761, 745637183, 747248249,
                750469573, 751935154, 752966796, 753931104, 754552635, 754566180,
                754943841, 756715598, 758274779, 760003838, 760400119, 760696146,
                760986090, 761861597, 762214438, 762214650, 766779984, 767424894,
                768223868, 768224465, 768225217, 768865460, 768871217, 769514560,
                770094844, 771381093, 771427955, 772131949, 772696884, 772735942,
                773816712, 773843260, 774370025, 774379465, 775011398, 775429615,
                776042634, 756565411]

    cache_dir = r'\\allen/programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_pilot_analysis'
    # cache_dir = r'/allen/programs/braintv/workgroups/ophysdev/OPhysCore/Analysis/2018-08 - Behavior Integration test'
    for lims_id in lims_ids[::-1]:
        create_analysis_files(lims_id, cache_dir, overwrite_analysis_files=False)


        # esf.plot_mean_first_flash_response_by_image_block(analysis, save_dir=cache_dir, ax=None)

    # analysis.flash_response_df = ut.annotate_flash_response_df_with_block_set(analysis.flash_response_df)
    # fdf = analysis.flash_response_df.copy()
    # data = ut.add_early_late_block_ratio_for_fdf(fdf)
    # save_dir = os.path.join(cache_dir, 'multi_session_summary_figures')
    # esf.plot_mean_response_across_image_block_sets(data, analysis.dataset.analysis_folder, save_dir=save_dir, ax=None)

    # for cell_specimen_id in dataset.cell_specimen_ids:
    #     sf.plot_mean_trace_and_events(cell_specimen_id, analysis, ax=None, save=True)
    #     for trial_num in range(25):
    #         sf.plot_single_trial_with_events(cell_specimen_id, trial_num, analysis, ax=None, save=True)
    #
    # if dataset.events is not None:
    #     sf.plot_event_detection(dataset.dff_traces, dataset.events, dataset.analysis_dir)

    # for cell in dataset.get_cell_indices():
    #     sf.plot_image_response_for_trial_types(analysis, cell, save_dir=analysis.dataset.analysis_dir)
    #     sf.plot_image_response_for_trial_types(analysis, cell, save_dir=save_dir)

        # sf.plot_mean_response_by_repeat(analysis, cell, save_dir=analysis.dataset.analysis_dir)
        # sf.plot_mean_response_by_image_block(analysis, cell, save_dir=analysis.dataset.analysis_dir)
        # sf.plot_mean_response_by_repeat(analysis, cell, save_dir=save_dir)
        # sf.plot_mean_response_by_image_block(analysis, cell, save_dir=save_dir)

    # logger.info('done')
