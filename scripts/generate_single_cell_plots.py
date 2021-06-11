from multiprocessing import Pool
from visual_behavior.visualization.qc import single_cell_across_experiments as scae
import visual_behavior_glm.GLM_analysis_tools as gat

import warnings
warnings.filterwarnings("ignore")

print('loading GLM results...')
glm_version = '16_events_all_L2_optimize_by_session'
glm_results = gat.retrieve_results({'glm_version': glm_version}, results_type='full')
print('done loading GLM results')

saveloc = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/single_cell_plots/response_and_glm_summary'
csids = glm_results['cell_specimen_id'].unique()
args = zip(csids, [glm_version]*len(csids), [True]*len(csids), [saveloc]*len(csids), [False]*len(csids))

print('starting to generate plots. There will be {} of them. This will take a while...'.format(len(csids)))
with Pool(32) as pool:
    pool.starmap(scae.make_single_cell_across_experiment_plot, args)
print('done generating plots')