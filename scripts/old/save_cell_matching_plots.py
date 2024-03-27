from multiprocessing import Pool
from visual_behavior.visualization.qc import single_cell_across_experiments as scae
import visual_behavior_glm.GLM_analysis_tools as gat

from visual_behavior.data_access import loading, from_lims
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

cell_table = loading.get_cell_table()

unfail_path = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/single_cell_plots/cell_matching_qc/potential_cells_to_unfail.csv'
unfail_table = pd.read_csv(unfail_path)
unfail_csids = unfail_table['cell_specimen_id'].to_list()

potential_rois = cell_table.query('cell_specimen_id in @unfail_csids and valid_roi == False')['cell_roi_id']

saveloc = '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/single_cell_plots/cell_matching_qc/all'

def make_plot(cell_roi_id):
    print('making plot for roi {}'.format(cell_roi_id))
    if scae.roi_has_dff(int(cell_roi_id)):
        row = cell_table.query('cell_roi_id == @cell_roi_id').iloc[0]
        cell_specimen_id = int(row['cell_specimen_id'])
        ophys_experiment_id = int(row['ophys_experiment_id'])
        genotype = from_lims.get_genotype_for_ophys_experiment_id(ophys_experiment_id)
        cre_line = genotype.split('/')[0]

        print('roi {}, csid {}, oied {}, cre_line {}'.format(cell_roi_id, cell_specimen_id, ophys_experiment_id, cre_line))

        fig = scae.make_cell_matching_across_experiment_plot(
            cell_specimen_id,
            experiment_id_to_highlight=ophys_experiment_id,
            disable_progress_bar=True
        )
        print('done making fig for roi {}'.format(cell_roi_id))
        
        fn = 'cre_line={}__cell_specimen_id={}__cell_roi_id={}.png'.format(
            cre_line,
            cell_specimen_id,
            cell_roi_id
        )
        print('filename = {}'.format(fn))
        fig.savefig(
            os.path.join(saveloc, fn)
        )
        print('done making plot for roi {}'.format(cell_roi_id))
    else:
        print('skipping roi {}, its dff is NaN'.format(cell_roi_id))

with Pool(32) as pool:
    pool.map(make_plot, potential_rois)
print('done generating plots')