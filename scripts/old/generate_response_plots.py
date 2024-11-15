from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
import visual_behavior.visualization.ophys.summary_figures as sf
import visual_behavior.database as db
import visual_behavior.data_access.loading as loading
import pandas as pd

import argparse

def generate_save_plots(experiment_id, split_by):
    dataset = loading.get_ophys_dataset(experiment_id)
    analysis = ResponseAnalysis(dataset, overwrite_analysis_files=False, dataframe_format='tidy', use_extended_stimulus_presentations=True)
    for cell_specimen_id in dataset.cell_specimen_table.query('valid_roi==True').index.values:
        sf.make_cell_response_summary_plot(analysis, cell_specimen_id, split_by, save=True, show=False, errorbar_bootstrap_iterations=1000)

def summarize_responses(experiment_id):
    dataset = loading.get_ophys_dataset(experiment_id)
    analysis = ResponseAnalysis(dataset, overwrite_analysis_files=False, dataframe_format='tidy', use_extended_stimulus_presentations=True)

    valid_cells = list(dataset.cell_specimen_table.query('valid_roi==True').index.values)
    summaries = []
    for df_type in ['omission','stimulus','trials']:
        summary = (
            getattr(analysis, '{}_response_df'.format(df_type))
            .query('cell_specimen_id in @valid_cells')
            .drop_duplicates([c for c in ['cell_specimen_id','stimulus_presentations_id','trials_id'] if c in getattr(analysis, '{}_response_df'.format(df_type))])
            .groupby(['cell_specimen_id','engagement_state'])[['mean_response','mean_baseline']]
            .mean()
            .reset_index()
        )
        for key in analysis.dataset.metadata.keys():
            val = analysis.dataset.metadata[key]
            if type(val) is not list:
                summary[key] = val
        summary['event_type'] = df_type
        summaries.append(summary)
    return pd.concat(summaries)

def log_summary_to_db(df_to_log):
    conn = db.Database('visual_behavior_data')
    for idx,row in df_to_log.iterrows():
        entry = row.to_dict()
        db.update_or_create(
            conn['ophys_analysis']['event_triggered_responses'], 
            db.clean_and_timestamp(entry), 
            keys_to_check = ['ophys_experiment_id','cell_specimen_id', 'event_type','engagement_state']
        )
    conn.close()

def summarize_and_log(experiment_id):
    summary = summarize_responses(experiment_id)
    log_summary_to_db(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate engaged/disengaged response plots')
    parser.add_argument(
        '--experiment-id',
        type=int,
        default=0,
        metavar='ophys_experiment_id'
    )

    parser.add_argument(
        '--action',
        type=str,
        default='plot',
        metavar='action to execute ("plot" or "log")'
    )

    parser.add_argument(
        '--split-by',
        type=str,
        default='engagement_state',
        metavar='variable to split by'
    )

    args = parser.parse_args()
    if args.action == 'log':
        summarize_and_log(args.experiment_id)
    elif args.action == 'plot':
        generate_save_plots(args.experiment_id, args.split_by)
