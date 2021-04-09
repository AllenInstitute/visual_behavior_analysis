import matplotlib.pyplot as plt

import visual_behavior.data_access.loading as loading
import visual_behavior.visualization.utils as utils
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis


def plot_lick_triggered_average(lick_triggered_response_df, color='gray', ylabel='response', legend_label=None, ax=None):
    ldf = lick_triggered_response_df.copy()
    if ax is None:
        figsize = (7, 5)
        fig, ax = plt.subplots(figsize=figsize)
    label = None  # defining to avoid lint error - probably not what you want
    ax.plot(ldf.trace_timestamps.mean(), ldf.trace.mean(), legend_label=label, color=color)
    ax.vlines(x=0, ymin=0, ymax=1, color='gray', linestyle='--')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('time after first lick in bout (sec)')
    return ax


def plot_lick_triggered_average_for_container(container_id, save_figure=True):

    experiments = loading.get_filtered_ophys_experiment_table()
    container_data = experiments[experiments.container_id == container_id]

    figsize = (6, 4)
    fig, ax = plt.subplots(figsize=figsize)

    for session_number in container_data.session_number.unique():
        experiment_ids = container_data[container_data.session_number == session_number].index.values
        for experiment_id in experiment_ids:
            dataset = loading.get_ophys_dataset(experiment_id)
            analysis = ResponseAnalysis(dataset, use_events=True, use_extended_stimulus_presentations=False)
            ldf = analysis.get_response_df(df_name='lick_triggered_response_df')
            if len(ldf.cell_specimen_id.unique()) > 5:
                colors = utils.get_colors_for_session_numbers()
                ax = plot_lick_triggered_average(
                    ldf, color=colors[session_number - 1],
                    ylabel='pop. avg. \nevent magnitude',
                    legend_label=session_number,
                    ax=ax
                )
    ax.legend(loc='upper left', fontsize='x-small')
    fig.tight_layout()
    m = dataset.metadata.copy()
    title = str(m['mouse_id']) + '_' + m['full_genotype'].split('/')[0] + '_' + m['targeted_structure'] + '_' + str(m['imaging_depth']) + '_' + m['equipment_name'] + '_' + str(m['experiment_container_id'])
    fig.suptitle(title, x=0.53, y=1.02, fontsize=16)
    if save_figure:
        save_dir = loading.get_container_plots_dir()
        utils.save_figure(fig, figsize, save_dir, 'lick_triggered_average', title)
