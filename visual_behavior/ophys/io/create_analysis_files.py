
from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
from visual_behavior.ophys.plotting import summary_figures as sf


def create_analysis_files(experiment_id):
    dataset = VisualBehaviorOphysDataset(experiment_id)

    analysis = ResponseAnalysis(dataset)

    print('plotting cell responses')
    for cell in dataset.get_cell_indices():
        sf.plot_image_response_for_trial_types(analysis, cell)
    print('done')


if __name__ == '__main__':
    # import sys
    # experiment_id = sys.argv[1]
    experiment_id = 719996589

    create_analysis_files(experiment_id)

