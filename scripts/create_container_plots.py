import argparse

import visual_behavior.visualization.qc.container_plots as cp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ophys_container_id", type=int,
                        help="container ID to process")
    args = parser.parse_args()
    ophys_container_id = args.ophys_container_id

    print('ophys_container_id =', ophys_container_id)
    # cp.plot_average_timeseries_for_container(ophys_container_id, save_figure=True)

    cp.plot_pupil_timeseries_for_container(ophys_container_id, save_figure=True)
