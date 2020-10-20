from visual_behavior.visualization.qc import container_plots as cp
import argparse


def main():
    possible_plots = {
        "ophys_session_sequence": cp.plot_container_session_sequence,
        "max_projection_images": cp.plot_sdk_max_projection_images_for_container,
        "average_images": cp.plot_sdk_average_images_for_container,
        "dff_traces_heatmaps": cp.plot_dff_traces_heatmaps_for_container,
        "running_speed": cp.plot_running_speed_for_container,
        "lick_rasters": cp.plot_lick_rasters_for_container,
        "average_intensity_timeseries": cp.plot_average_intensity_timeseries_for_container,
        "motion_correction_xy_shift": cp.plot_motion_correction_xy_shift_for_container,
        "number_matched_cells": cp.plot_number_matched_cells_for_container,
        "fraction_matched_cells": cp.plot_fraction_matched_cells_for_container,
        "cell_matching_registration_overlay_grid": cp.plot_cell_matching_registration_overlay_grid,
        "cell_matching_registration_output": cp.plot_cell_matching_registration_output,
        "eye_tracking_sample_frames": cp.plot_eye_tracking_sample_frames,
        "segmented_rois_by_experiment": cp.plot_number_segmented_rois_for_container,
        "segmentation_masks": cp.plot_segmentation_masks_for_container,
        "segmentation_mask_overlays": cp.plot_segmentation_mask_overlays_for_container,
        "max_projection_images_movies": cp.plot_movie_max_projection_images_for_container,
        "average_images_movies": cp.plot_movie_average_images_for_container,
        "cell_snr_by_experiment": cp.plot_cell_snr_for_container,
        "pupil_area": cp.plot_pupil_area,
        "pupil_position": cp.plot_pupil_position,
        "FOV_average_intensity": cp.plot_average_intensity_for_container,
        "pmt_settings": cp.plot_pmt_for_container,
        "snr_by_pmt": cp.plot_snr_by_pmt_for_container,
        "snr_by_pmt_and_intensity": cp.plot_snr_by_pmt_gain_and_intensity_for_container,
        "behavior_summary": cp.plot_behavior_summary,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--container-id", type=int,
                        help="Container ID to process")
    parser.add_argument("--plots", type=str, nargs='+', default=None,
                        help=(f"Which plots to create for a container."
                              f"Possible plots: {possible_plots.keys()}"))
    args = parser.parse_args()

    if args.plots:
        for plot_name in args.plots:
            try:
                possible_plots[plot_name](args.container_id)
            except KeyError:
                raise RuntimeError(f"'{plot_name}'' is not a valid plot option!")
    else:
        for plot_name, plot_callable in possible_plots.items():
            try:
                plot_callable(args.container_id)
            except Exception as e:
                print('{} failed for container {}, error:  {}'.format(plot_name, args.container_id, e))


if __name__ == "__main__":
    main()
