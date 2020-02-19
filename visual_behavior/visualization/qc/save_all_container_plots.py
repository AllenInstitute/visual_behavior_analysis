from visual_behavior.visualization.qc import data_loading
from visua_behavior.visualization.qc import container_plots
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--container-id", type=int,
                    help="Container ID to process")
args = parser.parse_args()

def main():
    container_plots.plot_max_projection_images_for_container(args.container_id)
    container_plots.plot_eye_tracking_sample_frames(args.container_id)
    container_plots.plot_average_images_for_container(args.container_id)
    container_plots.plot_segmentation_masks_for_container(args.container_id)
    container_plots.plot_segmentation_mask_overlays_for_container(args.container_id)
    container_plots.plot_dff_traces_heatmaps_for_container(args.container_id)
    container_plots.plot_running_speed_for_container(args.container_id)
    container_plots.plot_lick_rasters_for_container(args.container_id)
    container_plots.plot_average_intensity_timeseries_for_container(args.container_id)
    container_plots.plot_number_matched_cells_for_container(args.container_id)
    container_plots.plot_fraction_matched_cells_for_container(args.container_id)
    container_plots.plot_motion_correction_xy_shift_for_container(ophys_container_id)

if __name__=="__main__":
    main()
