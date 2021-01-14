"""
Created on Saturday October 13 2018

@author: marinag
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.dataset.cell_matching_dataset import CellMatchingDataset

# from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
# from visual_behavior.ophys.plotting import experiment_summary_figures as esf
from visual_behavior.ophys.plotting import summary_figures as sf

import logging

logger = logging.getLogger(__name__)


def get_cache_dir():
    cache_dir = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/cell_matching_validation'
    return cache_dir


def get_container_info():
    container_info = pd.read_csv(os.path.join(get_cache_dir(), 'cell_matching_results', 'container_info.csv'))
    return container_info


def get_container_df():
    container_df = pd.read_csv(os.path.join(get_cache_dir(), 'cell_matching_results', 'container_df.csv'))
    return container_df


def get_container_analysis_dir(container_id):
    container_info = get_container_info()
    info = container_info[container_info.container_id == container_id]
    specimen = info.specimen_id.values[0]
    cre_line = info.cre_line.values[0]
    folder = str(int(container_id)) + '_' + str(int(specimen)) + '_' + cre_line
    container_analysis_dir = os.path.join(get_cache_dir(), 'cell_matching_results', folder)
    if not os.path.exists(container_analysis_dir):
        os.mkdir(container_analysis_dir)
    return container_analysis_dir


def get_lims_ids_for_container(container_id):
    container_df = get_container_df()
    lims_ids = container_df[str(int(container_id))].values
    lims_ids = lims_ids[np.isnan(lims_ids) == False]
    lims_ids = [int(lims_id) for lims_id in lims_ids]
    lims_ids = np.sort(lims_ids)
    return lims_ids


def get_cell_matching_dataset_dict(lims_ids, cache_dir):
    cell_matching_dataset_dict = {}
    for lims_id in lims_ids:
        cell_matching_dataset_dict[lims_id] = CellMatchingDataset(lims_id, cache_dir=cache_dir,
                                                                  from_processed_data=False)
    return cell_matching_dataset_dict


def get_ssim(img0, img1):
    from skimage.measure import compare_ssim as ssim
    ssim_pair = ssim(img0, img1, gaussian_weights=True)
    return ssim_pair


def get_registration_df(container_id, cell_matching_dataset_dict):
    import tifffile
    container_info = get_container_info()
    container_path = container_info[container_info.container_id == container_id].container_path.values[0]
    registration_images = [file for file in os.listdir(container_path) if 'register' in file]
    lims_ids = get_lims_ids_for_container(container_id)
    df_list = []
    for y, registration_image in enumerate(registration_images):
        registered_expt = registration_image.split('_')[3]
        if int(registered_expt) in lims_ids:
            target_expt = registration_image.split('_')[7].split('.')[0]
            if int(target_expt) in lims_ids:
                reg = tifffile.imread(os.path.join(container_path, registration_image))
                reg = reg / float(np.amax(reg))
                reg = reg.astype('float32')
                data = cell_matching_dataset_dict[int(target_expt)]
                avg_image = data.average_image
                avg_im_target = avg_image / float(np.amax(avg_image))
                ssim = get_ssim(avg_im_target, reg)
                df_list.append([target_expt, registered_expt, ssim])  # ,avg_im_target,avg_im_candidate,reg,image])
    columns = ['target_expt', 'registered_expt', 'ssim']
    reg_df = pd.DataFrame(df_list, columns=columns)
    container_analysis_dir = get_container_analysis_dir(container_id)
    reg_df.to_csv(os.path.join(container_analysis_dir, 'registration_results.csv'))
    return reg_df


def plot_registration_results(container_id, cell_matching_dataset_dict):
    import tifffile
    container_info = get_container_info()
    container_path = container_info[container_info.container_id == container_id].container_path.values[0]
    registration_images = [file for file in os.listdir(container_path) if 'register' in file]
    lims_ids = get_lims_ids_for_container(container_id)
    n_images = len(registration_images)
    n_per_plot = 8
    intervals = np.arange(0, n_images + n_per_plot, n_per_plot)
    for x, interval in enumerate(intervals):
        figsize = (20, 15)
        fig, ax = plt.subplots(4, n_per_plot, figsize=figsize)
        ax = ax.ravel()
        i = 0
        if x < len(intervals) - 1:
            for y, registration_image in enumerate(registration_images[intervals[x]:intervals[x + 1]]):
                registered_expt = registration_image.split('_')[3]
                if int(registered_expt) in lims_ids:
                    target_expt = registration_image.split('_')[7].split('.')[0]
                    if int(target_expt) in lims_ids:
                        reg = tifffile.imread(os.path.join(container_path, registration_image))
                        reg = reg / float(np.amax(reg))
                        reg = reg.astype('float32')
                        data = cell_matching_dataset_dict[int(target_expt)]
                        avg_image = data.average_image
                        avg_im_target = avg_image / float(np.amax(avg_image))
                        data = cell_matching_dataset_dict[int(registered_expt)]
                        avg_image = data.average_image
                        avg_im_candidate = avg_image / float(np.amax(avg_image))
                        image = np.empty((reg.shape[0], reg.shape[1], 3))
                        image[:, :, 0] = avg_im_target
                        image[:, :, 1] = reg
                        ax[i].imshow(avg_im_candidate, cmap='gray')
                        ax[i].set_title('candidate\n' + registered_expt)
                        ax[i + n_per_plot].imshow(avg_im_target, cmap='gray')
                        ax[i + n_per_plot].set_title('target\n' + target_expt)
                        ax[i + 2 * n_per_plot].imshow(reg, cmap='gray')
                        ax[i + 2 * n_per_plot].set_title('registered\n' + registered_expt)
                        ssim = get_ssim(avg_im_target, reg)
                        ax[i + 3 * n_per_plot].imshow(image)
                        ax[i + 3 * n_per_plot].set_title('ssim:' + str(np.round(ssim, 3)))
                        for xx in range(len(ax)):
                            ax[xx].axis('off')
                        i += 1

            save_dir = get_container_analysis_dir(container_id)
            sf.save_figure(fig, figsize, save_dir, 'figures', 'registration_overlays_' + str(x))
            plt.close()
        plt.close()


def get_ssim_matrix(lims_ids, reg_df):
    matrix = np.ones((len(lims_ids), len(lims_ids)))
    for i, lims_id1 in enumerate(lims_ids):
        for j, lims_id2 in enumerate(lims_ids):
            row = reg_df[(reg_df.target_expt == str(lims_id1)) & (reg_df.registered_expt == str(lims_id2))]
            if len(row) > 0:
                ssim = row.ssim.values[0]
                matrix[i, j] = ssim
                matrix[j, i] = ssim
    return matrix


def get_stim_names(cell_matching_dataset_dict, lims_ids):
    stim_names = []
    for lims_id in lims_ids:
        data = cell_matching_dataset_dict[lims_id]
        data.get_lims_data()
        stim_name = data.lims_data.experiment_name.values[0].split('_')[-1]
        stim_names.append(stim_name)
    return stim_names


def plot_ssim_matrix(matrix, container_id, lims_ids, cell_matching_dataset_dict, label='stim_names'):
    if label == 'stim_names':
        stim_names = get_stim_names(cell_matching_dataset_dict, lims_ids)
        labels = stim_names
        fig_title = 'registered_ssim_matrix_stim_names'
    else:
        labels = lims_ids
        fig_title = 'registered_ssim_matrix'
    figsize = (8, 8)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(matrix, square=True, cmap='magma', vmin=0.8, ax=ax, cbar_kws={'shrink': 0.7})
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels, rotation=0)
    ax.set_title('ssim for registered pairs')
    plt.suptitle(container_id, x=0.5, y=0.93, horizontalalignment='center')
    fig.tight_layout()
    save_dir = get_container_analysis_dir(container_id)
    sf.save_figure(fig, figsize, save_dir, 'figures', fig_title)
    plt.close()


def plot_ssim_values(reg_df, container_id):
    figsize = (6, 5)
    fig, ax = plt.subplots(figsize=figsize)
    lims_ids = get_lims_ids_for_container(container_id)
    for i, lims_id in enumerate(lims_ids):
        ssims = reg_df[reg_df.target_expt == str(lims_id)].ssim.values
        for ssim in ssims:
            ax.plot(i, ssim, 'o', color=sns.color_palette()[0])
        ax.set_xticks(np.arange(0, len(lims_ids)))
        ax.set_xticklabels(lims_ids, rotation=90)
        ax.set_xlabel('lims_id of target image')
        ax.set_ylabel('ssim of registered image')
        ax.legend()
    fig.tight_layout()
    save_dir = get_container_analysis_dir(container_id)
    sf.save_figure(fig, figsize, save_dir, 'figures', 'ssim_values')
    plt.close()


def get_lims_ids_for_matching_results(container_id):
    container_info = get_container_info()
    container_path = container_info[container_info.container_id == container_id].container_path.values[0]

    input_json = [file for file in os.listdir(container_path) if '_input.json' in file]
    json = pd.read_json(os.path.join(container_path, input_json[0]))

    result_lims_ids = []
    for i in range(len(json.experiment_containers.ophys_experiments)):
        result_lims_ids.append(json.experiment_containers.ophys_experiments[i]['id'])
    return result_lims_ids


def get_cell_matching_results_cell_indices(container_id):
    container_info = get_container_info()
    container_path = container_info[container_info.container_id == container_id].container_path.values[0]
    matching_result = os.path.join(container_path, "matching_result.txt")
    lims_ids = get_lims_ids_for_matching_results(container_id)
    df = pd.read_csv(matching_result, delim_whitespace=True, header=None, index_col=len(lims_ids), names=lims_ids)
    df.to_csv(os.path.join(get_container_analysis_dir(container_id),
                           str(int(container_id)) + '_matching_results_cell_indices.csv'))
    return df


def get_cell_matching_results_cell_specimen_ids(container_id):
    container_info = get_container_info()
    container_path = container_info[container_info.container_id == container_id].container_path.values[0]
    output_json = [file for file in os.listdir(container_path) if '_output.json' in file]
    json = pd.read_json(os.path.join(container_path, output_json[0]))
    lims_ids = get_lims_ids_for_matching_results(container_id)
    df = get_cell_matching_results_cell_indices(container_id)
    new_df_list = []
    for matching_cell_id in df.index:
        specimen_list = []
        c = 0
        for i, lims_id in enumerate(lims_ids):
            if df[df.index == matching_cell_id][lims_id].values[0] != -1:
                specimen_id = json[json.index == matching_cell_id]['cell_rois'].values[0][c]
                c += 1
            else:
                specimen_id = -1
            specimen_list.append(specimen_id)
        new_df_list.append(specimen_list)
    matching_df = pd.DataFrame(new_df_list, columns=lims_ids, index=df.index)
    matching_df.to_csv(os.path.join(get_container_analysis_dir(container_id),
                                    str(int(container_id)) + '_matching_results_cell_specimen_ids.csv'))
    return matching_df


def get_valid_cell_matching_results_cell_specimen_ids(container_id, cell_matching_dataset_dict):
    matching_df = get_cell_matching_results_cell_specimen_ids(container_id)
    valid_matching_df = pd.DataFrame(columns=matching_df.keys())
    for lims_id in matching_df.keys():
        dataset = cell_matching_dataset_dict[lims_id]
        new_cell_list = []
        for row in range(len(matching_df)):
            cell_id = matching_df[lims_id].iloc[row]
            if cell_id != -1:
                if len(dataset.roi_df[dataset.roi_df.id == cell_id]) > 0:  # if it exists
                    if dataset.roi_df[dataset.roi_df.id == cell_id].valid.values[0]:  # if its valid
                        new_cell_list.append(cell_id)  # include
                    else:
                        new_cell_list.append(-1)  # else dont include
                else:
                    print('**cell doesnt exist**')
                    new_cell_list.append(-1)
            else:
                new_cell_list.append(-1)
        valid_matching_df[lims_id] = new_cell_list
    valid_matching_df.to_csv(os.path.join(get_container_analysis_dir(container_id),
                                          str(int(container_id)) + '_matching_results_valid_cell_specimen_ids.csv'))
    return valid_matching_df


def get_matching_cell_counts(valid_matching_df):
    df = valid_matching_df.copy()
    keep_list = []
    for unique_cell_id in df.index:
        n_matches = len(np.where(df[df.index == unique_cell_id].values != -1)[0])
        if n_matches > 0:
            keep_list.append(int(unique_cell_id))
    # counts[n_matches] = counts[n_matches]+1
    df = df[df.index.isin(keep_list)]
    counts = pd.DataFrame(columns=np.arange(0, len(valid_matching_df.keys())) + 1, index=['n_matches'])
    for key in counts.keys():
        counts[key] = 0
    for matching_cell_id in df.index:
        n_matches = len(np.where(df[df.index == matching_cell_id].values != -1)[0])
        counts.loc['n_matches', n_matches] = counts[n_matches].values[0] + 1
    return counts


def get_cumulative_matching_cell_counts(counts):
    cdf = pd.DataFrame(columns=counts.keys(), index=['# cells'])
    cumulative_count = 0
    for i, count in enumerate(np.sort(counts.keys())[::-1]):
        current_count = counts[count].values[0]
        cumulative_count = current_count + cumulative_count
        cdf[count] = cumulative_count
    return cdf


def plot_cumulative_matching_cell_counts(cdf, container_id, ymax=None):
    figsize = (6, 4)
    fig, ax = plt.subplots(figsize=figsize)
    for count in np.sort(cdf.keys()):
        ax.plot(count, cdf[count], 'o')
        ax.set_title('cumulative number of cells matched')
        ax.set_ylabel('# cells matched')
        ax.set_xlabel('# sessions')
        ax.set_xticks(np.arange(1, len(cdf.keys()) + 1, 1))
        ax.set_ylim(ymin=0)
        if ymax is not None:
            ax.set_ylim(ymax=ymax)
    plt.suptitle(str(int(container_id)), x=0.55, y=1.03, horizontalalignment='center')
    fig.tight_layout()
    save_dir = get_container_analysis_dir(container_id)
    sf.save_figure(fig, figsize, save_dir, 'figures', 'cumulative_matched_cells_' + str(ymax))


def get_cell_matching_matrix(container_id, cell_matching_dataset_dict):
    df = get_valid_cell_matching_results_cell_specimen_ids(container_id, cell_matching_dataset_dict)
    lims_ids = df.keys()
    matrix = np.empty((len(lims_ids), len(lims_ids)))
    for i, lims_id1 in enumerate(lims_ids):
        for j, lims_id2 in enumerate(lims_ids):
            for unique_cell_id in df.index:
                tmp = df[(df.index == unique_cell_id)]
                tmp = tmp[[lims_id1, lims_id2]]
                if len(np.where(tmp.values != -1)[0]) == 2:
                    matrix[i, j] = matrix[i, j] + 1
    return matrix


def plot_matched_cells_matrix(container_id, cell_matching_dataset_dict, label='stim_names'):
    lims_ids = get_lims_ids_for_container(container_id)
    matrix = get_cell_matching_matrix(container_id, cell_matching_dataset_dict)
    if label == 'stim_names':
        stim_names = get_stim_names(cell_matching_dataset_dict, lims_ids)
        labels = stim_names
        fig_title = 'n_matched_cells_matrix_stim_names'
    else:
        labels = lims_ids
        fig_title = 'n_matched_cells_matrix'
    figsize = (10, 10)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(matrix, square=True, cmap='magma', ax=ax, cbar_kws={'shrink': 0.7}, annot=False, fmt='.3g')
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels, rotation=0)
    ax.set_title('# matched cells')
    plt.suptitle(str(int(container_id)), x=0.48, y=0.92, horizontalalignment='center')
    fig.tight_layout()
    save_dir = get_container_analysis_dir(container_id)
    sf.save_figure(fig, figsize, save_dir, 'figures', fig_title)


def plot_fraction_matched_cells_matrix(container_id, cell_matching_dataset_dict, label='stim_names'):
    lims_ids = get_lims_ids_for_container(container_id)
    matrix = get_cell_matching_matrix(container_id, cell_matching_dataset_dict)
    if label == 'stim_names':
        stim_names = get_stim_names(cell_matching_dataset_dict, lims_ids)
        labels = stim_names
        fig_title = 'fraction_matched_cells_matrix_stim_names'
    else:
        labels = lims_ids
        fig_title = 'fraction_matched_cells_matrix'

    new_matrix = matrix.copy()
    for i, row in enumerate(range(matrix.shape[0])):
        new_matrix[row] = matrix[row] / matrix[row][i]
    figsize = (10, 10)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(new_matrix, square=True, cmap='magma', vmin=0., vmax=1, ax=ax, cbar_kws={'shrink': 0.7},
                     annot=False, fmt='.1g')
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels, rotation=0)
    ax.set_title('fraction matched cells')
    plt.suptitle(str(int(container_id)), x=0.43, y=0.92, horizontalalignment='center')
    fig.tight_layout()
    save_dir = get_container_analysis_dir(container_id)
    sf.save_figure(fig, figsize, save_dir, 'figures', fig_title)


def get_coordinates(mask):
    m = mask.copy()
    (y, x) = np.where(m == 1)
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    return xmin, xmax, ymin, ymax


def plot_cell_zoom(dataset, cell_specimen_id, unique_cell_id, background_image, spacex=10, spacey=10, show_mask=False,
                   save=False, ax=None):
    if ax is None:
        figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)

    cell_mask = dataset.roi_dict[cell_specimen_id]
    mask = np.empty(cell_mask.shape)
    mask[cell_mask == 0] = np.nan
    mask[cell_mask == 1] = 1

    xmin, xmax, ymin, ymax = get_coordinates(mask)
    ax.imshow(background_image, cmap='gray', vmin=0, vmax=np.amax(background_image))

    if show_mask:
        ax.imshow(mask, cmap='jet', alpha=0.3, vmin=0, vmax=1)

    ax.set_xlim(xmin - spacex, xmax + spacex)
    ax.set_ylim(ymin - spacey, ymax + spacey)

    #     roi_data = dataset.roi_metrics[dataset.roi_metrics.id==cell_specimen_id]
    #     ax.set_xlim(roi_data[' minx'].values[0] - spacex, roi_data[' maxx'].values[0] + spacex)
    #     ax.set_ylim(roi_data[' miny'].values[0] - spacey, roi_data[' maxy'].values[0] + spacey)
    if cell_specimen_id != -1:
        valid = dataset.roi_df[dataset.roi_df.id == cell_specimen_id].valid.values[0]
        ax.set_title('unique_id: ' + str(unique_cell_id) + '\nspecimen_id: ' +
                     str(cell_specimen_id) + '\nlims_id: ' + str(dataset.lims_id) +
                     '\nvalid: ' + str(valid))
    else:
        ax.set_title('unique_id: ' + str(unique_cell_id) + '\nspecimen_id: ' +
                     str(cell_specimen_id) + '\nlims_id: ' + str(dataset.lims_id))
    ax.grid(False)
    ax.axis('off')

    return ax, mask


def plot_max_proj_around_mask_coordinates(dataset, mask, unique_cell_id, cell_specimen_id, background_image, space,
                                          ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    xmin, xmax, ymin, ymax = get_coordinates(mask)
    ax.imshow(background_image, cmap='gray', vmin=0, vmax=np.amax(background_image))
    ax.set_xlim(xmin - space, xmax + space)
    ax.set_ylim(ymin - space, ymax + space)
    if cell_specimen_id != -1:
        valid = dataset.roi_df[dataset.roi_df.id == cell_specimen_id].valid.values[0]
        ax.set_title('unique_id: ' + str(unique_cell_id) + '\nspecimen_id: ' +
                     str(cell_specimen_id) + '\nlims_id: ' + str(dataset.lims_id) +
                     '\nvalid: ' + str(valid))
    else:
        ax.set_title('unique_id: ' + str(unique_cell_id) + '\nspecimen_id: ' +
                     str(cell_specimen_id) + '\nlims_id: ' + str(dataset.lims_id))
    ax.axis('off')
    return ax


def plot_cell_matching_validation(container_id, cell_matching_dataset_dict):
    lims_ids = get_lims_ids_for_container(container_id)
    df = get_cell_matching_results_cell_specimen_ids(container_id)

    dataset = cell_matching_dataset_dict[lims_ids[0]]
    cell_mask = dataset.roi_masks[dataset.cell_specimen_ids[0]]
    mask = np.empty(cell_mask.shape)
    mask[cell_mask == 0] = np.nan
    mask[cell_mask == 1] = 1
    x = len(lims_ids)
    for unique_cell_id in df.index:
        space = 50
        figsize = (20, 10)
        fig, ax = plt.subplots(2, x, figsize=figsize)
        ax = ax.ravel()
        tmp = df[df.index == unique_cell_id]
        for i, lims_id in enumerate(lims_ids):
            dataset = cell_matching_dataset_dict[lims_id]
            cell_specimen_id = tmp[lims_id].values[0]

            background_image = dataset.max_projection.data
            if cell_specimen_id != -1:
                ax[i], mask = plot_cell_zoom(dataset, cell_specimen_id, unique_cell_id, background_image, spacex=space,
                                             spacey=space, show_mask=True, save=False, ax=ax[i])
            else:
                ax[i] = plot_max_proj_around_mask_coordinates(dataset, mask, unique_cell_id, cell_specimen_id,
                                                              background_image, space, ax=ax[i])
                ax[i].set_title('lims_id: ' + str(lims_id), fontsize=10)

            background_image = dataset.average_projection.data
            if cell_specimen_id != -1:
                ax[i + x], mask = plot_cell_zoom(dataset, cell_specimen_id, unique_cell_id, background_image,
                                                 spacex=space, spacey=space, show_mask=True, save=False, ax=ax[i + x])
            else:
                ax[i + x] = plot_max_proj_around_mask_coordinates(dataset, mask, unique_cell_id, cell_specimen_id,
                                                                  background_image, space, ax=ax[i + x])
                ax[i + x].set_title('lims_id: ' + str(lims_id), fontsize=10)
        save_dir = get_container_analysis_dir(container_id)
        sf.save_figure(fig, figsize, os.path.join(save_dir, 'figures'), 'matching_validation', str(unique_cell_id))
        plt.close()
