import os
import numpy as np
import matplotlib as mpl
import seaborn as sns


def save_figure(fig, figsize, save_dir, folder, fig_title, formats=['.png']):
    fig_dir = os.path.join(save_dir, folder)
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    filename = os.path.join(fig_dir, fig_title)
    mpl.rcParams['pdf.fonttype'] = 42
    fig.set_size_inches(figsize)
    for f in formats:
        fig.savefig(filename + f, transparent=True, orientation='landscape', bbox_inches='tight', dpi=300)


def get_colors_for_session_numbers():
    reds = sns.color_palette('Reds_r', 6)[:5][::2]
    blues = sns.color_palette('Blues_r', 6)[:5][::2]
    return reds + blues


def lighter(color, percent):
    color = np.array(color)
    white = np.array([255, 255, 255])
    vector = white - color
    return color + vector * percent


def get_session_type_color_map():
    colors = np.floor(np.array([list(x) for x in get_colors_for_session_numbers()]) * 255).astype(np.uint8)
    black = np.array([0, 0, 0]).astype(np.uint8)

    session_type_color_map = {
        'OPHYS_0_images_A_habituation': lighter(colors[0, :], 0.8),
        'OPHYS_1_images_A': colors[0, :],
        'OPHYS_2_images_A_passive': colors[1, :],
        'OPHYS_3_images_A': colors[2, :],
        'OPHYS_4_images_B': colors[3, :],
        'OPHYS_5_images_B_passive': colors[4, :],
        'OPHYS_6_images_B': colors[5, :],

        'OPHYS_1_images_B': colors[3, :],
        'OPHYS_2_images_B_passive': colors[4, :],
        'OPHYS_3_images_B': colors[5, :],
        'OPHYS_4_images_A': colors[0, :],
        'OPHYS_5_images_A_passive': colors[1, :],
        'OPHYS_6_images_A': colors[2, :],

        'OPHYS_0_images_G_habituation': lighter(colors[3, :], 0.8),
        'OPHYS_1_images_G': colors[3, :],
        'OPHYS_2_images_G_passive': colors[4, :],
        'OPHYS_3_images_G': colors[5, :],
        'OPHYS_4_images_H': colors[0, :],
        'OPHYS_5_images_H_passive': colors[1, :],
        'OPHYS_6_images_H': colors[2, :],

        'OPHYS_7_receptive_field_mapping': lighter(black, 0.5),
        'None': black,
        None: black,
        np.nan: black,
        'VisCodingTargetedMovieClips': lighter(black, 0.5),
        'full_field_test': lighter(black, 0.2)}

    return session_type_color_map


def get_location_color(location, project_code):
    colors = sns.color_palette()
    if (project_code == 'VisualBehavior') or (project_code == 'VisualBehaviorTask1B'):
        location_colors = {'Slc17a7_VISp_175': colors[9],
                           'Slc17a7_VISp_375': colors[0],
                           'Vip_VISp_175': colors[4],
                           'Sst_VISp_275': colors[2],
                           'Sst_VISp_290': colors[2]}

    elif (project_code == 'VisualBehaviorMultiscope') or (project_code == 'VisualBehaviorMultiscope4areasx2d'):
        location = location.split('_')
        location = location[0] + '_' + location[1]
        location_colors = {'Slc17a7_VISp': sns.color_palette('Blues_r', 5)[0],
                           'Slc17a7_VISl': sns.color_palette('Blues_r', 5)[1],
                           'Slc17a7_VISal': sns.color_palette('Blues_r', 5)[2],
                           'Slc17a7_VISam': sns.color_palette('Blues_r', 5)[3],
                           'Slc17a7_VISpm': sns.color_palette('Blues_r', 5)[4],
                           'Vip_VISp': sns.color_palette('Purples_r', 5)[0],
                           'Vip_VISl': sns.color_palette('Purples_r', 5)[1],
                           'Vip_VISal': sns.color_palette('Purples_r', 5)[2],
                           'Vip_VISam': sns.color_palette('Purples_r', 5)[3],
                           'Vip_VISpm': sns.color_palette('Purples_r', 5)[4],
                           'Sst_VISp': sns.color_palette('Greens_r', 5)[0],
                           'Sst_VISl': sns.color_palette('Greens_r', 5)[1],
                           'Sst_VISal': sns.color_palette('Greens_r', 5)[2],
                           'Sst_VISam': sns.color_palette('Greens_r', 5)[3],
                           'Sst_VISpm': sns.color_palette('Greens_r', 5)[4]}

    return location_colors[location]


# def lighter(color, percent):
#     color = color * 255
#     color = np.array(color)
#     white = np.array([255, 255, 255])
#     return color + (white * percent)
#

def make_color_transparent(rgb_color, background_rgb=[255, 255, 255], alpha=0.5):
    return [alpha * c1 + (1 - alpha) * c2
            for (c1, c2) in zip(rgb_color, background_rgb)]
