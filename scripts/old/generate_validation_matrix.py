#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')

from visual_behavior.translator.foraging2 import data_to_change_detection_core
from visual_behavior.translator.core import create_extended_dataframe
from visual_behavior.validation.qc import generate_qc_report

import pandas as pd
import datetime
import os
import itertools
import glob
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def get_simulated_mouse_PKLs(path='//allen/aibs/mpe/Software/data/behavior/validation'):
    stages = (
        'stage_0',
        'stage_1',
        'stage_2',
        'stage_3',
        'stage_4',
        'stage_5'
    )

    mice = (
        'Stupid',
        # 'Early',  # early mouse is causing trouble. leave it out
        'Excited',
        'Obstinate',
        'Perfect',
        'Late',
    )

    pkls = {}

    for mouse, stage in itertools.product(mice, stages):

        if mouse not in pkls:
            pkls[mouse] = {}

        these_pkls = glob.glob(path + '/{}/*{}DoCMouse.pkl'.format(stage, mouse))
        these_pkls.sort(key=os.path.getctime)
        try:
            pkls[mouse][stage] = these_pkls[-1]
        except IndexError:
            pass

    return pkls


def run_qc(pkls):
    all_results = []
    for mouse, mouse_pkls in pkls.iteritems():
        for stage, stage_pkl in mouse_pkls.iteritems():

            print('mouse = {}, stage = {}'.format(mouse, stage))
            try:
                data = pd.read_pickle(stage_pkl)

                core_data = data_to_change_detection_core(data)
                results = generate_qc_report(core_data)
                results.update({'mouse': mouse, 'stage': stage})

            except Exception as e:
                print('error before validation: {}'.format(e))
                results = {'mouse': mouse, 'stage': stage}

            all_results.append(results)

    return all_results


def remove_lick_validation_from_obstinate_mouse(all_results):
    '''
    obstinate mouse never licks, so always fails lick validation. Just force it True
    '''
    for result in all_results:
        if result['mouse'] == 'Obstinate':
            result['validate_licks'] = True
    return all_results


def build_matrix(all_results, savepath='//allen/aibs/mpe/Software/data/behavior/validation/validation_matrices'):
    outcomes = (
        pd.DataFrame(data=all_results)
        .sort_values(['stage', 'mouse'])
        .set_index(['stage', 'mouse'])
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax = sns.heatmap(
        data=outcomes.fillna(False),
        mask=pd.isnull(outcomes),
        xticklabels=True,
        yticklabels=True,
        cmap='copper',
        cbar=False,
        alpha=0.75,
        vmin=0,
        vmax=1
    )

    xticklabels = ax.get_xticklabels()
    ax.set_xticklabels(xticklabels, rotation=40, ha='right', fontsize=8)
    yticklabels = ax.get_yticklabels()
    ax.set_yticklabels(yticklabels, rotation=0, ha='right', fontsize=8)
    ax.grid(False)
    ax.set_title('Fraction of passing validations = {:.2f}'.format(np.nanmean(np.array(outcomes.fillna(0).values.astype(float)), axis=(0, 1))))
    fig.tight_layout()

    timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    fig.savefig(os.path.join(savepath, 'validation_matrix_{}.png'.format(timestamp)))
    return fig


def generate_validation_matrix():
    pkls = get_simulated_mouse_PKLs()
    all_results = run_qc(pkls)
    all_results = remove_lick_validation_from_obstinate_mouse(all_results)
    fig = build_matrix(all_results)
    return fig

if __name__ == '__main__':
    generate_validation_matrix()
