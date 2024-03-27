#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')

import warnings
import pandas as pd
from argschema import ArgSchema, ArgSchemaParser, fields
from visual_behavior.translator import foraging2, foraging
from visual_behavior.visualization.extended_trials.daily import make_daily_figure
from visual_behavior.translator.core import create_extended_dataframe

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


class ChangeDetectionDailyPlotSchema(ArgSchema):
    """docstring for DailyPlotSchema."""

    data_path = fields.String(
        description='path to change detection output file',
    )
    figure_path = fields.String(
        description='path to change detection output file',
    )


def make_and_save_plot(data_path,figure_path):

    data = pd.read_pickle(data_path)

    try:
        core_data = foraging.data_to_change_detection_core(data)
    except KeyError:
        warnings.warn('attempting to load foraging2 file. CAUTION: no support for custom "time"')
        core_data = foraging2.data_to_change_detection_core(data)

    extended_trials = create_extended_dataframe(
        trials=core_data['trials'],
        metadata=core_data['metadata'],
        licks=core_data['licks'],
        time=core_data['time'],
    )

    fig = make_daily_figure(extended_trials)
    fig.set_size_inches(11, 8.5)
    fig.savefig(
        figure_path,
        transparent=False,
        orientation='landscape',
        dpi=300,
    )
    return fig


def main():
    mod = ArgSchemaParser(schema_type=ChangeDetectionDailyPlotSchema)
    make_and_save_plot(
        data_path=mod.args['data_path'],
        figure_path=mod.args['figure_path'],
    )


if __name__ == '__main__':
    main()
