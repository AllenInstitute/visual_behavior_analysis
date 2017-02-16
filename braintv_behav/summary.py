import os
import click
from dro import utilities as dro
from . import basepath

@click.command()
@click.option('--mouse', prompt='Mouse to compile.')
def load_and_save(mouse):

    mouse_pkl_dir = os.path.join(basepath,mouse,'output')

    trials = dro.load_from_folder(
        mouse_pkl_dir,
        load_existing_dataframe=False,
        save_dataframe=False,
        filename_contains=mouse,
        )

    trials.to_csv('{}_all_trials.csv'.format(mouse))

if __name__ == '__main__':
    load_and_save()