import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.processing as processing
import visual_behavior.database as db
import argparse
import pandas as pd
import time

parser = argparse.ArgumentParser(description='log dff stats')
parser.add_argument('--oeid', type=int, default=0, metavar='ophys experiment ID')
parser.add_argument(
        '--verbose',
        help='boolean, not followed by an argument. Enables verbose mode. False by default.',
        action='store_true'
    )

def main(oeid, verbose=True):
    t0 = time.time()
    # load the session
    session = loading.get_ophys_dataset(oeid)
    if verbose:
        print('done loading session at {:0.2f} seconds'.format(time.time() - t0))

    # add dff stats to the cell_specimen_table
    df = processing.add_dff_stats_to_specimen_table(session)
    if verbose:
        print('done adding dff_stats at {:0.2f} seconds'.format(time.time() - t0))
    
    # add a column containing the ophys_experiment_id
    df['ophys_experiment_id'] = oeid
    if verbose:
        print('done appending oeid {:0.2f} seconds'.format(time.time() - t0))
        print('full list of dataframe columns:')
        for column in df.columns:
            print('\t{}'.format(column))

    # convert the df to a list of dicts:
    records = df.drop(columns=['image_mask','roi_mask']).reset_index().to_dict(orient='records')
    if verbose:
        print('done converting df to list of dicts at {:0.2f} seconds'.format(time.time() - t0))

    # for each row, log the record to mongo
    for record in records:
        if verbose:
            print('logging record for roi_id {} to mongo at {:0.2f} seconds'.format(record['cell_roi_id'], time.time() - t0))
        db.log_cell_dff_data(record)
    if verbose:
        print('done logging records to mongo at {:0.2f} seconds'.format(time.time() - t0))

if __name__ == "__main__":
    args = parser.parse_args()

    if args.oeid != 0:
        main(args.oeid, args.verbose)
    else:
        print('must enter an ophys_experiment_id. Use the --oeid flag, followed by the ID')