import os
import numpy as np
import pandas as pd

import nwb

from braintv_behav.io import data_or_pkl

@data_or_pkl
def create_settings(data):

    mouse = data['mouse_id']
    task = data['task_id']
    start_time = data['startdatetime'] # ISO 8601? or other standard?
    start_time_dt = pd.to_datetime(start_time)

    filename = '{}-{}.nwb'.format(mouse,session_dt.strftime("%Y%m%d%H%M%S"))
    identifier = nwb.create_identifier(filename[:-4])
    description = '{} performing the task "{}" on {}.'.format(mouse,task,start_time)

    return dict(
        filename = os.path.join(save_dir,filename),
        identifier = identifier,
        description = description,
        start_time = start_time,
        overwrite = True,
    )

@data_or_pkl
def load_template_df(data):
    stimdf = pd.DataFrame(data['stimuluslog'])

    template_df = (
        stimdf[['image_category','image_name']]
        .drop_duplicates()
        .sort_values(['image_category','image_name'])
        .reset_index()
    )
    del template_df['index']
    template_df['template_array_frame'] = template_df.index

    return template_df

def scrub_path(aibsdata_path):
    return (
        aibsdata_path
        .replace('/',os.path.sep)
        .replace('\\',os.path.sep)
        .replace('//aibsdata2','/data')
        .replace('//aibsdata','/data')
    )

@data_or_pkl
def create_image_templates(data):

    template_df = load_template_df(data)
    image_dict = pd.read_pickle(scrub_path(data['image_dict_path']))

    sample_image = image_dict[template_df.iloc[0]['image_category']][template_df.iloc[0]['image_name']]

    width,height = sample_image.shape
    shape = (
        len(template_df),width, height
    )
    image_templates = np.empty(shape,sample_image.dtype)

    for rr,row in template_df.iterrows():
        frame = row['template_array_frame']
        image_templates[frame,:,:] = image_dict[row['image_category']][row['image_name']]

    return image_templates

def save_image_templates(image_templates,borg):
    template = borg.create_timeseries("TimeSeries", "categorical_image_stack", "template")

    template.set_description('images that were presented in this session')
    template.set_comments('this is a comment')

    template.set_data(image_templates, unit='frame', conversion=1.0, resolution=1.0)
    template.ignore_time()

    template.set_value('dimension', image_templates.shape)
    template.set_value('format', 'raw')
    template.set_value('bits_per_pixel', 8)

    template.set_value('category',template_df['image_category'].values.astype('S1'))
    template.set_value('name',template_df['image_name'].values.astype('S5'))

    template.finalize()
    return template

@data_or_pkl
def load_basetime(data):
    vsync = np.hstack((0,data['vsyncintervals']))
    return (vsync.cumsum()) / 1000.0

@data_or_pkl
def create_nwb(data):

    settings = create_settings(data)

    borg = nwb.NWB(
        **settings
    )

    image_templates = create_image_templates(data)
    save_image_templates(image_templates,borg)
