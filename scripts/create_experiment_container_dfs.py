import os
import pandas as pd
import numpy as np


import matplotlib
matplotlib.use('Agg')

import pandas as pd
from visual_behavior.ophys.io.create_multi_session_mean_df import get_multi_session_mean_df

if __name__ == '__main__':
    # save_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\BehaviorImaging\DoC\n_way_cell_matching_validation\integration_test'
    # cache_dir = r'\\allen\programs\braintv\workgroups\ophysdev\OPhysCore\Analysis\2018-08 - Behavior Integration test'

    cre_line_dict = {713968361:'Slc17a7',
                    710324788:'Slc17a7',
                    713426167:'Slc17a7',
                    716520611:'Slc17a7',
                    734705403:'Sst',
                    705093670:'Slc17a7',
                     692785818:'Slc17a7',
                     710269829:'Sst',
                     694420833:'Sst',
                     707282086:'Sst',
                     719239258:'Vip',
                    }

    container_dict = {713968361:745083659,
                    710324788:745084555,
                    713426167:745083990,
                    716520611:759756815,
                    734705403:760842228,
                    705093670:760838377,
                     692785818:760838250,
                     710269829:759199675,
                     694420833:None,
                     707282086:None,
                     719239258:None,
                    }

    # containers = [
    #     r'\\allen\programs\braintv\production\visualbehavior\prod0\specimen_713968361\experiment_container_745083659',
    #     r'\\allen\programs\braintv\production\visualbehavior\prod0\specimen_710324788\experiment_container_745084555',
    #     r'\\allen\programs\braintv\production\visualbehavior\prod0\specimen_713426167\experiment_container_745083990',
    #     r'\\allen\programs\braintv\production\visualbehavior\prod0\specimen_716520611\experiment_container_759756815',
    #     r'\\allen\programs\braintv\production\visualbehavior\prod0\specimen_734705403\experiment_container_760842228',
    #     r'\\allen\programs\braintv\production\visualbehavior\prod0\specimen_705093670\experiment_container_760838377',
    #     r'\\allen\programs\braintv\production\visualbehavior\prod0\specimen_692785818\experiment_container_760838250',
    #     r'\\allen\programs\braintv\production\visualbehavior\prod0\specimen_710269829\experiment_container_759199675'
    #     ]

    # get df of experiment container metadata
    df_list = []
    for specimen_id in np.sort(container_dict.keys()):
        container_path = [container for container in containers if str(specimen_id) in container]
        if len(container_path)>0:
            container_path = container_path[0]
        else:
            container_path = None
        df_list.append([specimen_id, container_dict[specimen_id], cre_line_dict[specimen_id], container_path])
    container_info = pd.DataFrame(df_list,columns=['specimen_id','container_id','cre_line','container_path'])
    container_info.to_csv(os.path.join(cache_dir,'cell_matching_results','container_info.csv'),)


    # get df of experiment sessions for container IDs
    df_list = []
    container_ids = []
    container_ids = np.sort(container_info[np.isnan(container_info.container_id)==False].container_id.values)
    container_ids = [int(container_id) for container_id in container_ids]
    for container_id in container_ids:
        container_path = container_info[container_info.container_id==container_id].container_path.values[0]
        input_json = [file for file in os.listdir(container_path) if '_input.json' in file]
        json = pd.read_json(os.path.join(container_path,input_json[0]))

        lims_ids = []
        for i in range(len(json.experiment_containers.ophys_experiments)):
            lims_id = int(json.experiment_containers.ophys_experiments[i]['id'])
            lims_ids.append(lims_id)
        row = lims_ids
        df_list.append(row)

    container_df = pd.DataFrame(df_list)
    container_df = container_df.T
    container_df.columns = container_ids
    container_df.to_csv(os.path.join(cache_dir,'cell_matching_results','container_df.csv'))

#
# import tifffile
# registration_images = [file for file in os.listdir(folder) if 'register' in file]


