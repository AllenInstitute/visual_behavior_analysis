
import os

def get_demixed_exp(unmix_traces_path, meso_data) :

    exp_list = meso_data.experiment_id
    exp_succeeded = os.listdir(unmix_traces_path)
    exp_demixed=[]
    exp_failed=[]

    for exp in exp_list.values :
        if str(exp) in exp_succeeded :
            exp_demixed.append(str(exp))
        else :
            exp_failed.append(str(exp))

    return exp_demixed, exp_failed