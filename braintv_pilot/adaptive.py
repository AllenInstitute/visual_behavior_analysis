

"""
Inter-session adaptive

approach:

run a script daily (via chmod or something else) which analyzes recent history and sets the next parameters


"""
import os
# try:
#     from transitions.extensions import GraphMachine as Machine
# except ImportError:
#     print 'could not import GraphMachine'
from collections import OrderedDict
import pandas as pd
from transitions import Machine
from sklearn.metrics import accuracy_score
from dro import utilities as dro
from . import basepath
from . import metrics
from .latest import local_dir,copy_latest
from .data import annotate_trials

STATE_STORE = '/data/neuralcoding/justin/training_stages'


def compute_metrics(group):
    result = {
        'accuracy': metrics.discrim(group,'change','detect',metric=accuracy_score),
        'd-prime': metrics.discrim(group,'change','detect',metric=metrics.d_prime),
        'd-prime_peak': metrics.peak_dprime(group),
        'discrim_p': metrics.discrim(group,'change','detect',metric=metrics.discrim_p),
        'response_bias': metrics.response_bias(group,'detect'),
        'earned_water': metrics.earned_water(group), 
        'total_water': metrics.total_water(group), 
        'num_trials': metrics.num_trials(group),
        'num_contingent_trials': metrics.num_contingent_trials(group),
        'reaction_time_50th%ile': metrics.reaction_times(group,percentile=50),
        'reaction_time_05th%ile': metrics.reaction_times(group,percentile=5),
        'reaction_time_95th%ile': metrics.reaction_times(group,percentile=95),
    }

    return pd.Series(result, name='metrics')

class DoC(object):
    """docstring for Mouse"""

    def __init__(self, mouse):
        super(DoC, self).__init__()

        self.state_parameters = OrderedDict()

        self.state_parameters['full_field_gratings'] = {} # add parameters to this dict
        self.state_parameters['windowed_gratings'] = {}
        self.state_parameters['flash_100ms'] = {}
        self.state_parameters['flash_300ms'] = {}
        self.state_parameters['flash_500ms'] = {}

        self.default_initial_state = 'full_field_gratings'

        self.mouse = mouse
        self.mouse_dir = os.path.join(basepath,mouse)
        self.state_txt = 'training_stage_{}.txt'.format(mouse)

        self.trials = None
        self.session_summary = None

        # define the state machine
        self.machine = Machine(
            model=self,
            states=self.state_parameters.keys(),
            initial=self.load_latest_state(),
            after_state_change='_save_state',
            )

        self.machine.add_transition(
            trigger='progress',
            source='full_field_gratings',
            dest='windowed_gratings',
            conditions='two_out_of_three_aint_bad',
            )

        self.machine.add_transition(
            trigger='progress',
            source='windowed_gratings',
            dest='flash_100ms',
            conditions='two_out_of_three_aint_bad',
            )

        self.machine.add_transition(
            trigger='progress',
            source='flash_100ms',
            dest='flash_300ms',
            conditions='two_out_of_three_aint_bad',
            )

        self.machine.add_transition(
            trigger='progress',
            source='flash_300ms',
            dest='flash_500ms',
            conditions='new_day',
            )

        self.machine.add_transition(
            trigger='progress',
            source='flash_500ms',
            dest='flash_500ms',
            )

        self._new_day = False

        self.load_trials()
        self.analyze_trials()

    def load_trials(self):
        copy_latest()

        self.trials = annotate_trials(
            dro.load_from_folder(
                local_dir,
                load_existing_dataframe=False,
                save_dataframe=False,
                filename_contains=self.mouse,
                )
            )
        assert len(self.trials['mouse_id'].unique())==1

    def analyze_trials(self):

        self.session_summary = (
            self.trials
                .groupby(['mouse_id','training_day'])
                .apply(compute_metrics)
                .reset_index()
                .set_index('training_day')
                )

    # def update_params(self):
    #     with open(os.path.join(self.mouse_dir,'adjustments'))

    def _save_state(self):
        with open(self.state_txt,'wb') as f:
            f.write(self.state)


    def _load_state(self):
        with open(self.state_txt,'rb') as f:
            return f.read().strip()

    def load_latest_state(self):
        try:
            return self._load_state()
        except IOError:
            return self.default_initial_state # replace with actually loading the prior state


    def two_out_of_three_aint_bad(self):
        # check recent history in self.df to see if we've met the criteria

        criteria = (self.session_summary[-3:]['d-prime_peak']>2).sum() > 1

        if criteria==True:
            self._two_of_three = False
            return True
        else:
            return False

    def new_day(self):
        # is it a new day?

        if self._new_day==True:
            self._new_day = False
            return True
        else:
            return False

"""

Intra-session adaptive (e.g. for adaptive approach to ITI)

there are two approaches that could be employed here.

1. define the adaptive procedure locally, within the script. (as in opyrant)
- replace the main execution while loop with a generator which generates new parameters on each trial
- the generator needs access to trial history

2. run a daemon that monitors the publisher and uses a ZRO Proxy to update parameters on the fly.
- improve the log handling in the Stim base class
- add an `update_params(**kwargs)` to the stim base class that updates the parameter AND logs the change (ideally including the frame when the change occured.)
- when and where does the daemon launch? do we just run it on a server? is it spun off from the script in a subprocess?


"""

