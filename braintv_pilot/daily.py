

"""
Inter-session adaptive

approach:

run a script daily (via chmod or something else) which analyzes recent history and sets the next parameters

"""
import os
import json
from collections import OrderedDict
import datetime as dt

import click
import yaml
from transitions import Machine

from dro import utilities as dro
from mouse_info.orm import Mouse as MouseBase

from . import basepath
from .latest import local_dir,copy_latest
from .data import annotate_trials
from .criteria import two_out_of_three_aint_bad


training_path = '/data/neuralcoding/Justin/training/'

# this will subclass Chris's MouseInfo object
class Mouse(MouseBase):
    """docstring for Mouse"""

    def __init__(self, id_):
        super(Mouse, self).__init__(id_)

        self.mouse_dir = os.path.join(basepath,id_)

        self._trials = None

        self.training_stages_path = os.path.join(
            training_path,
            id_,
            'training_stages.yml',
            )
        self._training_stages = None
        
        self.transitions_path = os.path.join(
            training_path,
            id_,
            'transitions.yml',
            )
        self._transitions = None
        self.current_stage_path = os.path.join(
            training_path,
            id_,
            'current_stage.txt',
            )
        self._current_stage = None

        self.initialize_machine()


    def initialize_machine(self):
        self._machine = Machine(
            model=self,
            states=self.training_stages.keys(),
            initial=self.current_stage,
            transitions=self.transitions,
            )

    @property
    def active_parameters(self):
        return self.training_stages[self.state]

    @property
    def training_stages(self):
        if self._training_stages is None:
            with open(self.training_stages_path,'r') as f:
                self._training_stages = yaml.load(f)

            # for stage in self.trials['stage'].unique():
            #     if stage not in self._training_stages.keys():
            #         self._training_stages.update({stage:{}})

        return self._training_stages

    @training_stages.setter
    def training_stages(self, value):
        self._training_stages = value

    @property
    def transitions(self):
        if self._transitions is None:
            with open(self.transitions_path,'r') as f:
                self._transitions = yaml.load(f)
        return self._transitions

    @transitions.setter
    def transitions(self, value):
        self._transitions = value

    def reload_trials(self):
        self._trials = annotate_trials(
            dro.load_from_folder(
                os.path.join(self.mouse_dir,'output'),
                load_existing_dataframe=False,
                save_dataframe=False,
                filename_contains=self.__id,
                )
            )
        assert len(self._trials['mouse_id'].unique())==1
        # self._session_summary = None

    @property
    def trials(self):
        if self._trials is None:
            self.reload_trials()
        return self._trials

    def save_params(self):
        timestamp = dt.datetime.now().strftime('%Y%m%d%H%M%S')
        parameter_file = os.path.join(self.mouse_dir,'adjustment','daily-{}.json'.format(timestamp))
        try:
            with open(parameter_file,'w') as f:
                json.dump(f,self.active_parameters)
        except IOError as e:
            print 'uh oh! IOError: {}'.format(e)

        with open(self.current_stage_path,'a') as f:
            f.write(self.state+'\n')
            self._current_stage = self.state

    @property
    def current_stage(self):
        if self._current_stage is None:
            try:
                with open(self.current_stage_path,'r') as f:
                    lines = f.readlines()
                    self._current_stage = lines[-1].strip() 
                # self._current_stage = self.trials.iloc[-1]['stage']
            except IOError:
                self._current_stage = self.transitions[0]['source']

            return self._current_stage
        else:
            return self._current_stage

    @property
    def machine(self):
        if self._machine is None:
            self.initialize_machine()
            return self._machine
        else:
            return self._machine


    def two_out_of_three_aint_bad(self):
        return two_out_of_three_aint_bad(self)


@click.command()
@click.argument('mouse_id')
@click.option('--dry-run/--no-dry-run', default=False)
# @click.option('--force',type=str)
def main(mouse_id,dry,force):

    mouse = Mouse(mouse_id)


    mouse.progress()

    print mouse.state
    if dry:
        print mouse.active_parameters
    else:
        mouse.write_params(state)



if __name__ == '__main__':
    
    main()

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

