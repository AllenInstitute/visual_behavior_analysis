#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 08 15:09:08 2016

@author: derricw
"""
import numpy as np
import pickle
from zro import Publisher
import time


default_pkl = '/data/neuralcoding/Behavior/Data/M258173/output/170105150329-task=DoC_MNIST_stage=0v1_probes_n=3_mouse=M258173.pkl'


fast_forward = 20

class MockSession(Publisher):
    def __init__(self,pkl, rep_port=12000, pub_port=9998):
        super(MockSession, self).__init__(rep_port=rep_port,
                                       pub_port=pub_port)
        self.pkl = pkl
        self.data = None
        
    def load(self):

        if self.data is None:
            with open(self.pkl,'rb') as f:
                self.data = pickle.load(f)

        header = {
            'index':-1,
            'init_data': {
                # 'task_id': data['task'],
                'task_id': 'DetectionOfChange_Test',
                'mouse_id': 'MICKEY',
                },
            'params': self.data['params']
            }
        self.publish(header)

    def publish(self,data):
        print data
        super(MockSession, self).publish(data)
        
    def run_session(self):
        self.load()

        t0 = time.time()
        while len(self.data['triallog'])>0:
            trial = self.data['triallog'].pop(0)
            try:
                next_trial = self.data['triallog'][0]
                publish_time = next_trial['starttime'] / fast_forward
                print publish_time
                while (time.time()-t0) < publish_time:
                    time.sleep(0.1)
            except IndexError:
                pass
            finally:
                self.publish(trial)
            # time.sleep(DELAY)
        self.close()

    def close(self):
        self.publish({'index': -2,'pkl': self.pkl})
        self.data = None
    
if __name__ == "__main__":

    import sys
    if len(sys.argv)>1:
        pkl = sys.argv[1]
    else:
        pkl = default_pkl

    loop = True

    mock = MockSession(pkl)

    while loop:
        # mock = MockSession(pkl)
        mock.run_session()
