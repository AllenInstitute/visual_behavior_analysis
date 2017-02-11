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

DELAY = 0.2

class MockSession(Publisher):
    def __init__(self,pkl, rep_port=12000, pub_port=9998):
        super(MockSession, self).__init__(rep_port=rep_port,
                                       pub_port=pub_port)
        self.data = None
        
    def load(self):

        if self.data is None:
            with open(pkl,'rb') as f:
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
        for trial in self.data['triallog']:
            self.publish(trial)
            time.sleep(DELAY)
        self.close()

    def close(self):
        self.publish({'index': -2})
    
if __name__ == "__main__":

    import sys
    if len(sys.argv)>1:
        pkl = sys.argv[1]
    else:
        pkl = default_pkl

    loop = True

    mock = MockSession(pkl)
    mock.run_session()

    while loop:
        mock.run_session()
