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

class MockSession(Publisher):
    def __init__(self,pkl, rep_port=12000, pub_port=12001):
        super(MockSession, self).__init__(rep_port=rep_port,
                                       pub_port=pub_port)
        
        with open(pkl,'rb') as f:
            data = pickle.load(f)

        header = {
            'index':-1,
            'init_data': {
                'task_id': data['task'],
                'mouse_id': data['mouseid'],
                }
            }
        self.publish(header)
        self.trials = data['triallog']

    def publish(self,data):
        print data
        super(MockSession, self).publish(data)
        
    def run_session(self):
        for trial in self.trials:
            self.publish(trial)
            time.sleep(1.0)
        self.publish({'index': -2})
    
if __name__ == "__main__":

    import sys
    if len(sys.argv)>1:
        pkl = sys.argv[1]
    else:
        pkl = default_pkl

    mock = MockSession(pkl)
    time.sleep(5)
    mock.run_session()
