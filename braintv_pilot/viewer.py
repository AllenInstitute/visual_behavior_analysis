# -*- coding: utf-8 -*-
"""
Created on Wed Jun 08 15:35:33 2016

@author: derricw
"""
from zro import Subscriber

class MyViewer(Subscriber):
    def __init__(self, rep_port=13000):
        super(MyViewer, self).__init__(rep_port)
        
        self.add_subscription("localhost", 12001)
        
    def handle_data(self, from_str, data):
        print from_str
        print data
        
if __name__ == "__main__":
    
    v = MyViewer()
    v.run_forever()