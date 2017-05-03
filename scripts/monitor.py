#!/usr/bin/env python
import subprocess
import itertools
import logging, traceback
from collections import defaultdict
from slackclient import SlackClient
from behavior_subscriber import BehaviorSubscriber
from zro import ZroError


slack_token = "xoxb-149706607461-B3uHZ3maGsyQQIJotdnqVx4T"
sc = SlackClient(slack_token)

## setup error logging
LOG_FILE = '/local1/slackitall.log'
logger = logging.getLogger('behavior_monitor')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

fh = logging.FileHandler(LOG_FILE)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)

eh = logging.handlers.SMTPHandler(
    mailhost=('aicas-1.corp.alleninstitute.org', 25), 
    fromaddr='justink@alleninstitute.org', 
    toaddrs='justink@alleninstitute.org', 
    subject='autotrain error', 
    )
eh.setLevel(logging.ERROR)
eh.setFormatter(formatter)

for h in (fh,eh):
    logger.addHandler(h)

ACTIVE = defaultdict(lambda: 'Unknown mouse')

CLUSTER = defaultdict(lambda: '#cluster_x')
for c,n in itertools.product('ABCDEF',range(6)):
    CLUSTER['{}{}'.format(c,n+1)] = '#cluster_{}'.format(c.lower())


def mouse_started(mouse,user,rig=None):
    sc.api_call(
        "chat.postMessage",
        channel=CLUSTER[rig],
        text=":tv: :mouse: {} has started {} on {}".format(user,mouse,rig),
    )

def mouse_finished(mouse,rig,pkl):
    try:
        water = reward_volume(pkl)
    
    except IndexError:
        water = np.nan
    sc.api_call(
        "chat.postMessage",
        channel=CLUSTER[rig],
        text=":tada: {} finished on {} & received *{:0.2f}uL* water :droplet:".format(mouse,rig,water),
    )

def mouse_not_licking(mouse='M999999',rig=None):
    sc.api_call(
        "chat.postMessage",
        channel=CLUSTER[rig],
        text=":rotating_light: :warning: {} hasn't licked at all this session! check for a hardware failure in {} :warning: :rotating_light:".format(mouse,rig),
    )
    
import numpy as np
import pandas as pd
def reward_volume(pkl):
    pkl = pkl.replace('//','/').replace('\\','/').replace('aibsdata','data').replace('behavior/data','Behavior/Data')
    data = pd.read_pickle(pkl)
    n, v = len(data['rewards'][:,0]), np.sum(data['rewards'][:,3])
    return v

from behavior_subscriber.config import SUBSCRIBER_CONFIG
SUBSCRIBER_CONFIG.update(rep_port=9001)

sub = BehaviorSubscriber(**SUBSCRIBER_CONFIG)

@sub.data_hook('*')
def update_active(data):
    try:
        if data['index']==-1:
            logging.debug('{} starting on {}'.format(data['init_data']['mouse_id'],data['rig_name']))
            ACTIVE[data['rig_name']] = data['init_data']['mouse_id']

        elif data['rig_name'] not in ACTIVE.keys():
            try:
                header = sub.proxy.get_current_header(data['rig_name'])
            except ZroError as e:
                logger.error(e)
                header = {'mouse_id': 'unknown'}

            ACTIVE[data['rig_name']] = header['mouse_id']
            logger.info('updated ACTIVE status: {}'.format(dict(ACTIVE)))

        if (data['index'] % 10) == 0:
            logger.debug('rig {}, mouse {},index {}'.format(data['rig_name'],ACTIVE[data['rig_name']],data['index']))

    except KeyError:
        logger.info(data)

@sub.data_hook("*")
def new_session(data):
    try:
        if data['index']==-1:
            mouse_started(
                data['init_data']['mouse_id'],
                data['params']['user_id'],
                data['rig_name'],
            )
    except KeyError:
        logger.info(data)


@sub.data_hook("*")
def tadaaa(data):
    if data["index"] == -2:
        mouse = ACTIVE[data['rig_name']]
        logger.info('{} finished on {}'.format(mouse,data['rig_name']))
        mouse_finished(mouse,data['rig_name'],data['pkl'])
        ACTIVE.pop(data['rig_name'])

LICKED = defaultdict(lambda: None)

@sub.data_hook("*")
def check_for_licks(data):
    LICK_BY = 15 # minutes. email will be sent if mouse hasn't licked by this time
    try:
        if data['index']==-1:
            LICKED[data['rig_name']] = False

        elif data["index"] == -2:
            LICKED.pop(data['rig_name'])

        elif LICKED[data['rig_name']]==False:
            LICKED[data['rig_name']] = len(data['lick_times'])>0
            logger.info('rig {}, licked {}, start time {}'.format(data['rig_name'],LICKED[data['rig_name']],data['starttime']))

            if data['starttime'] > (LICK_BY * 60):
                mouse = ACTIVE[data['rig_name']]
                mouse_not_licking(mouse,data['rig_name'])
                LICKED[data['rig_name']] = True
    except KeyError:
        logger.info(data)


if __name__ == "__main__":
    sub.run_forever()
