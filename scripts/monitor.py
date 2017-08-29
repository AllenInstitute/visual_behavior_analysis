#!/usr/bin/env python
import subprocess
import itertools
import logging, traceback
import numpy as np
from collections import defaultdict
from slackclient import SlackClient
from behavior_subscriber import BehaviorSubscriber
from zro import ZroError

with open('/local1/SECRET.txt','rb') as f:
    PASSWD = f.readlines()[0].strip()

from jira import JIRA
jira = JIRA(server='http://jira.corp.alleninstitute.org',basic_auth=('justink', PASSWD))


slack_token = "xoxb-149706607461-B3uHZ3maGsyQQIJotdnqVx4T"
sc = SlackClient(slack_token)

## setup error logging
LOG_FILE = '/local1/slackitall.log'
logger = logging.getLogger('slackitall')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

fh = logging.FileHandler(LOG_FILE)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)

eh = logging.handlers.SMTPHandler(
    mailhost=('aicas-1.corp.alleninstitute.org', 25),
    fromaddr='justink@alleninstitute.org',
    toaddrs='justink@alleninstitute.org',
    subject='slackitall error',
    )
eh.setLevel(logging.ERROR)
eh.setFormatter(formatter)

for h in (fh,eh):
    logger.addHandler(h)

ACTIVE = defaultdict(lambda: 'Unknown mouse')

CLUSTER = defaultdict(lambda: '#cluster_x')
for c,n in itertools.product('ABCDEF',range(6)):
    CLUSTER['{}{}'.format(c,n+1)] = '#cluster_{}'.format(c.lower())


def mouse_started(mouse,rig=None):
    sc.api_call(
        "chat.postMessage",
        channel=CLUSTER[rig],
        text=":mouse: {} has started on {} :tv:".format(mouse,rig),
    )

def mouse_finished(mouse,rig,pkl):
    try:
        water = reward_volume(pkl)

    except IndexError:
        water = np.nan
    sc.api_call(
        "chat.postMessage",
        channel=CLUSTER[rig],
        text=":tada: {} finished on {} & received *{:0.3f}uL* water :droplet:".format(mouse,rig,water),
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

        elif (data['rig_name'] not in ACTIVE.keys()) and (data['rig_name']!='WF1'):
            try:
                header = sub.proxy.get_current_header(data['rig_name'])
            except ZroError as e:
                logger.error(e)
                header = {'mouse_id': 'unknown'}
                issue = jira.create_issue(
                    project='VB',
                    summary=str(e),
                    components=[{'name':'accumulator'}],
                    description=str(e),
                    issuetype={'name':'Bug'},
                )
                try:
                    jira.add_remote_link(
                        issue,
                        jira.issue('VB-92'),
                        relationship='duplicates',
                    )
                except:
                    pass

            ACTIVE[data['rig_name']] = header['mouse_id']
            logger.info('updated ACTIVE status: {}'.format(dict(ACTIVE)))

        if (data['index'] % 25) == 0:
            logger.info('rig {}, mouse {},index {}'.format(data['rig_name'],ACTIVE[data['rig_name']],data['index']))

    except KeyError:
        logger.error(data)

@sub.data_hook("*")
def new_session(data):
    try:
        if data['index']==-1:
            mouse_started(
                data['init_data']['mouse_id'],
                data['rig_name'],
            )
    except KeyError:
        logger.error(data)


@sub.data_hook("*")
def tadaaa(data):
    try:
        if data["index"] == -2:
            mouse = ACTIVE[data['rig_name']]
            logger.info('{} finished on {}'.format(mouse,data['rig_name']))
            mouse_finished(mouse,data['rig_name'],data['pkl'])
            ACTIVE.pop(data['rig_name'])
    except KeyError as e:
        # logger.warning(e)
        pass

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
        # logger.error(data)
        pass



WARNING_TIMES = [2,5,10,20]
DEFAULT_LAST_CHANGE = dict(
    last_change = 0,
    'warnings' = {t:False for t in WARNING_TIMES},
)
LAST_CHANGE = defaultdict(lambda: DEFAULT_LAST_CHANGE)

def no_changes_happening(mouse,rig,starttime,minutes_since_last_change):
    message = "{} is {} minutes into the session and it's been {} minutes since the last change, is everything OK?".format(
        mouse,
        starttime,
        minutes_since_last_change,
        )
    sc.api_call(
        "chat.postMessage",
        channel=CLUSTER[rig],
        text=":rotating_light: :warning: {} :warning: :rotating_light:".format(message),
    )

@sub.data_hook("*")
def check_for_last_change(data):
    rig = data['rig_name']
    try:
        if data['index']==-1:
            LAST_CHANGE[rig] = DEFAULT_LAST_CHANGE

        elif data["index"] == -2:
            LAST_CHANGE.pop(rig)

        elif np.isnan(data['change_time'])==False:

            last_change = data['change_time']
            LAST_CHANGE[] = DEFAULT_LAST_CHANGE
            LAST_CHANGE[rig]['last_change'] = data['change_time']
            ## change_detected, reset warning flags
        else:
            minutes_since_last_change = data['starttime']-last_change)/60.

            for minutes in [t in WARNING_TIMES if LAST_CHANGE[rig][t]==False]:
                if minutes_since_last_change > minutes:

                    no_changes_happening(
                        mouse,
                        rig,
                        data['starttime']/60.,
                        minutes_since_last_change,
                        )
                    LAST_CHANGE[rig][minutes] = True

    except KeyError:
        # logger.error(data)
        pass



if __name__ == "__main__":
    sub.run_forever()
