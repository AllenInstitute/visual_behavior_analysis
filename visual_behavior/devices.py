from six import iteritems
import datetime

# These mappings between computer name and rig name broke when computers
# were updated by MPE, and are only valid before this date.
VALID_BEFORE_DATE = datetime.datetime(2019, 5, 1)

RIG_NAME = {
    'W7DTMJ19R2F': 'A1',
    'W7DTMJ35Y0T': 'A2',
    'W7DTMJ03J70R': 'Dome',
    'W7VS-SYSLOGIC2': 'A3',
    'W7VS-SYSLOGIC3': 'A4',
    'W7VS-SYSLOGIC4': 'A5',
    'W7VS-SYSLOGIC5': 'A6',
    'W7VS-SYSLOGIC7': 'B1',
    'W7VS-SYSLOGIC8': 'B2',
    'W7VS-SYSLOGIC9': 'B3',
    'W7VS-SYSLOGIC10': 'B4',
    'W7VS-SYSLOGIC11': 'B5',
    'W7VS-SYSLOGIC12': 'B6',
    'W7VS-SYSLOGIC13': 'C1',
    'W7VS-SYSLOGIC14': 'C2',
    'W7VS-SYSLOGIC15': 'C3',
    'W7VS-SYSLOGIC16': 'C4',
    'W7VS-SYSLOGIC17': 'C5',
    'W7VS-SYSLOGIC18': 'C6',
    'W7VS-SYSLOGIC19': 'D1',
    'W7VS-SYSLOGIC20': 'D2',
    'W7VS-SYSLOGIC21': 'D3',
    'W7VS-SYSLOGIC22': 'D4',
    'W7VS-SYSLOGIC23': 'D5',
    'W7VS-SYSLOGIC24': 'D6',
    'W7VS-SYSLOGIC31': 'E1',
    'W7VS-SYSLOGIC32': 'E2',
    'W7VS-SYSLOGIC33': 'E3',
    'W7VS-SYSLOGIC34': 'E4',
    'W7VS-SYSLOGIC35': 'E5',
    'W7VS-SYSLOGIC36': 'E6',
    'W7DT102905': 'F1',
    'W10DT102905': 'F1',
    'W7DT102904': 'F2',
    'W7DT102903': 'F3',
    'W7DT102914': 'F4',
    'W7DT102913': 'F5',
    'W7DT12497': 'F6',
    'W7DT102906': 'G1',
    'W7DT102907': 'G2',
    'W7DT102908': 'G3',
    'W7DT102909': 'G4',
    'W7DT102910': 'G5',
    'W7DT102911': 'G6',
    'W7VS-SYSLOGIC26': 'Widefield-329',
    'OSXLTTF6T6.local': 'DougLaptop',
    'W7DTMJ026LUL': 'DougPC',
    'W7DTMJ036PSL': 'Marina2P_Sutter',
    'W7DT2PNC1STIM': '2P6',
    'W7DTMJ234MG': 'peterl_2p',
    'W7DT2P3STiM': '2P3',
    'W7DT2P4STIM': '2P4',
    'W7DT2P5STIM': '2P5',
    'W10DTSM112721': 'NP0',
    'W10DTSM118294': 'NP1',
    'W10DTSM118295': 'NP2',
    'W10DTSM118296': 'NP3',
    'W10DTSM118293': 'NP4',
    'meso1stim': 'MS1',
    'localhost': 'localhost'
}

RIG_NAME = {k.lower(): v for k, v in iteritems(RIG_NAME)}

COMPUTER_NAME = dict((v, k) for k, v in iteritems(RIG_NAME))


# -> devices.py
def get_rig_id(computer_name):
    '''
    This provides a map between the computer name and the rig ID.

    For data files generated before the release of camstim 5.1 , the rig ID
    is not included and instead must be determined by using the computer name.
    For data produced after, the rig ID is included in the pkl file and should
    not be determined by using this function. On 2019-05-01 the names of the
    computers began to change, and so any attempt to use this function to determine
    rig names from computer names for data generated after this date will fail.

    Parameters
    ----------
    computer_name : str
        The name of the computer that collected the behavior data

    >>> get_rig_id('W7DTMJ19R2F')
    A1

    '''

    return RIG_NAME.get(computer_name.lower(), 'unknown')


def get_computer_name(rig_id):
    '''
    This provides a map between the computer name and the rig ID.

    >>> get_computer_name('A1')
    W7DTMJ19R2F

    Parameters
    ----------
    rig_id : str
        rig name
    '''

    return COMPUTER_NAME.get(rig_id, 'unknown')
