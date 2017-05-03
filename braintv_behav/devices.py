rig_dict = {
    'W7DTMJ19R2F':'A1',
    'W7DTMH35Y0T':'A2',
    'W7DTMJ03J70R':'Dome',
    'W7VS-SYSLOGIC2':'A3',
    'W7VS-SYSLOGIC3':'A4',
    'W7VS-SYSLOGIC4':'A5',
    'W7VS-SYSLOGIC5':'A6',
    'W7VS-SYSLOGIC7':'B1',
    'W7VS-SYSLOGIC8':'B2',
    'W7VS-SYSLOGIC9':'B3',
    'W7VS-SYSLOGIC10':'B4',
    'W7VS-SYSLOGIC11':'B5',
    'W7VS-SYSLOGIC12':'B6',
    'W7VS-SYSLOGIC13':'C1',
    'W7VS-SYSLOGIC14':'C2',
    'W7VS-SYSLOGIC15':'C3',
    'W7VS-SYSLOGIC16':'C4',
    'W7VS-SYSLOGIC17':'C5',
    'W7VS-SYSLOGIC18':'C6',
    'W7VS-SYSLOGIC19':'D1',
    'W7VS-SYSLOGIC20':'D2',
    'W7VS-SYSLOGIC21':'D3',
    'W7VS-SYSLOGIC22':'D4',
    'W7VS-SYSLOGIC23':'D5',
    'W7VS-SYSLOGIC24':'D6',
    'W7VS-SYSLOGIC26':'Widefield-329',
    'OSXLTTF6T6.local':'DougLaptop',
    'W7DTMJ026LUL':'DougPC',
    }


# -> devices.py
def get_rig_id(in_val,input_type='computer_name'):
    '''
    This provides a map between the computer name and the rig ID
    Will need updated if computers are swapped out
    '''

    computer_dict = dict((v,k) for k,v in rig_dict.iteritems())
    if input_type == 'computer_name' and in_val in rig_dict.keys():
        return rig_dict[in_val]
    elif input_type == 'rig_id' and in_val in computer_dict.keys():
        return computer_dict[in_val]
    else:
        return 'unknown'
