import uuid
from visual_behavior.uuid_utils import create_mouse_namespace, create_session_uuid, \
    make_deterministic_session_uuid

def test_create_mouse_namespace():

    EXPECTED_NAMESPACE = uuid.UUID('b37f73ec-0c79-502c-aa12-0363a7318008')

    mouse_id = 'M999999'
    mouse_namespace = create_mouse_namespace(mouse_id)
    assert mouse_namespace == EXPECTED_NAMESPACE

def test_create_session_uuid():

    EXPECTED_UUID = 'd26d7b5d-a1c7-50ea-92f9-5a82562d8d2f'

    mouse_id = 'M999999'
    session_datetime_iso_utc = '2018-05-22T22:02:42+00:00'

    session_uuid = create_session_uuid(mouse_id, session_datetime_iso_utc)

    assert str(session_uuid) == EXPECTED_UUID

def test_make_deterministic_session_uuid():
    kwargs = {
        'mouse_id': 'M999999',
        'startdatetime': '2018-05-22T20:55:42.118000-07:00',
    }
    EXPECTED_UUID = '1b48889e-465d-5732-a32e-cc2f67d2f581'
    behavior_session_uuid = make_deterministic_session_uuid(**kwargs)

    assert str(behavior_session_uuid) == EXPECTED_UUID

    kwargs = {
        'mouse_id': 'M999999',
        'startdatetime': '2018-05-23T03:55:42.118000+00:00',
    }
    EXPECTED_UUID = '1b48889e-465d-5732-a32e-cc2f67d2f581'
    behavior_session_uuid = make_deterministic_session_uuid(**kwargs)

    assert str(behavior_session_uuid) == EXPECTED_UUID
