import uuid
from dateutil import tz, parser

NAMESPACE_VISUAL_BEHAVIOR = uuid.UUID('a4b1bc02-4490-4a61-82db-f5c274a77080')


def create_mouse_namespace(mouse_id):
    return uuid.uuid5(NAMESPACE_VISUAL_BEHAVIOR, mouse_id)


def create_session_uuid(mouse_id, session_datetime_iso_utc):
    mouse_namespace = create_mouse_namespace(mouse_id)
    return uuid.uuid5(mouse_namespace, session_datetime_iso_utc)


def make_deterministic_session_uuid(mouse_id, startdatetime):
    start_time_datetime = parser.parse(startdatetime)
    start_time_datetime_utc = start_time_datetime.astimezone(tz.gettz("UTC")).isoformat()
    behavior_session_uuid = create_session_uuid(str(mouse_id), start_time_datetime_utc)
    return behavior_session_uuid
