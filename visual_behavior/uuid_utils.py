import uuid

NAMESPACE_VISUAL_BEHAVIOR = uuid.UUID('a4b1bc02-4490-4a61-82db-f5c274a77080')


def create_mouse_namespace(mouse_id):
    return uuid.uuid5(NAMESPACE_VISUAL_BEHAVIOR, mouse_id)


def create_session_uuid(mouse_id, session_datetime_iso_utc):
    mouse_namespace = create_mouse_namespace(mouse_id)
    return str(uuid.uuid5(mouse_namespace, session_datetime_iso_utc))
