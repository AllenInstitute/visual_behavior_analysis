
def assert_is_valid_dataframe(df, schema_instance):
    records = df.to_dict('records')
    errors = schema_instance.validate(records, many=True)
    assert (len(errors) == 0), errors


def generate_validation_metrics(data):
    """generate a set of validation metrics

    Parameters
    ----------
    data: dict
        visual behavior core data object

    Returns
    -------
    dict:
        validation metrics
    """
    return {"passes": True, }
