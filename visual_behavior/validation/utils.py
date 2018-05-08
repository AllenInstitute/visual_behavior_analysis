
def assert_is_valid_dataframe(df, schema_instance):
    records = df.to_dict('records')
    errors = schema_instance.validate(records, many=True)
    assert (len(errors) == 0), errors
