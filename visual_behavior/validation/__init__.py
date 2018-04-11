
def is_valid_dataframe(df, schema_instance):
    records = df.to_dict('records')
    errors = schema_instance.validate(records, many=True)
    return (len(errors) == 0)
