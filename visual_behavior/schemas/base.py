import pandas as pd
from marshmallow import Schema, post_load


class PandasSchemaBase(Schema):  # keeping marshmallow convention for this i guess...

    def dump(self, obj, many=None, update_fields=False):
        if many is True:
            return super(PandasSchemaBase, self).dump(
                obj.to_dict("records"),
                many=many,
                update_fields=update_fields
            )
        else:
            return super(PandasSchemaBase, self).dump(
                obj.to_dict(),
                many=many,
                update_fields=update_fields
            )

    @post_load(pass_many=True)
    def _convert_to_pandas(self, data, many):
        return pd.DataFrame(data=data) if many is True else pd.Series(data=data)
