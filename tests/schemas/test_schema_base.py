import pytest
import json
import numpy as np
import pandas as pd
from marshmallow import fields, ValidationError

from visual_behavior.schemas import base


class _TestSchema0(base.PandasSchemaBase):

    a = fields.String()
    b = fields.Float()


class _TestSchema1(base.PandasSchemaBase):

    a = fields.String()
    b = fields.Float()
    c = fields.Raw()


@pytest.mark.parametrize("input, many, schema", [
    (
        pd.DataFrame(data={
            "a": ["0", "1", "2", "3", ],
            "b": [0.55, 0.66, 0.77, 0.99, ],
        }),
        True,
        _TestSchema0(),
    ),
    (
        pd.DataFrame(data={
            "a": ["0", "1", "2", "3", ],
            "b": [0.55, np.nan, np.inf, -np.inf, ],
            "c": [np.nan, [1, 2, 3.0, np.nan, ], [], "", ],
        }),
        True,
        _TestSchema1(),
    ),
    (
        pd.Series(data={"a": "0", "b": 0.55, }),
        False,
        _TestSchema0(),
    ),
    (
        pd.Series(data={"a": "0", "b": np.nan, "c": [], }),
        False,
        _TestSchema1(),
    ),
])
def test_dumps_loads_PandasSchemaBase(input, many, schema):
    if isinstance(input, pd.Series):
        pd.testing.assert_series_equal(
            schema.loads(schema.dumps(input, many=many).data, many=many).data,
            input,
            check_index_type=False,
            check_dtype=False
        )
    else:
        pd.testing.assert_frame_equal(
            schema.loads(schema.dumps(input, many=many).data, many=many).data,
            input,
            check_column_type=False,
            check_index_type=False,
            check_dtype=False,
            check_like=True
        )


@pytest.mark.parametrize("input, schema, expected", [
    ({"a": "0", "b": 0.55, }, _TestSchema0(), {}, ),
    ({"a": 0, "b": 0.55, }, _TestSchema0(), {"a": ["Not a valid string."], }, ),
])
def test_dump_validate_PandasSchemaBase(input, schema, expected):
    assert schema.load(input).errors == expected


@pytest.mark.parametrize("input, many, schema", [
    (
        pd.DataFrame(data={
            "a": ["0", "1", "2", "3", ],
            "b": [0.55, 0.66, 0.77, 0.99, ],
        }),
        True,
        _TestSchema0(),
    ),
    (
        pd.DataFrame(data={
            "a": ["0", "1", "2", "3", ],
            "b": [0.55, np.nan, np.inf, -np.inf, ],
            "c": [np.nan, [1, 2, 3.0, np.nan, ], [], "", ],
        }),
        True,
        _TestSchema1(),
    ),
    (
        pd.Series(data={"a": "0", "b": 0.55, }),
        False,
        _TestSchema0(),
    ),
    (
        pd.Series(data={"a": "0", "b": np.nan, "c": [], }),
        False,
        _TestSchema1(),
    ),
])
def test_json_conformant(input, many, schema):
    """test the serialization of all the weird shit im worried about...
    """
    json.loads(schema.dumps(input, many=many).data)  # this will hopefully fail if we arent json conformant
