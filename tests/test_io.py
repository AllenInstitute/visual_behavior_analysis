import pytest
import os
import pandas as pd

from visual_behavior import io


@pytest.mark.parametrize("value, to_pickle", [
    (pd.DataFrame(data={"foo": {0: 1, 1: 2, }, }), False, ),
    (pd.DataFrame(data={"foo": {0: 1, 1: 2, }, }), True, ),
])
def test_data_or_pkl(tmpdir, value, to_pickle):
    if to_pickle:
        pickle_path = os.path.join(str(tmpdir), "test.pkl")
        value.to_pickle(pickle_path)
        pd.testing.assert_frame_equal(
            io.data_or_pkl(lambda value: value)(pickle_path).sort_index(axis=1),
            value.sort_index(axis=1)
        )
    else:
        pd.testing.assert_frame_equal(
            io.data_or_pkl(lambda value: value)(value).sort_index(axis=1),
            value.sort_index(axis=1)
        )
