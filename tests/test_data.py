import pytest
import pandas as pd

from visual_behavior import data
from visual_behavior.processing import get_trials


@pytest.fixture
def trials_fixture(pizza_data_fixture):
    return get_trials(pizza_data_fixture)


expected_df_0 = pd.DataFrame(data={
    "cumulative_rewards": [0, 1, 2, ],
    "cumulative_volume": [0.000, 0.007, 0.014, ],
    "index": pd.RangeIndex(start=0, stop=3, step=1),
    "events": [
        [
            ['initial_blank', 'enter', 7.513004185308317, 0],
            ['initial_blank', 'exit', 7.5131104567819404, 0],
            ['pre_change', 'enter', 7.513202823576774, 0],
            ['early_response', '', 7.935050252171282, 25],
            ['abort', '', 7.935161158537837, 25],
            ['timeout', 'enter', 7.935301198517099, 25],
            ['timeout', 'exit', 7.935412104883654, 25],
        ],
        [
            ['initial_blank', 'enter', 8.268973099743558, 46],
            ['initial_blank', 'exit', 8.269071425686445, 46],
            ['pre_change', 'enter', 8.269162137162374, 46],
            ['pre_change', 'exit', 10.536494153744696, 181],
            ['stimulus_window', 'enter', 10.536653395423242, 181],
            ['stimulus_changed', '', 11.338332948195802, 230],
            ['auto_reward', '', 11.3384018094622, 230],
            ['response_window', 'enter', 11.503911849633479, 238],
            ['response_window', 'exit', 12.1045244500534, 274],
            ['stimulus_window', 'exit', 16.541491561515294, 540],
            ['no_lick', 'exit', 16.5417219819067, 540]
        ],
        [
            ['initial_blank', 'enter', 16.72633605754683, 552],
            ['initial_blank', 'exit', 16.7264621928473, 552],
            ['pre_change', 'enter', 16.726556546024817, 552],
            ['pre_change', 'exit', 18.993592591587138, 686],
            ['stimulus_window', 'enter', 18.99372799667347, 686],
            ['stimulus_changed', '', 21.346745609928735, 828],
            ['auto_reward', '', 21.346815795450258, 828],
            ['response_window', 'enter', 21.495668361492516, 835],
            ['response_window', 'exit', 22.112903674344476, 872],
            ['stimulus_window', 'exit', 24.998558548298565, 1044],
            ['no_lick', 'exit', 24.99879227932778, 1044],
        ],
    ],
    "licks": [
        [(7.934872470921013, 25), ],
        [],
        [
            (21.646282848975986, 844),
            (21.796896343268116, 853),
            (22.0132746644172, 866),
            (22.129987227559337, 873),
            (22.22999994040852, 879),
            (22.363486181066726, 887),
            (22.46347903008906, 893),
        ],
    ],
    "rewards": [
        [],
        [(11.338415383077212, 230), ],
        [(21.346828706937707, 828), ],
    ],
    "stimulus_changes": [
        [],
        [("im065", "im063", "im063", 229, 11.338229987359984)],
        [("im063", "im065", "im065", 827, 21.346641655901575)],
    ],
    "success": [False, True, True, ],
    "trial_params": [
        {'catch': False, 'auto_reward': True, 'change_time': 0.7802031782835195},
        {'catch': False, 'auto_reward': True, 'change_time': 0.7802031782835195},
        {'catch': False, 'auto_reward': True, 'change_time': 1.9704493509311094},
    ]
})


def test_annotate_parameters(trials_fixture, pizza_data_fixture):
    test_key_dict = {"test_trial_count": "trial_count", }

    expected_df_1 = expected_df_0.copy(deep=True)

    expected_df_1["test_trial_count"] = [pizza_data_fixture["trial_count"]] * 3

    # raise ValueError(pizza_data_fixture.keys())
    pd.testing.assert_frame_equal(
        data.annotate_parameters(trials_fixture, pizza_data_fixture).iloc[:3],
        expected_df_0,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )

    pd.testing.assert_frame_equal(
        data.annotate_parameters(trials_fixture, pizza_data_fixture, test_key_dict).iloc[:3],
        expected_df_1,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )


# def test_explode_startdatetime(trials_fixture, pizza_data_fixture):
#     pass  # can't find start_time in pizza


def test_annotate_n_rewards(trials_fixture):
    expected_df_1 = expected_df_0.copy(deep=True)
    expected_df_1["number_of_rewards"] = [0, 1, 1, ]

    pd.testing.assert_frame_equal(
        data.annotate_n_rewards(trials_fixture).iloc[:3],
        expected_df_1,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )


# def test_annotate_rig_id(trials_fixture, pizza_data_fixture):
#     pass  # no rig id


def test_annotate_reward_volume(trials_fixture, pizza_data_fixture):
    expected_df_1 = expected_df_0.copy(deep=True)
    expected_df_1["reward_volume"] = [0, 0.007, 0.007, ]

    pd.testing.assert_frame_equal(
        data.annotate_reward_volume(trials_fixture, pizza_data_fixture).iloc[:3],
        expected_df_1,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )


def test_annotate_change_detect(trials_fixture):
    expected_df_1 = expected_df_0.copy(deep=True)
    expected_df_1["change"] = [True, True, True, ]
    expected_df_1["detect"] = [False, False, False, ]

    pd.testing.assert_frame_equal(
        data.annotate_change_detect(trials_fixture).iloc[:3],
        expected_df_1,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True
    )
