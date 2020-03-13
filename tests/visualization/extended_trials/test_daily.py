import os
from visual_behavior.visualization.extended_trials import daily


def test_make_daily_figure(
    tmpdir, 
    exemplar_extended_trials_fixture,
):
    _dir = str(tmpdir.mkdir('test-make-daily-figure'))  # py.path to str
    daily_figure = daily.make_daily_figure(
        exemplar_extended_trials_fixture
    )

    daily_figure.savefig(
        os.path.join(_dir, 'test-daily-figure.png', )
    )