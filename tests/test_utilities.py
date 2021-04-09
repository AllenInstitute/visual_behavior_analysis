from visual_behavior.utilities import local_time
from visual_behavior.utilities import find_nearest_index
from visual_behavior.utilities import dprime
from visual_behavior.utilities import trial_number_limit
from visual_behavior.utilities import Movie
import numpy as np
import pytest
import os


def test_local_time():
    EXPECTED = "2018-05-23T03:55:42.118000-07:00"

    coerced = local_time("2018-05-23T03:55:42.118000", timezone='America/Los_Angeles')
    assert coerced == EXPECTED


def test_find_nearest_index():
    time_array = np.array([0., 0.03333333, 0.06666667, 0.1, 0.13333333, 0.16666667])
    assert find_nearest_index(0.14, time_array) == 4
    assert (find_nearest_index([0.11, 0.14], time_array) == np.array([3, 4])).all()


@pytest.mark.skipif(not os.path.exists('//allen/programs/braintv'), reason="no access to network path, skipping test on network PKL files")
def test_movie_load():
    movie_path = "//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/test_fixtures/853773937_video-0.avi"
    movie = Movie(movie_path)

    # get frame by timestamp
    frame1 = movie.get_frame(time=10.1, timestamps='file')
    assert frame1[100, 100, 0] == 143

    # get frame by frame number
    frame2 = movie.get_frame(frame=1000)
    assert frame2[100, 100, 0] == 150


def test_trial_number_limit():
    assert trial_number_limit(1, 5) == 0.9

    assert trial_number_limit(1, 50) == 0.99

    assert trial_number_limit(1, 100) == 0.995

    assert trial_number_limit(0.5, 100) == 0.5

    assert trial_number_limit(0, 100) == 0.005


def test_dprime():

    d_prime = dprime(1.0, 0.0)
    assert d_prime == 4.6526957480816815

    d_prime = dprime(1.0, 0.0, limits=False)
    assert d_prime == 4.6526957480816815

    d_prime = dprime(1.0, 0.0, limits=(0.01, 0.99))
    assert d_prime == 4.6526957480816815

    d_prime = dprime(1.0, 0.0, limits=(0.0, 1.0))
    assert d_prime == np.inf

    d_prime = dprime(
        go_trials=[1, 1, 1, 1, 1, 1, 1],
        catch_trials=[0, 0],
        limits=True
    )
    assert d_prime == 2.1397235428816046

    d_prime = dprime(
        go_trials=[1, 1, 1, 1, 1, 1, 1],
        catch_trials=[0, 0],
        limits=False
    )
    assert d_prime == 4.6526957480816815

    d_prime = dprime(
        go_trials=[0, 1, 0, 1],
        catch_trials=[0, 1],
        limits=False
    )
    assert d_prime == 0.0
