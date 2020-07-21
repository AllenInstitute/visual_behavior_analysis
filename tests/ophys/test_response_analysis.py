import pytest
from visual_behavior.ophys.response_analysis import response_processing as rp
import numpy as np

@pytest.fixture
def cell_impulses():
    cell1_impulse = np.array([0, 1, 2, 3, 2, 1, 0])
    cell2_impulse = np.array([5, 4, 3, 2, 1, 0, 0])
    cell3_impulse = np.array([0, 0, 1, 2, 3, 4, 5])
    data = np.vstack([cell1_impulse, cell2_impulse, cell3_impulse])
    return data

@pytest.fixture
def cell_impulse_events():
    cell1_impulse = np.array([0, 0, 0, 1, 0, 0, 0])
    cell2_impulse = np.array([2, 0, 0, 0, 0, 0, 0])
    cell3_impulse = np.array([0, 0, 1, 0, 0, 1, 0])
    data = np.vstack([cell1_impulse, cell2_impulse, cell3_impulse])
    return data

@pytest.fixture
def event_inds():
    event_inds = np.linspace(50, 900, 25).astype(int)
    return event_inds

@pytest.fixture
def dff_trace_array(cell_impulses, event_inds):
    time = np.zeros(1000)
    time[event_inds] = 1
    cell_data = []
    for ind_cell in range(cell_impulses.shape[0]):
        cell_impulse = cell_impulses[ind_cell, :]
        data_this_cell = np.convolve(time, cell_impulse)[:1000]
        cell_data.append(data_this_cell)
    data = np.vstack(cell_data)
    return data

@pytest.fixture
def ca_events_array(cell_impulse_events, event_inds):
    time = np.zeros(1000)
    time[event_inds] = 1
    cell_data = []
    for ind_cell in range(cell_impulse_events.shape[0]):
        cell_impulse = cell_impulse_events[ind_cell, :]
        data_this_cell = np.convolve(time, cell_impulse)[:1000]
        cell_data.append(data_this_cell)
    data = np.vstack(cell_data)
    return data

def test_eventlocked_traces(dff_trace_array, event_inds, cell_impulses):
    start_ind_offset = 0
    end_ind_offset = 7
    sliced_dataout = rp.eventlocked_traces(dff_trace_array, event_inds, start_ind_offset, end_ind_offset)
    np.testing.assert_array_equal(sliced_dataout.mean(axis=1).T, cell_impulses)

def test_ca_events_smoothing(cell_impulse_events):
    smoothed_impulse_events = rp.filter_events_array(cell_impulse_events)

    # Make sure maxima are where we expect them to be
    assert np.argmax(smoothed_impulse_events[0, :]) == 3
    assert np.argmax(smoothed_impulse_events[1, :]) == 0
    assert np.argmax(smoothed_impulse_events[2, :]) == 5

    # Assert causality
    assert smoothed_impulse_events[0, 2] == 0

    # Make sure we are convolving the right direction
    assert smoothed_impulse_events[2, 0] == 0


