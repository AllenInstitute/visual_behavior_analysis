### PRECONDITIONS AND POSTCONDITIONS ###                                  # noqa: E266

# A precondition is something that must be true at the start of a function in order
# for it to work correctly.

# A postcondition is something that the function guarantees is true when it finishes.

## IMPORTS ##                                                             # noqa: E266

from visual_behavior.data_access import from_lims, from_lims_utilities


def validate_value_in_dict_keys(input_value, dictionary, dict_name):
    assert input_value in dictionary, "Error: input value is not in {} keys.".format(dict_name)


### NUMERIC THRESHOLDS ###                                                # noqa: E266

def validate_non_negative_input(input_value, variable_name):
    assert input_value >= 0, "Error: {} must be non negative".format(variable_name)


def validate_greater_than_zero(input_value, variable_name):
    assert input_value > 0, "Error: {} must be greater than zero".format(variable_name)


def validate_above_threshold(input_value, threshold_value, variable_name):
    assert input_value > threshold_value, "Error: {} must be greater than {}.".format(variable_name,
                                                                                      threshold_value)


def validate_greater_or_equal_threshold(input_value, threshold_value, variable_name):
    assert input_value >= threshold_value, "Error: {} must be greater or equal to {}.".format(variable_name,
                                                                                              threshold_value)


def validate_below_threshold(input_value, threshold_value, variable_name):
    assert input_value < threshold_value, "Error: {} must be less than {}.".format(variable_name,
                                                                                   threshold_value)


def validate_below_or_equal_threshold(input_value, threshold_value, variable_name):
    assert input_value <= threshold_value, "Error: {} must be less than or equal to {}.".format(variable_name,
                                                                                                threshold_value)


def validate_equals_threshold(input_value, threshold_value, variable_name):
    assert input_value == threshold_value, "Error: {} must equal {}.".format(variable_name,
                                                                             threshold_value)


### TYPES ###                                                             # noqa: E266
def validate_microscope_type(ophys_session_id, correct_microscope_type):
    session_microscope_type = from_lims_utilities.get_microscope_type(ophys_session_id)
    assert session_microscope_type == correct_microscope_type, "Error: incorrect microscope type.\
                                                               {} provided but {} necessary.".format(session_microscope_type,
                                                                                                     correct_microscope_type)


def validate_id_type(input_id, correct_id_type):
    """takes an input id, looks up what type of id it is and then validates
       whether it is the same as the desired/correct id type

    Parameters
    ----------
    input_id : int
         the numeric any of the common types of ids associated or used with an optical physiology.
         Examples: ophys_experiment_id, ophys_session_id, cell_roi_id etc. See ID_TYPES_DICT in
         from_lims module for complete list of acceptable id types 
    correct_id_type : string
        [description]
    """
    validate_value_in_dict_keys(correct_id_type, from_lims.ID_TYPES_DICT, "ID_TYPES_DICT")
    input_id_type = from_lims.get_id_type(input_id)
    assert input_id_type == correct_id_type, "Incorrect id type. Entered Id type is {},\
                                              correct id type is {}".format(input_id_type,
                                                                            correct_id_type)


### DATAFRAMES ###                                                        # noqa: E266

# def validate_column_not_null_or_empty(dataframe, column_name):

# def validate_column_datatype(dataframe, column_name):
