import six
import pandas as pd
from functools import wraps


def data_or_pkl(func):
    """ Decorator that allows a function to accept a pickled experiment object
    or a path to the object.

    >>> @data_or_pkl
    >>> def print_keys(data):
    >>>     print data.keys()

    """
    @wraps(func)
    def pkl_wrapper(first_arg, *args, **kwargs):
        if isinstance(first_arg, six.string_types):
            return func(pd.read_pickle(first_arg), *args, **kwargs)
        else:
            return func(first_arg, *args, **kwargs)

    return pkl_wrapper
