import re
import numpy as np
import pandas as pd
import deepdish as dd
from warnings import warn

try:
    from collections import Mapping
except ImportError:
    from collections.abc import Mapping

from six.moves import cPickle as pickle
from six import iteritems, string_types, binary_type


DEEPDISH_ATTR_TYPES = (
    int, float, bool, string_types, binary_type, np.int8, np.int16,
    np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float16,
    np.float32, np.float64, np.bool_, np.complex64, np.complex128, np.ndarray,
    pd.DataFrame, pd.Series, pd.Panel,
)

REFERENCE_KEYWORD = "$ref:#"
REFERENCE_PATTERN = r"^\$ref:\#(/.*)$"
OBJECT_GROUP_NAME = "data"
PICKLE_GROUP_NAME = "pickled"
FAILURE_GROUP_NAME = "failed"
KEY_MAP_GROUP_NAME = "keymap"
INT_ID_FORMAT = "i{id_}"  # storing an int attr or a string int attr can cause problems, name sucks tho...


def is_pickleable(obj):
    """Whether or not a python object is pickleable

    Parameters
    ----------
    obj : object
        python object to check

    Returns
    -------
    boolean
        whether or not it's pickleable
    Exception or None
        exception raised when trying to pickle
    """
    try:
        pickle.dumps(obj)
        return True, None
    except Exception as e:
        return False, e


def we_can_pizza_that(obj):
    """Recursively iterates through `obj` to return a deepdish dictionary that
    will be stored as a deepdish hdf5 in a shape that we expect

    Parameters
    ----------
    obj : python object

    Notes
    -----
    .. notes::
    creates shallow copies lists and dictionaries

    - the idea is based on derric's wecanpicklethat

    Returns
    -------
    dict
        data : python object(s) that deepdish can handle without pickling
        pickled: python object(s) that deepdish will pickle
    """
    pickles = {}
    failures = {}
    key_map = generate_key_map(obj)

    return {
        OBJECT_GROUP_NAME: __we_can_pizza_that(obj, pickles, failures),
        KEY_MAP_GROUP_NAME: key_map,
        PICKLE_GROUP_NAME: pickles,
        FAILURE_GROUP_NAME: failures,
    }


def __we_can_pizza_that(obj, pickles={}, failures={}):
    """Maybe don't go recursive? memory usage could be insane...

    Notes
    -----
    - recursive function intended to be called with
    `mouse_info.utils.we_can_pizza_that`
    - always pickles exceptions, regardless of whether or not they can be
    deepdished properly, there just doesn't seem like a reliable way to determine
    whether or not an exception is built-in
    """
    if isinstance(obj, DEEPDISH_ATTR_TYPES):
        return obj
    elif isinstance(obj, list):
        return [
            __we_can_pizza_that(sub_obj, pickles, failures)
            for sub_obj in obj
        ]
    elif isinstance(obj, tuple):
        return tuple([
            __we_can_pizza_that(sub_obj, pickles, failures)
            for sub_obj in obj
        ])
    elif isinstance(obj, dict):
        return {
            __we_can_pizza_that(k, pickles, failures): __we_can_pizza_that(v, pickles, failures)
            for k, v in iteritems(obj)
        }
    elif obj is None:  # different logic for singleton-esque identity
        return obj
    else:
        reference_format = "{keyword}/{group_name}/{id_}"

        is_pickleable_, exception = is_pickleable(obj)

        if is_pickleable_:
            id_ = INT_ID_FORMAT.format(id_=len(pickles))
            pickles[id_] = obj
            return reference_format.format(
                keyword=REFERENCE_KEYWORD,
                group_name=PICKLE_GROUP_NAME,
                id_=id_
            )
        else:
            id_ = INT_ID_FORMAT.format(id_=len(failures))
            failures[id_] = (str(obj), str(exception), )
            return reference_format.format(
                keyword=REFERENCE_KEYWORD,
                group_name=FAILURE_GROUP_NAME,
                id_=id_
            )


def we_can_unpizza_that(filepath, root_group="/data"):
    """Loads a pizza-formatted hdf5 file. Will attempt to resolve references to
    pickled objects

    Parameters
    ----------
    filepath : string
        path to the pizza file
    root_group : string
        group name to unpack

    Returns
    -------
    obj
        unpizza-ed object

    Notes
    -----
    - if a reference to a pickle fails to unpickle, hopefully a warning will be
    raised, and the reference will be left in the data in the form:
        `$ref:#<group name>`
    """
    return __resolve_references(
        dd.io.load(filepath, group=root_group, unpack=False),
        filepath
    )


def __resolve_references(obj, filepath):
    """Resolves references to pickled group
    """
    if isinstance(obj, string_types):
        return resolve_reference(obj, filepath)
    elif isinstance(obj, list):
        return [__resolve_references(obj_, filepath) for obj_ in obj]
    elif isinstance(obj, tuple):
        return tuple([__resolve_references(obj_, filepath) for obj_ in obj])
    elif isinstance(obj, dict):
        return {k: __resolve_references(v, filepath) for k, v in iteritems(obj)}
    else:
        return obj


def resolve_reference(string, filepath):
    """Easier to test
    """
    try:
        return dd.io.load(
            filepath,
            group=re.match(REFERENCE_PATTERN, string).group(1),
            unpack=False
        )
    except (TypeError, IndexError):
        return string
    except Exception as e:
        warn(UserWarning(e))
        return string


def generate_key_map(obj):
    """Generates map of roots to keys

    Parameters
    ----------
    obj : python object

    Returns
    -------
    dictionary
        key_map relating attr names to children names

    Notes
    -----
    - supports `dict` and `collections.abc.Mapping` subclasses for generating
    maps, all other types will return None
    - keys are tuples because we can only have alphanum
    """
    key_map = {}
    __generate_key_map(obj, key_map, ["", "data"])  # ["", "data"] because "/".join will yield "/data"

    return key_map


def __generate_key_map(obj, key_map, root):
    """It's recursive, don't call this directly for obvious reasons...

    Notes
    -----
    - we store roots as tuples because we can only use alphanum
    - root is a list because it allows for easier manipulation inline
    - sorts keys to create predictable key ordering across python 2 and 3,
    converts each value to a string for comparison
    """
    if isinstance(obj, dict) or isinstance(obj, Mapping):
        keys = []
        for key, value in iteritems(obj):
            __generate_key_map(value, key_map, root + [key])
            keys.append(key)
        value = tuple(sorted(keys, key=str))  # ordering isn't guaranteed so a set makes more sense here..., convert to strings to compare against all types
    else:
        value = None

    key_map[tuple(root)] = value


def get_keys(filepath, root="/data"):
    """Gets keys from pizza formatted hdf5

    Parameters
    ----------
    filepath : string
        path to the pizza formatted hdf5
    root : string, default="/data"
        attr to get keys for children names

    Returns
    -------
    tuple or None
        children names or None if doesn't have children
    """
    return dd.io.load(
        filepath,
        group="/" + KEY_MAP_GROUP_NAME
    )[tuple(root.split("/"))]
