from six import iteritems

try:
    from collections import Mapping
except ImportError:
    from collections.abc import Mapping

from .io import we_can_unpizza_that, get_keys


CACHE_THRESHOLD = float("inf")
UNDEFINED_KEYS = -1


class Pizza(Mapping):

    __DEFAULT_ROOT_GROUP = "/data"
    __DEFAULT_USE_CACHE = True

    def __init__(
            self,
            filepath,
            root_group=__DEFAULT_ROOT_GROUP,
            use_cache=__DEFAULT_USE_CACHE
    ):
        self.__filepath = filepath
        self.__root_group = root_group
        self.__cache = {}
        self.__use_cache = use_cache
        self.__keys = UNDEFINED_KEYS

    def __new__(
            cls,
            filepath,
            root_group=__DEFAULT_ROOT_GROUP,
            use_cache=__DEFAULT_USE_CACHE
    ):
        if get_keys(filepath, root_group) is not None:
            return super(Pizza, cls).__new__(cls)  # this is the class instance we will supply arguments to
        else:
            return we_can_unpizza_that(filepath, root_group)

    def __getitem__(self, key):
        if key not in self.__cache:
            item = Pizza(self.__filepath, self.__root_group + "/" + key)

            # do size estimate here...
            self.__cache[key] = item

        return self.__cache[key]

    def __contains__(self, key):
        return key in self.keys()

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        for key in self.keys():
            yield key

    def keys(self):
        if self.__keys == UNDEFINED_KEYS:
            self.__keys = get_keys(self.__filepath, self.__root_group)

        return self.__keys

    def clear(self):
        """Clears the cache
        """
        self.__cache.clear()
        self.__keys = UNDEFINED_KEYS

    def as_dict(self):
        return {
            key: (value.as_dict() if isinstance(value, Pizza) else value)
            for key, value in iteritems(self)
        }
