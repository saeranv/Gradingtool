from __future__ import print_function
from loadenv import *

import pickle

"""
Cache for astro
Static class for now

Note: May need to do this:

 git config --global core.autocrlf false

 to prevent git from adding lines to pkl

"""

CACHE_DIR = os.path.join(CURR_DIR, "cache")


def pickledata(data, cpickle_path, cpickle_name=None):
    """ Non-binary pickle """
    if cpickle_name is None:
        pkl_file_path = os.path.join(cpickle_path)
    else:
        pkl_file_path = os.path.join(cpickle_path, cpickle_name)

    with open(pkl_file_path, "wb") as outf:
        pickle.dump(data, outf, protocol=2)

    return pkl_file_path


def unpickledata(pkl_file_path):
    """ Read non-binary picle """
    with open(pkl_file_path, "rb") as inf:
        unpickled_object = pickle.load(inf)
        return unpickled_object


# TODO: wonder if I should have a cache/interface parent
class Cache(object):
    def __init__(self):
        self._app_listener = None
        self._gh_listener = None

    def set_listeners(self):
        """ different interface method from gh/app """
        self._app_listener = os.path.join(CACHE_DIR, "listen_app.pkl")
        self._gh_listener = os.path.join(CACHE_DIR, "listen_gh.pkl")

    @property
    def app_listener(self):
        if self._app_listener is None:
            cache.set_listener()
        return self._app_listener

    @property
    def gh_listener(self):
        if self._gh_listener is None:
            cache.set_listener()
        return self._gh_listener

    def send(self, v):
        return pickledata(v, self.gh_listener)

    def recieve(self):
        return unpickledata(self.app_listener)


if __name__ == "__main__":
    pass
