from functools import wraps
from time import time


def timeit(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = round(time(), 3)
        try:
            return func(*args, **kwargs)
        finally:
            end_ = round(time(), 3) - start
            message = f"Total Time in {func.__name__}: {end_} s"
            print(message)
            # logger.debug(message)
    return _time_it
