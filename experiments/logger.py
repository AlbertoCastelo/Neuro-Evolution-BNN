import logging
import sys
from logging.handlers import RotatingFileHandler

import os

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOG_FILENAME = 'bayesian_neat.log'
VERSION_FILE = os.environ.get('VERSION_FILE')
LOG_LEVEL = 'ERROR'


def get_logger(path=None):
    if _Logger._instance is None or path is not None:
        _Logger(path=path)
    return _Logger._instance


class _Logger:
    _instance = None

    def __init__(self, path):
        logger = logging.getLogger()
        # logger.setLevel(logging.DEBUG)
        logger.handlers = []

        # stdout handle
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        logger = self._get_stdout_handler(logger=logger, formatter=formatter)

        # rotation_file handle
        if path:
            path += 'log'
            # create directory if it doesn't exist
            if not os.path.exists(path):
                os.makedirs(path)

            # delete file if it exist
            filename = os.path.join(path, LOG_FILENAME)
            if os.path.isfile(filename):
                os.remove(filename)
            self._get_file_handler(logger=logger, formatter=formatter, filename=filename)

        # TODO: fix logging leve. When setting to ERROR it does not log INFO nor DEBUG levels
        logger.setLevel(logging.DEBUG)
        _Logger._instance = logger

    def _get_file_handler(self, logger, formatter, filename):
        handler = RotatingFileHandler(filename, maxBytes=1000000, backupCount=3)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _get_stdout_handler(self, logger, formatter):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger


logger = get_logger()
