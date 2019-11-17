import logging
import sys
from logging.handlers import RotatingFileHandler

import os

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOG_FILENAME = 'bayesian_neat.log'
VERSION_FILE = os.environ.get('VERSION_FILE')
LOG_LEVEL = 'ERROR'


def get_logger(path=None, levels={}):
    if _Logger._instance is None or path is not None:
        _Logger(path=path, levels=levels)
    return _Logger._instance


class _Logger:
    _instance = None

    def __init__(self, path, levels: dict):
        self.path = path
        self._initialize_log_directory()

        # initialize custom levels
        for i, level in enumerate(levels):
            setattr(logging.Logger, level, add_debug_level(10 + i, name=level.upper()))
        logger = logging.getLogger()
        logger.handlers = []

        # stdout handle
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        logger = self._get_stdout_handler(logger=logger, formatter=formatter)

        if path:
            setattr(logger, 'log_base_path', self.path)
            logger = self.add_new_file_handler(formatter, 'debug', logger)

        # add handlers for custom levels if they are activated
        for level, activated in levels.items():
            if activated:
                logger = self.add_new_file_handler(formatter, level, logger)
        logger.setLevel(logging.DEBUG)

        _Logger._instance = logger

    def add_new_file_handler(self, formatter, level, logger):

        # delete file if it exist
        filename = os.path.join(self.path, f'{level}_{LOG_FILENAME}')
        if os.path.isfile(filename):
            os.remove(filename)
        logger = self._get_file_handler(logger=logger, formatter=formatter,
                                        filename=filename, level=level)
        return logger

    def _initialize_log_directory(self):
        # rotation_file handle
        if self.path:
            self.path += 'log'
            # create directory if it doesn't exist
            if not os.path.exists(self.path):
                os.makedirs(self.path)

    def _get_file_handler(self, logger, formatter, filename, level):
        handler = RotatingFileHandler(filename, maxBytes=1000000, backupCount=3)
        handler.setLevel(getattr(logging, level.upper()))
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _get_stdout_handler(self, logger, formatter):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger


def add_debug_level(num, name):
    def fn(self, message, *args, **kwargs):
        if self.isEnabledFor(num):
            self._log(num, message, args, **kwargs)

    logging.addLevelName(num, name)
    setattr(logging, name, num)
    return fn


logger = get_logger()
