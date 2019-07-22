import os

from neat.configuration import get_configuration


def create_configuration(filename):
    path = get_config_files_path()
    filename = ''.join([path, filename])
    return get_configuration(filename=filename)


def get_config_files_path():
    return os.path.dirname(os.path.realpath(__file__))
