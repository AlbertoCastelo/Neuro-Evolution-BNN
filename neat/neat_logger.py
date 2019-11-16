
from experiments.logger import logger, get_logger
from neat.configuration import get_configuration

# log-levels to the left will include the ones to the right.
# For instance, 'time' will only include TIME and INFO
NEAT_LEVELS = ['network', 'population', 'time']


def get_neat_logger(path=None):
    config = get_configuration()
    neat_levels_activation = []
    for level in NEAT_LEVELS:
        activated = getattr(config, f'log_{level}', False)
        if activated:
            neat_levels_activation.append(True)
        else:
            neat_levels_activation.append(False)
    levels = dict(zip(NEAT_LEVELS, neat_levels_activation))

    logger = get_logger(path=path, levels=levels)
    return logger
