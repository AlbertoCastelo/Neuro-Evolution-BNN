from neat.Population import Evolution
from neat.configuration import get_configuration


def run():
    # load configuration
    config = get_configuration()

    # setup evolution
    evolution = Evolution()
    evolution.run()


if __name__ == '__main__':
    run()
