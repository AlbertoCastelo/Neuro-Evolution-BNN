from neat.configuration import get_configuration, DefaultConfiguration


def main():
    default_config = DefaultConfiguration()
    default_config._write_to_json(filename='./files_generated/sample-config.json')

if __name__ == '__main__':
    main()