import json


def read_json_file_to_dict(filename) -> dict:
    with open(filename, 'rb') as file:
        data = json.load(file)
    return data


def write_json_file_from_dict(data: dict, filename):
    with open(filename, 'w') as file:
        json.dump(data, file)
