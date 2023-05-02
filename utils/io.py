import json


def write_json(data, path):
    with open(path, 'w') as output_handle:
        json.dump(data, output_handle)


def read_json(path):
    with open(path, 'r') as input_handle:
        data = json.load(input_handle)

    return data
