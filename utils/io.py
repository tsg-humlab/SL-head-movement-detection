import json
import os


def mkdir_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def write_json(data, path):
    with open(path, 'w') as output_handle:
        json.dump(data, output_handle)


def read_json(path):
    with open(path, 'r') as input_handle:
        data = json.load(input_handle)

    return data
