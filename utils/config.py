import pickle

import yaml
from pathlib import Path


class Config:
    def __init__(self, config_path=Path(__file__).parent.parent.resolve() / 'config.yml'):
        self.config_path = config_path
        self.content = self.load_config()

        self.overview = self.content['overview']

    def load_config(self):
        with open(self.config_path, 'r') as input_handle:
            config = yaml.safe_load(input_handle)

        return config

    def load_data_dump_labels(self):
        with open(self.content['data_dump']['labels'], 'rb') as input_handle:
            video_data = pickle.load(input_handle)

        return video_data
