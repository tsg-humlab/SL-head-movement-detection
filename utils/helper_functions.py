import pandas as pd

def sort_dict(dictionary):
    return {key: value for key, value in sorted(dictionary.items())}
