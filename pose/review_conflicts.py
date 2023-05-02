import argparse
import os
from glob import glob
from pathlib import Path

from pose.analyze_results import is_output_dir
from pose import KEYPOINTS
from utils.io import write_json, read_json


def review(results_dir, output_json):
    assert is_output_dir(results_dir)

    if not os.path.exists(output_json):
        write_json({}, output_json)

    files = glob(results_dir / KEYPOINTS)

    ...


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', metavar='results-dir', type=Path)
    parser.add_argument('output_json', metavar='output-json', type=Path)
    args = parser.parse_args()

    review(args.results_dir, args.output_json)
