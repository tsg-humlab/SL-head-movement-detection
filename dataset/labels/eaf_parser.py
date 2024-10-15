import os
import re
from glob import glob
from pathlib import Path

from pympi.Elan import Eaf

from utils.config import Config


class HeadMovementParser:
    def __init__(self, fps):
        self.fps = fps
        self.fpms = fps * 0.001

    def parse(self, eaf):
        movements = self.parse_tier(eaf.tiers['Head movement'], eaf.timeslots)

        return movements

    def parse_tier(self, tier, timeslots):
        """ Parse a tier from an EAF file and save label, start, and end times (ms) """
        annotations = []

        for annotation in tier[0].values():
            label = annotation[2]
            start = int(timeslots[annotation[0]] * self.fpms)
            end = int(timeslots[annotation[1]] * self.fpms)

            annotations.append(Annotation(label, start, end))

        return annotations


class Video:
    def __init__(self, ngt_id, signer):
        self.ngt_id = ngt_id
        self.signer = signer

    def __getitem__(self, signer_id):
        matches = [self.signer.signer_id == signer_id]

        if matches[0]:
            return self.signer

    @classmethod
    def from_eaf(cls, eaf_path, fps=25):
        eaf = Eaf(eaf_path, 'pympi')
        ngt_id = eaf_path.split("/")[-1].split("_")[0]
        signer_id = eaf_path.split("/")[-1].split("_")[1].split(".")[0]
        movement_parser = HeadMovementParser(fps=fps)
        annotations = movement_parser.parse(eaf)

        return cls(ngt_id, Signer(signer_id, annotations))

    def __repr__(self):
        return f'Video {self.ngt_id} with {self.signer}'


class Signer:
    def __init__(self, speaker_id, annotations):
        self.signer_id = speaker_id
        self.annotations = annotations

    def __repr__(self):
        return f'Speaker {self.signer_id} with {len(self.annotations)} annotations'


class Annotation:
    def __init__(self, label, start, end):
        assert start < end

        self.label = label
        self.start = start
        self.end = end

    def __repr__(self):
        return f'Annotation {self.label} at {self.start}-{self.end}'


def main():
    config = Config()
    eaf_dir = config.content['media']['eaf']
    print(f'Searching for annotation files in {eaf_dir}')

    files = glob(fr'{eaf_dir}{os.sep}*.eaf')
    videos = []
    print(f'Found {len(files)} annotation files')

    for file in files:
        video = Video.from_eaf(file)

        if len(video.signer.annotations) > 0:
            videos.append(video)

    print('Parsed annotation directory')

    return videos


if __name__ == '__main__':
    main()
