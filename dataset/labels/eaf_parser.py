from glob import glob
from pathlib import Path
import re

from pympi.Elan import Eaf

from utils.config import Config


class HeadMovementParser:
    def __init__(self, fps):
        self.fps = fps
        self.fpms = fps * 0.001

    def parse(self, eaf):
        return self.get_head_annotations(eaf)

    def get_head_annotations(self, eaf):
        movements_left = self.parse_tier(eaf.tiers['Head movement S1'], eaf.timeslots)
        movements_right = self.parse_tier(eaf.tiers['Head movement S2'], eaf.timeslots)

        return movements_left, movements_right

    @staticmethod
    def get_speaker_ids(eaf):
        speakers = set()

        for descriptor in eaf.media_descriptors:
            if descriptor['MIME_TYPE'] == 'video/mpeg':
                filename = Path(descriptor['MEDIA_URL']).stem

                hit = re.search('S[0-9][0-9][0-9]', filename)
                if hit is not None:
                    speakers.add(hit.group())

        speakers = list(speakers)

        if len(speakers) == 1:
            if int(speakers[0][1:]) % 2 == 0:
                speakers.insert(0, None)
            else:
                speakers.append(None)
        elif len(speakers) != 2:
            raise RuntimeError(f'The number of speakers in a given CorpusNGT video should be 2, {len(speakers)} given')
        elif int(speakers[0][1:]) % 2 == 0:
            speakers.reverse()

        return speakers

    def parse_tier(self, tier, timeslots):
        annotations = []

        for annotation in tier[0].values():
            label = annotation[2]
            start = int(timeslots[annotation[0]] * self.fpms)
            end = int(timeslots[annotation[1]] * self.fpms)

            annotations.append(Annotation(label, start, end))

        return annotations


class Video:
    def __init__(self, ngt_id, signer_left, signer_right):
        self.ngt_id = ngt_id
        self.signer_left = signer_left
        self.signer_right = signer_right

    @classmethod
    def from_eaf(cls, ngt_id, eaf, fps=25):
        movement_parser = HeadMovementParser(fps=fps)
        signer_id_left, signer_id_right = movement_parser.get_speaker_ids(eaf)
        annotations_left, annotations_right = movement_parser.parse(eaf)

        return cls(ngt_id, Signer(signer_id_left, annotations_left), Signer(signer_id_right, annotations_right))

    def __repr__(self):
        return f'Video {self.ngt_id} with {self.signer_left} (left) and {self.signer_right} (right)'


class Signer:
    def __init__(self, speaker_id, annotations):
        self.signer_id = speaker_id
        self.annotations = annotations

    def __repr__(self):
        return f'Speaker {self.signer_id} with {len(self.annotations)} annotations'


class Annotation:
    def __init__(self, label, start, end):
        self.label = label
        self.start = start
        self.end = end

    def __repr__(self):
        return f'Annotation {self.label} at {self.start}-{self.end}'


def main():
    config = Config()
    eaf_dir = config.content['media']['body_720']
    files = glob(fr'{eaf_dir}\*.eaf')
    videos = []

    print(f'Found {len(files)} annotation files')

    for file in files:
        eaf = Eaf(file, 'pympi')
        video = Video.from_eaf(Path(file).stem, eaf)

        if len(video.signer_left.annotations) > 0 or len(video.signer_right.annotations) > 0:
            videos.append(video)

    print('Parsed annotation directory')

    return videos


if __name__ == '__main__':
    main()
