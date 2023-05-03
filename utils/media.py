from moviepy.video.io.VideoFileClip import VideoFileClip


def get_metadata(media_file):
    video = VideoFileClip(str(media_file))

    duration = video.duration
    fps = round(video.fps)

    return duration, fps
