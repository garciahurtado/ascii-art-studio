from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
import sounddevice as audio
import time

class VideoPlayer():
    """A simple video player that uses MoviePy"""

    def __init__(self, video_filename, resolution=None, zoom=1/2, sound_rate=44100):
        """Constructor for VideoPlayer"""

        self.sound_rate = sound_rate
        self.audio = None
        self.start_time = None
        self.paused = False

        if resolution is None:
            # First we open it just to get the resolution metadata
            video_meta = VideoFileClip(
                video_filename,
                target_resolution=None,
                resize_algorithm='neighbor'
            )
            resolution = video_meta.reader.size
            video_meta.close()

        self.resolution = [int(resolution[0]*zoom), int(resolution[1]*zoom)]

        # Since we will be doubling the resolution as the final stop, we want to shrink the video first to compensate
        self.video = VideoFileClip(
            video_filename,
            target_resolution=(self.resolution[1], self.resolution[0]),
            resize_algorithm='bicubic'

        )
        self.meta = self.video.reader.infos
        self.num_frames = self.meta['video_nframes']
        self.fps = self.meta['video_fps']
        self.duration = self.meta['video_duration']
        self.frames = self.video.iter_frames()
        self.current_frame = 0

        #audioclip = AudioFileClip(video_filename)
        #self.audio = self.video.set_audio(audioclip)
        #self.audio = self.video.audio.to_soundarray(fps=audioclip.fps, buffersize=50000, nbytes=4)


    def play(self, sound=False):
        self.paused = False
        self.start_time = time.time()
        #audio.play(self.audio)

        #if(sound):
        #    audio.play(self.audio, self.sound_rate)

    def pause(self):
        self.paused = True

    def get_frame(self, timecode=None):
        """Return the frame appearing at timecode from the start of the video, or calculate the current
        frame if no timecode is passed, and return that image."""

        if(timecode is None):
            if not self.paused:
                now = time.time()
            else:
                now = self.start_time

            timecode = now - self.start_time

        return self.video.get_frame(timecode)

    def get_next_frame(self, limit = None):
        if limit and self.current_frame >= limit:
            return False

        self.current_frame += 1

        # Self.frames is a generator
        return self.frames
