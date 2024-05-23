from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np

def main():
    video_frames = read_video('input/08fd33_0.mp4')
    tracker = Tracker('model/best.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                   read_from_stub=True,
                                   stub_path='stubs/track_stubs.pkl')
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    save_video(output_video_frames, 'output/output_video.avi')


if __name__ == '__main__':
    main()