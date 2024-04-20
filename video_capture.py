# Manages video input and image capture

# Imports
import cv2

class VideoCapture:
    def __init__(self, video_path):
        # Initialize video capture (video file)
        self.cap = cv2.VideoCapture(video_path)

        # Check if the file is opened successfully
        if not self.cap.isOpened():
            raise ValueError(f"Error: Couldn't open the file {video_path}")

    def capture_frame(self):
        # captures and returns a video frame
        ret, frame = self.cap.read()

        if ret is not True:
            self.release()
            return None

        return frame

    def release(self):
        # Release the video capture object
        self.cap.release()