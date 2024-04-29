# Main script brings all parts together
import numpy as np

# Import classes and functions from modules
import model
from video_capture import VideoCapture
from lane_detection import LaneDetector
import projection
import cv2
import config

# Define model path and load model
model_path = 'models/unet_lane_detection-TuSimple_01_roi.pth'

unet = LaneDetector(model_path=model_path, temporal_window=15)

# Initialize video capture
video_capture = VideoCapture(video_path='video_capture/test_curved2.mp4')


while True:
    # Capture frame of a video
    frame = video_capture.capture_frame()


    if frame is None:
        break  # Exit the loop at the end of the video

    # Preprocess the frame and apply ROI
    pre_processed_frame = unet.pre_process_frame(frame, config.roi_coordinates)

    # Predict lane markings
    prediction = unet.make_prediction(pre_processed_frame)

    # Temporal averaging
    avg_prediction = unet.temporal_avg()

    # Predictions postprocessing
    binary_mask = unet.post_process_prediction(prediction, avg_prediction)

    # Calculate lane centers

    # Extract lane marking and fit curves
    lane_marking = projection.LaneMarking(binary_mask, frame)
    lane_marking.extract_lane_markings(num_contours=5) # Set a number of contours (default=3)
    contours_segments = lane_marking.split_contours()
    left_centers, right_centers = lane_marking.filter_contours(contours_segments=contours_segments)
    lane_marking.fit_polynomial_curve(left_centers, right_centers)
    #lane_marking.avg_polynomials()
    # Project lane markings into the frame
    lane_marking.project_lane_marking()



    # Project centers into the frame
    #lane_marking.draw_lane_center()

    # Display the frame
    cv2.imshow('Lane Detection: U-Net', frame)
    #cv2.imshow('Lane Detection', projected_frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
