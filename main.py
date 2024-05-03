# Main script brings all parts together

# Import classes and functions from modules
import model
from video_capture import VideoCapture
from lane_detection import LaneDetector
import projection
import cv2
import config
from trajectory_estimator import Trajectory_estimator

# Define model path and load model
model_path = 'models/unet_lane_detection-TuSimple_01_roi.pth'

unet = LaneDetector(model_path=model_path, temporal_window=15)

# Initialize video capture
video_capture = VideoCapture(video_path='video_capture/test_curved.mp4')



# Initialize trajectory estimator
initial_left_coeffs = []
initial_right_coeffs = []
initial_trajectory_points = []
trajectory_estimator = Trajectory_estimator(initial_left_coeffs, initial_right_coeffs, initial_trajectory_points)

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

    # Initialize the lane_marking object
    lane_marking = projection.LaneMarking(binary_mask, frame)

    # Extract lane marking
    lane_marking.extract_lane_markings(num_contours=5) # Set a number of the biggest contours to be detected

    # Segment detected contours for smoother polyfit and lane center calculation
    contours_segments = lane_marking.split_contours()

    # Filtering contours based on contours segments center of mass average position
    # Only current lane's boundaries are left after filtering
    left_centers, right_centers = lane_marking.filter_contours(contours_segments=contours_segments)

    # Fit detected contours with polynomials and add their coefficients in marking history (prev_fitted_contours)
    lane_marking.fit_polynomial_curve(left_centers, right_centers)

    # Interpolate the correct detections with appropriate weights
    lane_marking.avg_polynomials()
    left_coeffs, right_coeffs = projection.avg_coefficients

    # Project lane markings into the frame
    lane_marking.project_lane_marking()

    # Update boundaries and Calculate trajectory points
    trajectory_estimator.update_boundaries(left_coeffs, right_coeffs)
    trajectory_estimator.calculate_trajectory(frame, binary_mask)

    # Display the frame
    cv2.imshow('Lane Detection: U-Net', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
