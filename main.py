# Main script brings all parts together

# Import classes and functions from modules
from video_capture import VideoCapture
from lane_detection import LaneDetector
import projection
import cv2
import config
from trajectory_estimator import Trajectory_estimator
from vehicle_position import VehiclePositionEstimator
import cProfile
import pstats

# Define model path and load model
model_path = 'models/unet_lane_detection-TuSimple_01_roi.pth'

unet = LaneDetector(model_path=model_path, temporal_window=5)

# Initialize video capture
video_capture = VideoCapture(video_path='video_capture/test_curved2.mp4')

# Initialize position estimator
position_estimator = VehiclePositionEstimator(
    config.FOCAL_LENGTH,
    config.INPUT_IMAGE_WIDTH,
    config.PITCH_ANGLE,
    config.CAMERA_HEIGHT,
    config.V_FOV
)

# Initialize trajectory estimator
initial_left_coeffs = []
initial_right_coeffs = []
trajectory_estimator = Trajectory_estimator(initial_left_coeffs, initial_right_coeffs)

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
    lane_marking.extract_lane_markings(num_contours=5)  # Set a number of the biggest contours to be detected

    # Segment detected contours for smoother polyfit and lane center calculation
    contours_segments = lane_marking.split_contours()

    # Filtering contours based on contours segments center of mass average position
    # Only current lane's boundaries are left after filtering
    left_filtered_centers, right_filtered_centers = lane_marking.filter_contours(contours_segments=contours_segments)

    lane_marking.check_detection()

    # With averaging
    # Fit detected contours with polynomials and add their coefficients in marking history (prev_fitted_contours)
    if left_filtered_centers and right_filtered_centers:
        lane_marking.fit_polynomial_curve()
    else:
        print('Left_centers of right_centers list is empty. Unable to fit polynomials')

    # Interpolate the correct detections with appropriate weights
    lane_marking.avg_polynomials()
    left_coeffs, right_coeffs = projection.avg_coefficients
    """
    # Without averaging
    if left_filtered_centers and right_filtered_centers:
        left_coeffs, right_coeffs = lane_marking.fit_polynomial_curve()
    else:
        print('Left_centers of right_centers list is empty. Unable to fit polynomials')
    """

    # Project lane markings into the frame
    lane_marking.project_lane_marking()

    # Update boundaries and Calculate trajectory points
    trajectory_estimator.update_boundaries(left_coeffs, right_coeffs)
    trajectory_estimator.calculate_trajectory(frame, binary_mask)

    # Vehicle relative position towards the centerline
    bottommost_point = trajectory_estimator.get_bottommost_trajectory_point()
    distance, text = position_estimator.get_relative_position(bottommost_point)

    # Put distance info into frame
    text_position = (50, 50)
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 2
    text_color = (255, 255, 255) # White in BGR format
    cv2.putText(frame, text, text_position, font, font_scale, color=(255, 255, 255), thickness=2)

    # Display the frame
    cv2.imshow('Lane Detection: U-Net', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
