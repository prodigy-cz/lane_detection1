import config
import projection
import lane_detection
import numpy as np
import cv2

class Trajectory_estimator:
    def __init__(self, left_coeffs, right_coeffs):
        # Initialise Trajectory estimator with given lane boundaries (countours segments CoM points)
        self.left_line_coeffs = left_coeffs
        self.right_line_coeffs = right_coeffs
        self.trajectory = None
        self.bottommost_trajectory_point = None

    def update_boundaries(self, left_coeffs, right_coeffs):
        self.left_line_coeffs = left_coeffs
        self.right_line_coeffs = right_coeffs
        # print('left and right coeffs: ', self.left_line_coeffs, self.right_line_coeffs)

    def calculate_trajectory(self, frame, binary_mask):
        # Given polynomials into np arrays
        left_coeffs = np.array(self.left_line_coeffs)
        right_coeffs = np.array(self.right_line_coeffs)

        # Evaluate polynomials
        start_height = frame.shape[0] - binary_mask.shape[0]
        end_height = frame.shape[0]
        y_values = np.linspace(start=start_height, stop=end_height)  # Correction shift -> start projection from this part of image

        # Evaluate polynomials
        left_points = np.polyval(left_coeffs, y_values)
        right_points = np.polyval(right_coeffs, y_values)

        # Calculate trajectory points as the center points of given boundaries
        center_points = (left_points + right_points) / 2

        # Fit a curve to centerpoints to approximate the trajectory
        center_coeffs = np.polyfit(y_values, center_points, 2)
        center_curve = np.poly1d(center_coeffs)

        # Draw the trajectory on the frame
        for y in range(start_height, end_height, 10):
            x = int(center_curve(y))
            cv2.circle(frame, (x, y), radius=2, color=(0, 255, 255), thickness=-1)  # Draw a filled circle for each trajectory point

        # Store the trajectory polynomial
        self.trajectory = center_curve

        # Bottommost trajectory point
        bottommost_y = end_height # - 1
        # Get corresponding x value
        bottommost_x = center_curve(bottommost_y)

        self.bottommost_trajectory_point = (int(bottommost_x), int(bottommost_y))

        return self.trajectory

    def get_bottommost_trajectory_point(self):
        if self.bottommost_trajectory_point:
            # print(f"Bottommost point of the trajectory: {self.bottommost_trajectory_point}")
            return self.bottommost_trajectory_point
