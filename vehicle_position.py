# Vehicle position estimation

# Imports
import numpy as np


class VehiclePositionEstimator:
    def __init__(self, focal_length, image_width, pitch_angle, camera_height, V_FoV):
        self.focal_length = focal_length
        self.image_width = image_width
        self.pitch_angle = pitch_angle
        self.camera_height = camera_height
        self.V_FoV = V_FoV
        self.vehicles_position_x = image_width // 2

    def get_relative_position(self, bottommost_trajectory_point):
        # Vehicle's x position
        vehicle_position_x = self.vehicles_position_x
        # Bottommost point of the trajectory
        bottommost_trajectory_x = bottommost_trajectory_point[0]
        # Horizontal distance in pixels
        px_distance = vehicle_position_x - bottommost_trajectory_x
        # Convert pixel distance into real distance
        rw_distance = self.px_to_rw(px_distance)
        return rw_distance

    def px_to_rw(self, px_distance):
        # Calculate the angle between ground vertical axis and bottom edge of view
        theta_b = self.pitch_angle - self.V_FoV / 2
        # Calculate the distance between camera center of coordinates and bottommost point
        l_c = self.camera_height / np.cos(theta_b)
        # Calculate angle between l_c and bottommost centerline point
        beta_c = np.arctan(px_distance / self.focal_length)
        # Calculate the real world distance
        rw_distance = abs(l_c * np.tan(beta_c))
        if px_distance == 0:
            # Assign statement for visualization
            text = f"Vehicle is positioned in the center"
            return rw_distance, text
        elif px_distance < 0:
            side = "left"
        else:
            side = "right"
        # Assign statement for visualization
        text = f"Vehicle is shifted {rw_distance:.3f} m {side} from centerline"
        return rw_distance, text
