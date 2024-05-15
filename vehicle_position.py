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
        # print(f"Pixel distance: {px_distance} px")

        # Convert pixel distance into real distance
        rw_distance = self.px_to_rw(px_distance)

        return rw_distance

    def px_to_rw(self, px_distance):
        # Calculate the angle between ground vertical axis and bottom edge of view
        theta_b = self.pitch_angle - self.V_FoV / 2
        # print(f"theta_b: {np.rad2deg(theta_b):.2f} degrees")
        # print(f"Vertical FOV: {np.rad2deg(self.V_FoV):.2f} degrees")

        # Calculate the distance between camera center of coordinates and bottommost point
        l_c = self.camera_height / np.sin(theta_b)
        # print(f"l_c: {l_c:.2f}")

        # Calculate angle between l_c and bottommost centerline point
        beta_c = np.arctan(px_distance / self.focal_length)
        # print(f"beta_l: {np.rad2deg(beta_c):.2f} degrees")

        # Calculate the real world distance
        rw_distance = abs(l_c * np.tan(beta_c))

        if px_distance < 0:
            side = "left"
        else:
            side = "right"

        text = f"Vehicle is shifted {rw_distance:.2f} m {side} from centerline"
        return rw_distance, text


"""
if __name__ == "__main__":
    # Parameters
    focal_length = 528  # ZED 2 focal length for HD 720
    image_width = 1280
    pitch_angle = np.deg2rad(70)
    camera_height = 1.6
    H_FoV = np.deg2rad(101)  # ZED 2 horizontal field of view for HD 720
    V_FoV = np.deg2rad(68)  # ZED 2 vertical field of view for HD 720

    position_estimator = VehiclePositionEstimator(focal_length, image_width, pitch_angle, camera_height, V_FoV)

    bottommost_point = (1280//2, 720)  # Only example

    # Get the relative position
    distance = position_estimator.get_relative_position(bottommost_point)
"""
