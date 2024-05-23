# Visualization of lane marking and lane center

# Imports
from helper_functions import *

# Define blank lists to store curves coefficients
curves_for_avg = [[], []]
avg_coefficients = [[], []]


class LaneMarking:
    def __init__(self, binary_mask, frame):
        # Initialize Lane marking object
        self.frame = frame
        self.binary_mask = binary_mask
        self.largest_contours = None
        self.fitted_curves = []
        self.correction_shift = self.frame.shape[0] - self.binary_mask.shape[0]
        self.left_filtered_centers = []
        self.right_filtered_centers = []
        self.left_intersection_points = []
        self.right_intersection_points = []

    def extract_lane_markings(self, num_contours):
        # Extracts the largest lane markings from binary mask
        resized_mask = self.binary_mask
        contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Pick the largest contours from the sorted list
        self.largest_contours = contours[:num_contours]  # Number of contours to be extracted

        # Adjust the largest contours coordinates
        for contour in self.largest_contours:
            # Shift the y-coordinates of the contour points to align with the bottom of original frame
            for point in contour:
                point[0][1] += self.correction_shift

        # Draw the largest contours
        # for contour in self.largest_contours:
        # cv2.drawContours(self.frame, self.largest_contours, -1, (0, 0, 255), 2)

    # Split contours row-wise into smaller segments
    def split_contours(self):
        num_segments = 80
        segment_height = self.frame.shape[0] // num_segments  # Height of the segments after split
        contours = self.largest_contours.copy()
        contours_segments = []  # List of segments of all contours

        for contour in contours:
            contour_segments = []  # list of segments of one contour
            for i in range(num_segments):
                # Define segments boundaries for current segment
                start_y = i * segment_height  # Starting y value
                end_y = min(start_y + segment_height, self.frame.shape[0])  # End y value with min for cases end_y >
                # image height
                # Store points of the segment
                segment_points = []
                for point in contour:
                    if start_y <= point[0][1] < end_y:
                        segment_points.append(point)
                # Create contour from segment_points
                contour_segment = np.array(segment_points)

                if np.any(contour_segment):
                    # If the segment consists of some points append the segment to the list of segment of the current
                    # contour
                    contour_segments.append(contour_segment)
            # Append segments of current contour to the list of segments of all contours
            contours_segments.append(contour_segments)
        return contours_segments

    def filter_contours(self, contours_segments):
        left_centers = []
        right_centers = []
        left_intersection_points = []
        right_intersection_points = []
        left_filtered_centers = []
        right_filtered_centers = []

        for contour_segments in contours_segments:
            centers_of_mass = []
            for contour_segment in contour_segments:
                # Get the center of mass of the segment
                center_of_mass = get_center_of_mass(contour_segment)
                if center_of_mass is not None:
                    centers_of_mass.append(center_of_mass)

            centers_array = np.array(centers_of_mass)

            # Check if there are at least two centers of mass
            if centers_array.size > 1:
                # Check the average position of the center of mass
                average_x = np.mean(centers_array[:, 0])
                # print('average_x:', average_x)

                # Check if the average center is close to the center of the image (center of the vehicle)
                if (self.frame.shape[1] // 4) <= average_x < (self.frame.shape[1] // 2):
                    left_centers.append(centers_array)
                    # print('left centers: ', left_centers)

                elif (self.frame.shape[1] // 2) <= average_x < (self.frame.shape[1] * (1 - (1 // 4))):
                    right_centers.append(centers_array)
                    # print('right_centers: ', right_centers)
            else:
                print("No valid centers of mass found")

        # Left and right centers lists into flat arrays for DBSCAN clustering
        if left_centers:
            left_centers = np.vstack(left_centers)
        else:
            left_centers = np.array([])

        if right_centers:
            right_centers = np.vstack(right_centers)
        else:
            right_centers = np.array([])

        # Apply DBSCAN Clustering - from helper_functions
        left_clustered_centers = apply_dbscan(left_centers)
        right_clustered_centers = apply_dbscan(right_centers)

        # Calculate intersections
        left_x_range = self.frame.shape[1] * 1 // 8, self.frame.shape[1] * 4 // 8
        calculate_intersections(left_clustered_centers, self.frame, left_intersection_points, left_filtered_centers,
                                left_x_range)

        right_x_range = self.frame.shape[1] * 4 // 8, self.frame.shape[1] * 7 // 8
        calculate_intersections(right_clustered_centers, self.frame, right_intersection_points, right_filtered_centers,
                                right_x_range)

        if not left_filtered_centers or not right_filtered_centers:
            return [], []

        # Assign filtered lists of centers and intersection points into lane marking parameters
        self.left_filtered_centers = left_filtered_centers
        self.right_filtered_centers = right_filtered_centers
        self.left_intersection_points = left_intersection_points
        self.right_intersection_points = right_intersection_points

        #for center in left_filtered_centers:
         #   cv2.circle(self.frame, center, 5, (255, 255, 0), -1)

        # print('intersection_points: ', self.left_intersection_points)

        return left_filtered_centers, right_filtered_centers

    def fit_polynomial_curve(self, degree=2):
        left_centers = self.left_filtered_centers
        right_centers = self.right_filtered_centers
        # Check if left_centers and right_centers are not empty
        if left_centers:
            # Convert the list of arrays to a single array
            left_centers = np.vstack(left_centers)
            x_left_values = left_centers[:, 0]  # Extracting x values
            y_left_values = left_centers[:, 1]  # Extracting y values
        else:
            print("Left_centers list is empty.")
            return

        if right_centers:
            right_centers = np.vstack(right_centers)
            x_right_values = right_centers[:, 0]  # Extracting x values
            y_right_values = right_centers[:, 1]  # Extracting y values
        else:
            print("Right_centers list is empty.")
            return

        left_coefficients = np.polyfit(y_left_values, x_left_values, degree)
        right_coefficients = np.polyfit(y_right_values, x_right_values, degree)
        self.fitted_curves = left_coefficients, right_coefficients

        for i, coefficients in enumerate(self.fitted_curves):
            # Store coefficients for comparison
            if i == 0:
                curves_for_avg[0].append(coefficients)
            elif i == 1:
                curves_for_avg[1].append(coefficients)
            else:
                print('More than 2 lanes detected, i =', i)

        left_coeffs, right_coeffs = self.fitted_curves
        return left_coeffs, right_coeffs

    def avg_polynomials(self, window=3):  # window = num of predictions in count
        global avg_coefficients
        # Smooth detection with temporal averaging
        for j, prev in enumerate(curves_for_avg):
            if len(prev) <= 1:
                return None
            if len(prev) > window:
                prev.pop(0)

            avg_coeffs = np.mean(prev, axis=0)
            avg_coefficients[j] = avg_coeffs
        return avg_coefficients

    # Checks whether there are any centers close to the bottom of the image and add one if needed
    def check_detection(self):
        if not self.left_intersection_points or not self.right_intersection_points:
            return
        # Define range of y-values for the bottom of the image
        y_max = self.frame.shape[0]
        y_min = y_max - 80
        # Check if there are any points in the left_lane and right_lane within the specified range
        left_bottom = any(y_min <= point[1] <= y_max for point in self.left_filtered_centers)
        right_bottom = any(y_min <= point[1] <= y_max for point in self.right_filtered_centers)
        # Add a point to the lane list where no center is detected
        if not left_bottom:
            add_bottom_point(self.left_intersection_points, self.left_filtered_centers, y_max)
        if not right_bottom:
            add_bottom_point(self.left_intersection_points, self.left_filtered_centers, y_max)

    # Project the lane marking into the frame
    def project_lane_marking(self):
        # Iterate over the list with curves coefficients
        for i, curve_coefficients in enumerate(avg_coefficients):
            # Generate y values
            y_curve = np.linspace(start=(self.frame.shape[0] - 1), stop=self.correction_shift)  # Correction
            # shift - start projection from this part of the image

            # Calculate x values with polynomial coefficients
            x_curve = np.polyval(curve_coefficients, y_curve)

            # Convert x, y to integer
            curve_points = np.column_stack((x_curve.astype(int), y_curve.astype(int)))

            # Draw the curve on the frame
            cv2.polylines(self.frame, [curve_points], isClosed=False, color=(0, 255, 0), thickness=5)

        """
        for contour in self.largest_contours:

            # Find leftmost, rightmost, topmost, and bottommost points of the contour
            leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
            rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
            topmost = tuple(contour[contour[:, :, 1].argmin()][0])
            bottommost = tuple(contour[contour[:, :, 1].argmax()][0])

            # Draw points on the frame to visualize them
            cv2.circle(self.frame, leftmost, 5, (255, 0, 0), -1)
            cv2.circle(self.frame, rightmost, 5, (255, 0, 0), -1)
            cv2.circle(self.frame, topmost, 5, (255, 0, 0), -1)
            cv2.circle(self.frame, bottommost, 5, (255, 0, 0), -1)
        """

        # Draw the center of the vehicle into the final frame
        bottom_center_x = self.frame.shape[1] // 2
        bottom_center_y = self.frame.shape[0] - 1
        rectangle_width = 20
        rectangle_height = 20

        # Calculate the coordinates of the rectangle
        top_left = (bottom_center_x - rectangle_width // 2, bottom_center_y - rectangle_height)
        bottom_right = (bottom_center_x + rectangle_width // 2, bottom_center_y + rectangle_height)

        # Draw the rectangle
        cv2.rectangle(self.frame, top_left, bottom_right, (0, 0, 255), thickness=-1)

        return
