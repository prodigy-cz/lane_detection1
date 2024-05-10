# Visualization of lane marking and lane center

# Imports
import cv2
import numpy as np
from sklearn.cluster import KMeans

def cluster(X):
    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fix(X)

def get_center_of_mass(frame, contour):
    # for contour in self.largest_contours:
        # Calculate moments of the contour
        M = cv2.moments(contour)

        # Calculate the center of mass
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Center of mass
            cen_of_mass = [cx, cy]

            # Draw a circle at the center of mass
            #cv2.circle(frame, (cx, cy), 5, (5, 94, 255), -1)
            return cen_of_mass


def calculate_distance(x1, x2):
    return np.abs(x1 - x2) # return absolute x distance


def get_intersection(frame, point1, point2):

    # Get intersection of line with bottom of the frame

    # Frame height
    height = frame.shape[0]

    # Check if the line is not vertical
    if point2[0] - point1[0] != 0:
        # Calculate slope m
        m = (point2[1] - point1[1]) / (point2[0] - point1[0])
        # Calculate y intersect
        b = point1[1] - m * point1[0]

        # Calculate the intersection with bottom edge
        x_intersection = int((height - b) / m)

    else:
        # If the line is vertical, x_intersection = x coordinate of one of the points
        x_intersection = point1[0]

    # Intersection point
    intersection_point = (x_intersection, height)
    # Draw the circle on the image
    cv2.circle(frame, intersection_point, radius=5, color=(0, 255, 255), thickness=-1)  # Yellow circle

    print(intersection_point)

    return intersection_point

"""
def calculate_weight_factors(intersection_points):
    # Line Segment weighting strategy implementation
    # Line Segment weighting strategy implementation
    weights = []
    num_points = len(intersection_points)

    # Check if there's only one intersection point
    if num_points == 1:
        # Set a default weight value
        weights.append(1.0)
    else:
        for i, point_i in enumerate(intersection_points):
            weight = 0
            for j, point_j in enumerate(intersection_points):
                if i != j:  # Exclude current point from distance calculation
                    distance = calculate_distance(point_i, point_j)
                    weight += np.exp(-distance)
            weights.append(weight)

    return weights
"""
"""
def filter_outlying_arrays(list_of_centers_arrays, intersection_points, threshold):
    filtered_arrays = []
    weights = calculate_weight_factors(intersection_points)
    print('weights: ', weights)
    max_weight = max(weights)
    for i, weight in enumerate(weights):
        if weight / max_weight >= threshold:
            print('i: ', i)
            print('list_of_centers_arrays: ', list_of_centers_arrays)
            filtered_arrays.extend(list_of_centers_arrays[i])
    return filtered_arrays
"""

# Define blank lists to store curves coefficients
prev_fitted_curves = [[], []]
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

    def extract_lane_markings(self, num_contours):
        # Extracts the largest lane markings from binary mask
        resized_mask = self.binary_mask
        contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Pick the largest contours from the sorted list
        self.largest_contours = contours[:num_contours] # Number of contours to be extracted

        # Adjust the largest contours coordinates
        for contour in self.largest_contours:
            # Shift the y-coordinates of the contour points to align with the bottom of original frame
            for point in contour:
                point[0][1] += self.correction_shift

        # Draw the largest contours
        #for contour in self.largest_contours:
            #cv2.drawContours(self.frame, self.largest_contours, -1, (0, 0, 255), 2)


    def split_contours(self):
        # Split contours row-wise into smaller segments
        num_segments = 40
        segment_height = self.frame.shape[0] // num_segments # height of the segments after split
        contours = self.largest_contours.copy()
        #print('\n New Frame:')

        contours_segments = [] # List of segments of all contours
        for contour in contours:
            #print('\n New Contour:')
            contour_segments = [] # list of segments of one contour
            for i in range(num_segments):
                # Define segments boundaries for current segment
                start_y = i * segment_height # Starting y value
                end_y = min(start_y + segment_height, self.frame.shape[0]) # End y value with min for cases end_y > image height

                # Store points of the segment
                segment_points = []
                for point in contour:
                    if start_y <= point[0][1] < end_y:
                        segment_points.append(point)

                # Create contour from segment_points
                contour_segment = np.array(segment_points)

                if np.any(contour_segment):
                    # If the segment consists of some points append the segment to the list of segment of the current contour
                    contour_segments.append(contour_segment)

            # Append segments of current contour to the list of segments of all contours
            contours_segments.append(contour_segments)

            # Iterate over contour segments and draw each contour separately
            for segment_contour in contour_segments:
               cv2.drawContours(self.frame, [segment_contour], -1, color=(0, 100, 100), thickness=2)

        return contours_segments

    def filter_contours(self, contours_segments):
        contours_segments = contours_segments
        left_centers = []
        right_centers = []
        left_intersection_points = []
        right_intersection_points = []
        left_filtered_centers = []
        right_filtered_centers = []
        filtering_threshold = 0.5

        for contour_segments in contours_segments:
            centers_of_mass = []
            for contour_segment in contour_segments:
                # Get the center of mass of the segment
                center_of_mass = get_center_of_mass(self.frame, contour_segment)
                if center_of_mass is not None:
                    centers_of_mass.append(center_of_mass)

            # Convert centers_of_mass to a np array
            centers_array = np.array(centers_of_mass)

            # Check if there are at least two centers of mass
            if centers_array.size > 1:
                # Check the average position of the center of mass
                average_x = np.mean(centers_array[:, 0])
                #print('average_x:', average_x)

                # Check if the average center is close to the center of the image (center of the vehicle)
                if (self.frame.shape[1] // 4) <= average_x < (self.frame.shape[1] // 2):
                    left_centers.append(centers_array)
                    print('left centers: ', left_centers)

                elif (self.frame.shape[1] // 2) <= average_x < (self.frame.shape[1] * (1 - (1 // 4))):
                    right_centers.append(centers_array)
                    print('right_centers: ', right_centers)
            else:
                print("No valid centers of mass found")

        for centers_array in left_centers:
            min_y_point = centers_array[np.argmin(centers_array[:, 1])]  # Picks point with min y value
            max_y_point = centers_array[np.argmax(centers_array[:, 1])]  # Picks point with min y value
            intersection_point = get_intersection(self.frame, min_y_point, max_y_point)
            print('x_intersection_point: ', intersection_point)
            left_intersection_points.append(intersection_point)

        for k, point in enumerate(left_intersection_points):
            if (self.frame.shape[1] * (3 // 8)) <= point[0] < (self.frame.shape[1] * (8 // 8)):
                left_filtered_centers.extend(left_centers[k])

        for centers_array in right_centers:
            min_y_point = centers_array[np.argmin(centers_array[:, 1])]  # Picks point with min y value
            max_y_point = centers_array[np.argmax(centers_array[:, 1])]  # Picks point with min y value
            intersection_point = get_intersection(self.frame, min_y_point, max_y_point)
            right_intersection_points.append(intersection_point)


        for w, point in enumerate(right_intersection_points):
            if (self.frame.shape[1] * (5 // 8)) <= point[0] < (self.frame.shape[1] * (8 // 8)):
                right_filtered_centers.extend(right_centers[w])

        print('left_intersection_points: ', left_intersection_points)
        #left_filtered_centers = filter_outlying_arrays(left_centers, left_intersection_points, filtering_threshold)
        #right_filtered_centers = filter_outlying_arrays(right_centers, right_intersection_points, filtering_threshold)


        if not left_filtered_centers or not right_filtered_centers:
            print("No valid left or right filtered centers found.")
            # Handle the case when no valid centers are found, for example:
            return [], []

        for point in left_filtered_centers:
            cv2.circle(self.frame, point, 5, (255, 255, 255), -1)

        return left_filtered_centers, right_filtered_centers


    def compare_frame(self, lane_num):
        # Compare current frame detection with previous number of predictions
        if len(prev_fitted_curves) > 2:
            prev_fitted_curves.pop(0)
            print(prev_fitted_curves)
        if len(avg_coefficients[lane_num]) < 3:
            return True  # No coefficients to compare with
        else:
            current_coefficients = self.fitted_curves[lane_num]
            previous_coefficients = prev_fitted_curves[lane_num]
            # Iterate over coeffs array = over all lane markings
            # a and b values from f(y)=ay²+by+c
            current_a = current_coefficients[0]
            current_b = current_coefficients[1]
            current_c = current_coefficients[2]
            previous_a = previous_coefficients[-2][0]
            previous_b = previous_coefficients[-2][1]
            previous_c = previous_coefficients[-2][2]


            # Percentage difference
            percent_a = abs(current_a - previous_a) / abs(previous_a) * 100
            percent_b = abs(current_b - previous_b) / abs(previous_b) * 100
            percent_c = abs(current_c - previous_c) / abs(previous_c) * 100
            #print('current coeffs: ', current_coefficients)
            #print('current a:', current_a, 'prev a', previous_a)
            #print('current b:', current_b, 'prev b', previous_b)
            #print('current c:', current_c, 'prev c', previous_c)
            print('percent diff: ', percent_a, percent_b, percent_c)
            # Check percentage criteria
            #if percent_a <= 30 and percent_b <= 30: # Not used for unstable behavior
            if percent_a:
                return True
        return False

    def fit_polynomial_curve(self, left_centers, right_centers, degree=2):
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
                prev_fitted_curves[0].append(coefficients)
            elif i == 1:
                prev_fitted_curves[1].append(coefficients)

            # If the current frame coeffs are similar to previous, add it to the list for averaging
            if self.compare_frame(lane_num=i):
                # Store fitted curves coefficients into curves_for_avg list to use it for averaging and decision-making
                if i == 0:
                    curves_for_avg[0].append(coefficients)
                elif i == 1:
                    curves_for_avg[1].append(coefficients)
                else:
                    print('More than 2 lanes detected, i =', i)
        print('fitted_curves: ', self.fitted_curves)
        left_coeffs, right_coeffs = self.fitted_curves
        return left_coeffs, right_coeffs

    def avg_polynomials(self, window=5): # window = num of predictions in count
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

    def project_lane_marking(self):
        # Project the lane marking into the frame
        # Iterate over the list with curves coefficients
        # for curve_coefficients in avg_coefficients: # With averaging
        for curve_coefficients in self.fitted_curves:
            # Generate y values
            y_values = np.linspace(start=self.correction_shift, stop=self.frame.shape[0]) # Correction shift -> start projection from this part of image
            # Calculate x values with polynomial coefficients
            x_values = np.polyval(curve_coefficients, y_values)

            # Convert x, y to integer
            curve_points = np.column_stack((x_values.astype(int), y_values.astype(int)))

            # Draw the curve on the frame
            cv2.polylines(self.frame, [curve_points], isClosed=False, color=(0, 255, 0), thickness=5)



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

            # Check if the contour has at least 5 points
            #if len(contour) >= 5:
                # Calculate moments of the contour
                #get_center_of_mass(self.frame, contour)

            #get_intersection(self.frame, topmost, bottommost)

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