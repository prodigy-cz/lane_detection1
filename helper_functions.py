# Imports
from sklearn.cluster import DBSCAN
import numpy as np
import cv2


def extract_features(contours_segments):
    centers_array = []
    for contour_segments in contours_segments:
        for contour_segment in contour_segments:
            # Get the center of mass of the segment
            center_of_mass = get_center_of_mass(contour_segment)
            if center_of_mass is not None:
                centers_array.append(center_of_mass)
    return centers_array  # features


def get_center_of_mass(contour):
    # Calculate moments of the contour
    M = cv2.moments(contour)

    # Calculate the center of mass
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Center of mass
        cen_of_mass = [cx, cy]

        return cen_of_mass


def calculate_distance(x1, x2):
    return np.abs(x1 - x2)  # Return absolute x distance


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
    intersection_point = [x_intersection, height]
    # Draw the intersection point on the image
    # cv2.circle(frame, intersection_point, radius=5, color=(0, 255, 255), thickness=-1)  # Yellow circle

    return intersection_point


def calculate_intersections(clustered_centers, frame, intersection_points, filtered_centers, x_range):
    valid_intersection_points = []  # List of intersection points within the range

    for centers_array in clustered_centers:
        min_y_point = centers_array[np.argmin(centers_array[:, 1])]
        max_y_point = centers_array[np.argmax(centers_array[:, 1])]
        intersection_point = get_intersection(frame, min_y_point, max_y_point)
        # print(f"Calculated intersection point: {intersection_point}")

        # Check if the intersection_point is within range
        if x_range[0] <= intersection_point[0] < x_range[1]:
            # Add valid point to the list
            valid_intersection_points.append(intersection_point)
            # Add valid centers array to filtered centers
            filtered_centers.extend(centers_array)

    # Store valid intersection points for check_detection method
    intersection_points.extend(valid_intersection_points)


def add_bottom_point(intersection_points, lane_centers, y_max):
    # Calculate average intersection x value
    points_array = np.array(intersection_points)
    x_values = points_array[:, 0]
    avg_x = np.mean(x_values)
    bottom_point = [avg_x, y_max]
    # Add the point to lane centers for polynomial fitting
    lane_centers.append(bottom_point)
    return


def apply_dbscan(centers, eps=25, min_samples=2):
    if centers.size > 0:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels_dbscan = dbscan.fit_predict(centers)
        unique_labels = np.unique(labels_dbscan)

        clustered_centers = []
        for label in unique_labels:
            if label != -1:  # Ignore noise points
                clustered_centers.append(centers[labels_dbscan == label])
        return clustered_centers

    return []
