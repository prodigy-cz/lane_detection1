# Visualization of lane marking and lane center

# Imports
import cv2
import numpy as np
import config

prev_fitted_curves = [[], [], []]
prev_fitted_curves1 = []
prev_fitted_curves2 = []
prev_fitted_curves3 = []
avg_coefficients = [[], [], []]


class LaneMarking:
    def __init__(self, binary_mask, frame):
        # Initialize Lane marking object
        self.frame = frame
        self.binary_mask = binary_mask
        self.largest_contours = None
        self.fitted_curves = []
        self.correction_shift = self.frame.shape[0] - self.binary_mask.shape[0]

    def extract_lane_markings(self, num_contours=1):
        # Extracts the largest lane markings from binary mask
        resized_mask = self.binary_mask
        contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Pick the largest contours from the sorted list
        self.largest_contours = contours[:3]

    def compare_frame(self, lane_num):
        # Compare current frame detection with previous number of predictions

        if len(avg_coefficients[lane_num]) < 1:
            return True  # No coefficients to compare with
        else:
            current_coefficients = self.fitted_curves[lane_num]
            # Iterate over coeffs array = over all lane markings
            # a and b values from f(y)=ay²+by+c
            current_a = current_coefficients[0]
            current_b = current_coefficients[1]
            previous_a = avg_coefficients[lane_num][0]
            previous_b = avg_coefficients[lane_num][1]

            # Percentage difference
            percent_a = abs(current_a - previous_a) / abs(previous_a) * 100
            percent_b = abs(current_b - previous_b) / abs(previous_b) * 100
            print('percent diff: ', percent_a, percent_b)
            # Check percentage criteria
            if percent_a <= 50 and percent_b <= 50:
                return True
        return False

    def fit_polynomial_curve(self, degree=2):
        # Fit polynomial curve to each lane marking
        for i, marking in enumerate(self.largest_contours):
            # Extract marking coordinates
            x_coordinates = marking[:, 0, 0]
            y_coordinates = marking[:, 0, 1]
            # Shift the y_coordinates to fit the original frame
            y_coordinates_shifted = y_coordinates + self.correction_shift
            # Fit a polynomial curve to the contour
            coefficients = np.polyfit(y_coordinates_shifted, x_coordinates, degree)
            self.fitted_curves.append(coefficients)

            # If the current frame coeffs are similar to previous, add it to the list for averaging
            if self.compare_frame(lane_num=i) == True or self.compare_frame(lane_num=i) == False:
                # Store fitted curves coefficients into prev_fitted_curves list to use it for averaging and decision-making
                if i == 0:
                    prev_fitted_curves[0].append(coefficients)
                elif i == 1:
                    prev_fitted_curves[1].append(coefficients)
                elif i == 2:
                    prev_fitted_curves[2].append(coefficients)
                else:
                    print('More than 3 lanes detected, i =', i)
        print('prev_fitted_curves: ', prev_fitted_curves)

    def avg_polynomials(self):
        global avg_coefficients
        # Interpolate previous polynomials to smooth detection
        for j, prev in enumerate(prev_fitted_curves):
            if len(prev) <= 1:
                return None
            if len(prev) > 10:
                prev.pop(0)

            avg_coeffs = np.mean(prev, axis=0)
            avg_coefficients[j] = avg_coeffs
        print('avg_coefficients:', avg_coefficients)

    def project_lane_marking(self):
        # Project the lane marking into the frame
        # Iterate over the list with curves coefficients
        # for curve_coefficients in self.fitted_curves:
        for curve_coefficients in avg_coefficients:
            # Generate y values
            y_values = np.linspace(start=self.correction_shift, stop=self.frame.shape[0], num=100) # Correction shift -> start projection from this part of image

            # Calculate x values with polynomial coefficients
            x_values = np.polyval(curve_coefficients, y_values)

            # Convert x, y to integer
            curve_points = np.column_stack((x_values.astype(int), y_values.astype(int)))

            # Draw the curve on the frame
            cv2.polylines(self.frame, [curve_points], isClosed=False, color=(0, 255, 0), thickness=5)

    """
    def draw_lane_center(self):
        # Draw the lane center line between the curves with indices 1 and 2
        if len(self.fitted_curves) >= 3:
            # Get the fitted curves for indices 1 and 2
            curve1 = self.fitted_curves[1]
            curve2 = self.fitted_curves[2]

            # Choose a y-coordinate where you want to draw the lane center line
            y_center = self.frame.shape[0] // 2

            # Calculate the x-coordinates of the curves at the chosen y-coordinate
            x_curve1 = np.polyval(curve1, y_center) #- self.correction_shift)
            x_curve2 = np.polyval(curve2, y_center) #- self.correction_shift)

            # Calculate the average x-coordinate between the curves
            lane_center_x = (x_curve1 + x_curve2) / 2

            # Draw the lane center line
            cv2.line(self.frame, (int(lane_center_x), y_center), (int(lane_center_x), self.correction_shift),
                     color=(255, 150, 0), thickness=5)

    #def project_lane_centers():
        # Project lane centers on the frame

    """
