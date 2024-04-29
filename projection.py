# Visualization of lane marking and lane center

# Imports
import cv2
import numpy as np
import config

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
            # Print the center of mass
            #print("Center of mass (cx, cy):", cen_of_mass)

            # Draw a circle at the center of mass
            cv2.circle(frame, (cx, cy), 5, (5, 94, 255), -1)


            return cen_of_mass


prev_fitted_curves = [[], [], []]
avg_coefficients = [[], [], []]

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

        for contour in self.largest_contours:
            cv2.drawContours(self.frame, self.largest_contours, -1, (0, 0, 255), 2)
    def split_contours(self):
        # Split contours row wise into smaller segments
        num_segments = 100
        segment_height = self.frame.shape[0] // num_segments # height of the segments after split
        contours = self.largest_contours.copy()
        print('\n New Frame:')

        contours_segments = [] # List of segments of all contours
        for contour in contours:
            print('\n New Contour:')
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
        # for contour_segments in contours_segments:
            for segment_contour in contour_segments:
                cv2.drawContours(self.frame, [segment_contour], -1, color=(0, 100, 100), thickness=2)

        return contours_segments

    def filter_contours(self, contours_segments):
        contours_segments = contours_segments
        left_centers = []
        right_centers = []
        for contour_segments in contours_segments:
            centers_of_mass = []
            for contour_segment in contour_segments:
                # Get the center of mass of the segment
                center_of_mass = get_center_of_mass(self.frame, contour_segment)
                if center_of_mass is not None:
                    centers_of_mass.append(center_of_mass)

            # Convert centers_of_mass to a NumPy array
            centers_array = np.array(centers_of_mass)

            # Check if there are any centers of mass
            if centers_array.size > 0:
                # Check the average position of the center of mass
                average_x = np.mean(centers_array[:, 0])
                print('average_x:', average_x)

                # Check if the average center is close to the center of the image
                if (self.frame.shape[1] // 4) <= average_x < (self.frame.shape[1] // 2):
                    left_centers.extend(centers_array)
                    print('left')
                elif (self.frame.shape[1] // 2) <= average_x < (self.frame.shape[1] * 3 // 4):
                    right_centers.extend(centers_array)
                    print('right')
            else:
                print("No valid centers of mass found")
        print('len left_centers: ', len(left_centers))
        print('len right_centers: ', len(right_centers))
        print('left_centers: ', left_centers)
        return left_centers, right_centers


    def compare_frame(self, lane_num):
        # Compare current frame detection with previous number of predictions

        if len(avg_coefficients[lane_num]) < 3:
            return True  # No coefficients to compare with
        else:
            current_coefficients = self.fitted_curves[lane_num]
            # Iterate over coeffs array = over all lane markings
            # a and b values from f(y)=ay²+by+c
            current_a = current_coefficients[0]
            current_b = current_coefficients[1]
            current_c = current_coefficients[2]
            #previous_a = prev_fitted_curves[lane_num][-1][0]
            #previous_b = prev_fitted_curves[lane_num][-1][1]
            previous_a = avg_coefficients[lane_num][0]
            previous_b = avg_coefficients[lane_num][1]
            previous_c = avg_coefficients[lane_num][2]


            # Percentage difference
            percent_a = abs(current_a - previous_a) / abs(previous_a) * 100
            percent_b = abs(current_b - previous_b) / abs(previous_b) * 100
            percent_c = abs(current_c - previous_c) / abs(previous_c) * 100
            #print('current coeffs: ', current_coefficients)
            #print('current a:', current_a, 'prev a', previous_a)
            #print('current b:', current_b, 'prev b', previous_b)
            #print('current c:', current_c, 'prev c', previous_c)
            #print('percent diff: ', percent_a, percent_b, percent_c)
            # Check percentage criteria
            if percent_a <= 50:
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
            print("Warning: left_centers is empty.")
            return

        if right_centers:
            right_centers = np.vstack(right_centers)
            x_right_values = right_centers[:, 0]  # Extracting x values
            y_right_values = right_centers[:, 1]  # Extracting y values
        else:
            print("Warning: right_centers is empty.")
            return

        left_coefficients = np.polyfit(y_left_values, x_left_values, degree)
        right_coefficients = np.polyfit(y_right_values, x_right_values, degree)

        self.fitted_curves = left_coefficients, right_coefficients
        return

    """
            # If the current frame coeffs are similar to previous, add it to the list for averaging
            if self.compare_frame(lane_num=i): # == True or self.compare_frame(lane_num=i) == False:
                # Store fitted curves coefficients into prev_fitted_curves list to use it for averaging and decision-making
                if i == 0:
                    prev_fitted_curves[0].append(coefficients)
                elif i == 1:
                    prev_fitted_curves[1].append(coefficients)
                elif i == 2:
                    prev_fitted_curves[2].append(coefficients)
                else:
                    print('More than 3 lanes detected, i =', i)
        #print('prev_fitted_curves-1: ', prev_fitted_curves[0][-1])
        #print('prev_fitted_curves-2: ', prev_fitted_curves[0][-2])
        print('prev_fitted_curves: ', prev_fitted_curves[0])
        """

    """        
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
        print('avg_coefficients:', avg_coefficients[1])
    """

    def project_lane_marking(self):
        # Project the lane marking into the frame
        # Iterate over the list with curves coefficients
        # for curve_coefficients in avg_coefficients:
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

            #leftmost = tuple(self.largest_contours[self.largest_contours[:, :, 0].argmin()][0])
            #rightmost = tuple(self.largest_contours[self.largest_contours[:, :, 0].argmax()][0])
            #topmost = tuple(self.largest_contours[self.largest_contours[:, :, 1].argmin()][0])
            #bottommost = tuple(self.largest_contours[self.largest_contours[:, :, 1].argmax()][0])

            # Draw points on the frame to visualize them
            cv2.circle(self.frame, leftmost, 5, (255, 0, 0), -1)
            cv2.circle(self.frame, rightmost, 5, (255, 0, 0), -1)
            cv2.circle(self.frame, topmost, 5, (255, 0, 0), -1)
            cv2.circle(self.frame, bottommost, 5, (255, 0, 0), -1)

        #for contour in self.largest_contours:
            #print("Number of points in contour:", len(contour))

            if len(contour) >= 5:

                # Fit an ellipse to the contour
                ellipse = cv2.fitEllipse(contour)

                # Draw the ellipse on the frame
                #cv2.ellipse(self.frame, ellipse, (0, 0, 255), 2)

                # Get the angle of the major axis relative to the horizontal axis
                angle = ellipse[2]

                # Print the angle
                #print("Angle of major axis with horizontal axis:", angle)

        #for contour in self.largest_contours:
            # Check if the contour has at least 5 points
            if len(contour) >= 5:
                # Calculate moments of the contour
                get_center_of_mass(self.frame, contour)

        # Calculate the x-coordinate for the vertical line (middle of the image)
        middle_x = self.frame.shape[1] // 2

        # Draw the vertical middle line on the frame
        cv2.line(self.frame, (middle_x, 0), (middle_x, self.frame.shape[0]), [0, 255, 255], thickness=2)

        # first dividing line
        cv2.line(self.frame, ((self.frame.shape[1] // 4), 0), ((self.frame.shape[1] // 4), self.frame.shape[0]), [0, 255, 255], thickness=2)
        return
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