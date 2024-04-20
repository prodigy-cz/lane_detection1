# Preprocessing, postprocessing and lane centers calculation
import cv2
import numpy as np
# imports
import torch
import config

class LaneDetector:
    def __init__(self, model_path, temporal_window=20):
        self.model = self.load_lane_detection_model(model_path)
        self.temporal_window = temporal_window
        self.previous_predictions = []

    def load_lane_detection_model(self, model_path):
        # Loads the model
        print("[INFO: ] loading up model...")
        if torch.cuda.is_available():
            model = torch.load(model_path).to(config.DEVICE)
        else:
            model = torch.load(model_path, map_location=torch.device('cpu'))
        return model

    def make_prediction(self, pre_processed_image):
        # Make predictions on preprocessed image and returns NumPy array output

        self.model.eval()

        with torch.no_grad():
            # Inference logic
            image = pre_processed_image  # Inference on ROI
            image = np.transpose(image, (2, 0, 1))  # Transpose to Channel first format (C, H, W)
            image = np.expand_dims(image, 0)  # Extra dim to represent batch size (1, C, H, W), Torch requirement
            image = torch.from_numpy(image).to(config.DEVICE)  # NumPy array to a PyTorch tensor and GPU

            prediction = self.model(image).squeeze()  # Removes singleton dim -> (C, H, W)
            prediction = torch.sigmoid(prediction)  # Applies Sigmoid function -> [0, 1] output

            prediction = prediction.cpu().numpy()  # NumPy array again

            self.previous_predictions.append(prediction)

            return prediction

    def pre_process_frame(self, frame, roi_coordinates):
        # Preprocess frame - resizing, BGR2RGB, normalization
        pre_processed_frame = cv2.resize(frame, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT),
                                         interpolation=cv2.INTER_CUBIC)  # Resizing
        pre_processed_frame = cv2.cvtColor(pre_processed_frame, cv2.COLOR_BGR2RGB) # BGR to RGB
        pre_processed_frame = pre_processed_frame.astype("float32") / 255.0  # Normalization and scaling to range [0, 1]

        # Apply ROI to the copy of the frame and return roi image
        x, y, w, h = roi_coordinates
        roi_image = pre_processed_frame.copy()
        pre_processed_frame = roi_image[y:y+h, :w]

        return pre_processed_frame

    def temporal_avg(self):
        if len(self.previous_predictions) > self.temporal_window:
            self.previous_predictions.pop(0)
        num_predictions = len(self.previous_predictions)
        if num_predictions <= 1:
            return None
        avg_prediction = sum(self.previous_predictions) / num_predictions
        return avg_prediction

    def post_process_prediction(self, prediction, avg_prediction, threshold=config.THRESHOLD):
        # Apply threshold filtering and post_processing
        if avg_prediction is not None:
            binary_mask = (avg_prediction > threshold) * 255
        else:
            binary_mask = (prediction > threshold) * 255

        binary_mask = binary_mask.astype(np.uint8)
        binary_mask_cropped = binary_mask.copy()
        binary_mask = binary_mask_cropped[250:, :]

        # Apply gaussian and median blur
        binary_mask = cv2.GaussianBlur(binary_mask, (5, 5), 0)
        binary_mask = cv2.medianBlur(binary_mask, 5)

        #binary_mask = cv2.resize(binary_mask, (config.INPUT_IMAGE_WIDTH, config.roi_height))
        return binary_mask


#def calculate_lane_centers(binary_mask):
#    # Find contours and calculate lane centers
#    return lane_centers
