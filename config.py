# import the necessary packages
import torch
import os
import argparse

# random seed number
RANDOM_SEED = 42

# define the test split
TEST_SPLIT = 0.125 #not split but validation/training samples ratio - default by txt file id 0.125

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

#num_workers
NUM_WORKERS = os.cpu_count() if not None else 0

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

# define the input image dimensions
INPUT_IMAGE_WIDTH = 1280
INPUT_IMAGE_HEIGHT = 720

#ROI shape
roi_width = int(INPUT_IMAGE_WIDTH)
roi_height = int(INPUT_IMAGE_HEIGHT * 2 // 3)
roi_start_y = INPUT_IMAGE_HEIGHT - roi_height

# ROI coordinates (x, y, width, height)
roi_coordinates = (0, roi_start_y, roi_width, roi_height)

# Class weight to eliminate class imbalances
class_weight = [52]

# define threshold to filter weak predictions
THRESHOLD = 0.5

# base paths of the datasets
TRAIN_PATH = "/content/train_image_truth_pairs.txt" #"/content/drive/MyDrive/Colab_Notebooks/tvtLane/my_train_list-images.txt"

VALIDATION_PATH = "/content/val_image_truth_pairs.txt" #"/content/drive/MyDrive/Colab_Notebooks/tvtLane/my_val_list-images.txt"

# define path for testing - need to create dedicated testing dataset
# now reusing the validation images to check the functionality
TEST_PATH = "/content/drive/MyDrive/Colab_Notebooks/tvtLane/my_test_list-images.txt"

# plot name
PLOT_NAME = "plot.png"

# define the path to the base output directory
BASE_OUTPUT = "/content/drive/MyDrive/Colab_Notebooks/tvtLane/trained_models/output"

# model name
MODEL_NAME = "unet_lane_detection-CosineAnnealingWarmRestarts_5.pth"

# define the path to the output serialized model, model training plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, MODEL_NAME)
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, PLOT_NAME])

# Parsing arguments for training settings
def parse_arguments():
    # create parser
    parser = argparse.ArgumentParser(description="Set hyperparameters to train the model!")
    # learning rate
    parser.add_argument("--INIT_LR",
                        default=0.001,
                        type=float,
                        help="Learning rate")
    # batch size
    parser.add_argument("--BATCH_SIZE",
                        default=128,
                        type=int,
                        help="Batch size")
    # number of epochs
    parser.add_argument("--NUM_EPOCHS",
                        default=20,
                        type=int,
                        help="Number of epochs")

    # number of samples to train with
    parser.add_argument("--NUM_SAMPLES_TRAIN",
                        default=None,
                        type=int,
                        help="Number of samples to train with")

    # LR Scheduler Type
    parser.add_argument("--SCHEDULER_TYPE",
                        default='StepLR',
                        type=str,
                        help="Scheduler type = Learning Rate scheduler type (StepLR, CosineAnnealingWarmRestarts)")

    # LR Scheduler Step size
    parser.add_argument("--STEP_SIZE",
                        default=2,
                        type=int,
                        help="Step size = num of batches ofter which the LR will change")

    # Gamma LR scheduler coefficient
    parser.add_argument("--GAMMA",
                        default=0.9,
                        type=float,
                        help="Gamma LR Step coefficient")

    # Number of divisions for T_0 calculation
    parser.add_argument("--NUM_DIVISIONS",
                        default=21,
                        type=int,
                        help="NUM_DIVISIONS for T_0")

    args = parser.parse_args()

    #number of samples for validation
    if args.NUM_SAMPLES_TRAIN is not None:
        args.NUM_SAMPLES_TEST = int(args.NUM_SAMPLES_TRAIN * TEST_SPLIT) # must be int
    else:
        args.NUM_SAMPLES_TEST = None
    return args


