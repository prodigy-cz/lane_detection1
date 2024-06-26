# Imports
import config
from torch.nn import ConvTranspose2d, Conv2d, MaxPool2d, Module, ModuleList, ReLU
from torch.nn import functional as F
from torchvision.transforms import CenterCrop
import torch


class Block(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # convolution and ReLU layers
        self.conv1 = Conv2d(in_channels, out_channels, 3)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, 3)

    def forward(self, x):
        # apply convolution -> relu -> convolution block t the inputs and return it
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        super().__init__()

        # Store the encoder blocks and maxpooling layer
        # initializing the list of blocks (self.encoder_blocks) with ModuleList
        # ech block takes input channels and doubles it in the output feature map
        self.encoder_blocks = ModuleList(
          [Block(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)])

        # initiation of max pooling layer - reducing the spatial dimension (height and width) by a factor of 2
        self.pool = MaxPool2d(2)

    # forward function takes an image x input
    def forward(self, x):
        # Initialize an empty list to store the intermediate outputs from the blocks of the encoder - this will enable the model to pass encoder outputs to decoder where are processed with decoder feature maps
        block_outputs = []

        # loop through the encoder blocks
        for block in self.encoder_blocks:
          # Pass the inputs through the current encoder block, store the outputs, and then apply maxpooling on the output
          x = block(x)
          block_outputs.append(x)
          # max poool operation on the block output
          x = self.pool(x)

        # return the list containing the intermediate encoder outputs
        return block_outputs


class Decoder(Module):
    # take channels tuple as input (dimensions of channels) - decoder -> decreasing dimension with factor of 2
    def __init__(self, channels=(64,32,16)):
        super().__init__()

        # initialize the number of channels, upsampling blocks and decoder blocks
        self.channels = channels

        # list of upsampling blocks that use ConvTranspose2d layer to upsample the spatial dimension
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
                for i in range(len(channels) - 1)])

        # decoder blocks initialization
        self.dec_blocks = ModuleList(
            [Block(channels[i], channels[i + 1])
                for i in range(len(channels) - 1)])

    # Take the feature map x and the list of intermediate outputs from encoder (encoder_features) as an input
    def forward(self, x, encoder_features):

        # Loop through the number of channels
        for i in range(len(self.channels) - 1):
            # Pass the inputs through the i-th upsampler blocks
            x = self.upconvs[i](x)

            # Crop the current features from the encoder blocks - to ensure that x and encoder_features[i] match
            encoder_feature = self.crop(encoder_features[i], x)

            # Concatenate cropped upsampled feature map encoder_feature with current upsampled feature map x
            x = torch.cat([x, encoder_feature], dim=1)

            # Pass the concatenated output through the current decoder block
            x = self.dec_blocks[i](x)

        # return the final decoder output
        return x

    # Crop function to take an intermediate feature map from encoder (encoder_features) and x
    # (feature map output from the decoder) and crops the former to the dimension of the latter
    def crop(self, encoder_features, x):

        # Grab the dimensions of the inputs, and crop the encoder features to match the dimensions
        (_, _, H, W) = x.shape
        encoder_features = CenterCrop([H, W])(encoder_features)

        # return the cropped features
        return encoder_features


class UNet(Module):
    # encoder_channels - increase of channel dimension as the input passes through the encoder
    # decoder_channels - decrease of channel dimension as the input passes through decoder
    # channel_classes_number - number of segmen. class. - classifying each pixel - corresponds to the number of channel
    # retain_orig_dimension - indicate whether original dimension will remain the same
    # output_size - spatial dimension of the output
    def __init__(self, encoder_channels=(3, 16, 32, 64), # starts with 3 channels - RGB
                 decoder_channels=(64, 32, 16),
                 channel_classes_number=1, retain_orig_dimension=True,  # binary classification -> 1 channel + threshold
                 output_size=(config.roi_height,  config.INPUT_IMAGE_WIDTH)):  # same dim as input image
        super().__init__()

        # Initialize the encoder and decoder
        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_channels)

        # Initialize the regression head and store the class variables
        self.head = Conv2d(decoder_channels[-1], channel_classes_number, 1)
        self.retain_orig_dimension = retain_orig_dimension
        self.output_size = output_size

    def forward(self, x):
        # Grab the features from the encoder
        encoder_features = self.encoder(x)

        # Pass the encoder features through decoder - making sure dimensions are suited for concatenation
        # Utilize the encoder feature maps from the last block to the first (on the decoder side)
        decoder_features = self.decoder(encoder_features[::-1][0],
                                        encoder_features[::-1][1:])

        # Pass the decoder features through the regression head to obtain the segmentation mask
        segmentation_map = self.head(decoder_features)

        # Check to see if retaining the original output dimensions -> resize the output to match them
        if self.retain_orig_dimension:
            segmentation_map = F.interpolate(segmentation_map, self.output_size)

        # Return the final segmentation map
        return segmentation_map
