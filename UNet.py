"""
@file UNet.py
@brief This file contains the code for the UNet architecture.
@author Created by C. Factor on 01/03/2024 based on the UNet from practical 3
"""

import torch
import torch.nn as nn


# Simple UNet architecture - created in Class
class UNet(nn.Module):
    """
    @class UNet
    @brief This class defines the UNet architecture.
    It is a simple UNet architecture with 3 encoder layers and 3 decoder layers.
    """

    def __init__(self, in_channels, out_channels):
        """
        @brief Constructor for the UNet class
        @param in_channels The number of input channels
        @param out_channels The number of output channels
        """
        super().__init__()
        # Encoder part
        self.conv1 = self.conv_block(in_channels, 16, 3, 1, 1)
        self.maxpool1 = self.maxpool_block(2, 2, 0)
        self.conv2 = self.conv_block(16, 32, 3, 1, 1)
        self.maxpool2 = self.maxpool_block(2, 2, 0)
        self.conv3 = self.conv_block(32, 64, 3, 1, 1)
        self.maxpool3 = self.maxpool_block(2, 2, 0)

        self.middle = self.conv_block(64, 128, 3, 1, 1)

        self.upsample3 = self.transposed_block(128, 64, 3, 2, 1, 1)
        # Adding the concatenation to the upconv3
        self.upconv3 = self.conv_block(128, 64, 3, 1, 1)
        self.upsample2 = self.transposed_block(64, 32, 3, 2, 1, 1)
        self.upconv2 = self.conv_block(64, 32, 3, 1, 1)
        self.upsample1 = self.transposed_block(32, 16, 3, 2, 1, 1)
        self.upconv1 = self.conv_block(32, 16, 3, 1, 1)

        self.final = self.final_layer(16, 1, 1, 1, 0)

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        @brief This function creates a convolutional block with two convolutional layers
        and batch normalization.
        @param in_channels The number of input channels
        @param out_channels The number of output channels
        @param kernel_size The size of the kernel
        @param stride The stride of the convolution
        @param padding The padding of the convolution
        @return convolution The convolutional block"""
        convolution = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        return convolution

    # Not doing anything on the number of channels - so doesn't need in/out_channels
    def maxpool_block(self, kernel_size, stride, padding):
        """
        @brief This function creates a maxpool block with a maxpool layer and dropout.
        @param kernel_size The size of the kernel
        @param stride The stride of the maxpool
        @param padding The padding of the maxpool
        @return maxpool The maxpool block"""
        # Only need nn.Sequential for multiple operations in a block
        maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Dropout2d(0.5),
        )
        return maxpool

    def transposed_block(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding
    ):
        """
        @brief This function creates a transposed convolution
        @param in_channels The number of input channels
        @param out_channels The number of output channels
        @param kernel_size The size of the kernel
        @param stride The stride of the convolution
        @param padding The padding of the convolution
        @param output_padding The output padding of the convolution
        @return transposed The transposed convolution
        """
        transposed = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        return transposed

    def final_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        @brief This function creates the final convolutional layer.
        @param in_channels The number of input channels
        @param out_channels The number of output channels
        @param kernel_size The size of the kernel
        @param stride The stride of the convolution
        @param padding The padding of the convolution
        @return final The final convolutional layer"""
        final = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        return final

    def forward(self, x):
        """
        @brief This function defines the forward pass of the UNet architecture.
        @param x The input tensor
        @return final_layer The output tensor"""
        # downsampling part
        conv1 = self.conv1(x)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        # middle part
        middle = self.middle(maxpool3)

        # upsampling part
        upsample3 = self.upsample3(middle)
        # Add the concats
        upconv3 = self.upconv3(torch.cat([upsample3, conv3], 1))
        upsample2 = self.upsample2(upconv3)
        upconv2 = self.upconv2(torch.cat([upsample2, conv2], 1))
        upsample1 = self.upsample1(upconv2)
        upconv1 = self.upconv1(torch.cat([upsample1, conv1], 1))

        final_layer = self.final(upconv1)

        return final_layer
