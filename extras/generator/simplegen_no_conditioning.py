import torch.nn as nn
from models.helper_functions import *


""" Simplified Resnet Generator without conditioning """

# Basic Block for Generator
class ResnetBlock(nn.Module):
    """
    A basic ResNet block for a generator model that applies a series of transformations 
    including convolution, normalization, activation, and optionally dropout, followed 
    by another set of convolution and normalization to the input features, which helps
    in preserving the input information through an identity shortcut connection.
    """
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        # Configure padding type based on the specified option
        padding = get_pad_layer(padding_type)(1) if padding_type != 'zero' else 0
        layers = [
            padding,
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            norm_layer(dim),
            nn.ReLU(True),
            nn.Dropout(0.5) if use_dropout else nn.Identity(),
            padding,
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            norm_layer(dim)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        # Apply the convolution block to the input and add the original input to the output
        return x + self.conv_block(x)  # This addition forms the residual connection


# Generator
class ResnetGenerator(nn.Module):
    """
    ResNet-based generator that constructs an image from an initial random noise vector.
    It uses multiple ResNet blocks and upsampling layers to gradually increase the spatial
    dimensions and details, aiming to produce high-quality synthetic images.
    """
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        # Initial convolutional layer with padding and ReLU activation
        model = [get_pad_layer('reflect')(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        
        # Downsampling part to reduce spatial dimension while increasing the number of feature maps
        for i in range(2):
            mult = 2 ** i
            model.extend([
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)])
        
        # Adding multiple ResNet blocks for deep feature transformation
        mult = 2 ** 2
        for i in range(n_blocks):
            model.append(ResnetBlock(ngf * mult, 'reflect', norm_layer, use_dropout))
        
        # Upsampling part to restore the original spatial dimensions of the image
        for i in range(2, 0, -1):
            mult = 2 ** i
            model.extend([
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf * mult // 2),
                nn.ReLU(True)])
        
        # Final convolutional layer that outputs the generated image with tanh activation
        model.extend([
            get_pad_layer('reflect')(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()])
        self.model = nn.Sequential(*model)

    def forward(self, input):
        # Process the input through the model to generate the output image
        return self.model(input)
