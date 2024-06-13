import torch
import torch.nn as nn
from models.helper_functions import * 

"""" Simpler version of Discriminator """

class D_NLayersMulti(nn.Module):
    """
    This discriminator network is designed for conditional behavior, 
    incorporating multiple convolutional layers to process input images. 
    Each convolutional layer is accompanied by a linear transformation of timestep embeddings, 
    to enhance discrimination based on the provided timestep context.
    """
    # Initialize the discriminator with multiple convolutional layers.
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.layers = nn.ModuleList()  # List to hold all the convolutional layers
        self.timestep_embedding_transforms = nn.ModuleList()  # List to hold transformations for timestep embeddings

        kw = 4  # Kernel width
        padw = 1  # Padding width

        # First layer - basic convolutional layer without normalization but with leaky ReLU activation
        self.layers.append(nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ))
        # Timestep embedding transformation for the first layer
        self.timestep_embedding_transforms.append(nn.Linear(ndf * 4, ndf))

        nf_mult = 1  # Multiplier for number of features in current layer
        nf_mult_prev = 1  # Multiplier for number of features in previous layer
        for n in range(1, n_layers):  # Adding middle layers
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)  # Increase the number of filters with depth, cap at 8 times the ndf
            self.layers.append(nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),  # Normalization layer
                nn.LeakyReLU(0.2, True)  # Activation layer
            ))
            # Timestep embedding transformation for middle layers
            self.timestep_embedding_transforms.append(nn.Linear(ndf * 4, ndf * nf_mult))

        nf_mult_prev = nf_mult  # Setup for the final layer
        nf_mult = min(2 ** n_layers, 8)  # Final multiplier for the number of features
        self.layers.append(nn.Sequential(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),  # Final normalization
            nn.LeakyReLU(0.2, True)  # Final activation
        ))
        # Timestep embedding transformation for the final layer
        self.timestep_embedding_transforms.append(nn.Linear(ndf * 4, ndf * nf_mult))

        self.final_layer = nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)  # Output layer to produce a single output

    # Forward pass through the discriminator network with input image and timestep embeddings
    def forward(self, input, t_emb):
        result = input  # Start with the input image
        for layer, t_emb_transform in zip(self.layers, self.timestep_embedding_transforms):
            result = layer(result)  # Pass result through each convolutional layer
            t_emb_channel_specific = t_emb_transform(t_emb).view(t_emb.size(0), -1, 1, 1)  # Transform timestep embedding for specific layer
            t_emb_channel_specific = t_emb_channel_specific.expand(-1, -1, result.size(2), result.size(3))  # Expand to match feature map dimensions
            result = result + t_emb_channel_specific  # Add timestep embedding to the feature map
        return self.final_layer(result)  # Apply the final convolutional layer to produce the output
