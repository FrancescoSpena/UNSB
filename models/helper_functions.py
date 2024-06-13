import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import math

""" Here we define helper functions that we will use throughout the project """

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define padding layer based on type for use in convolutional layers
def get_pad_layer(pad_type):
    if pad_type in ['reflect', 'refl']:
        return nn.ReflectionPad2d
    elif pad_type in ['replicate', 'repl']:
        return nn.ReplicationPad2d
    elif pad_type == 'zero':
        return nn.ZeroPad2d
    else:
        raise NotImplementedError(f'Padding type {pad_type} not recognized')

# Module to normalize pixel values in images for stable training
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)
        
# Generate embeddings for timesteps in models that incorporate time dynamics
def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')
    return emb
                                  
# Class to embed timestep information into network inputs
class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.ReLU()):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)  # First layer: embedding to hidden dimension
        self.act = act  # Activation function
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Second layer: hidden dimension to output

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t = t.float()  # Ensure t is of type float
        t = self.fc1(t)
        t = self.act(t)
        t = self.fc2(t)
        return t
    
# Initialize network weights using a specific strategy for better training performance
def init_weights(net, init_gain=0.02, debug=False):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)  # Print class name during debugging
            init_gain = 0.02
            init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('LayerNorm') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>

# Set up network for use, optionally initialize weights, and set GPU configuration if available
def init_net(net, init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    if initialize_weights:
        init_weights(net, init_gain=init_gain, debug=debug)
    return net

# Module to normalize tensors based on a power rule, useful for data and feature normalization
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
    
# Function to convert normalized image data back to standard image format
def denormalize(tensor):
    return tensor.mul(0.5).add(0.5)  # Converts from [-1, 1] to [0, 1]

# Visualize a batch of images using a grid layout
def visualize_images(images, title="Generated Images"):
    images = images.cpu()  # Move images to CPU for visualization
    images = denormalize(images)  # Denormalize images to bring them to displayable format
    grid = vutils.make_grid(images, padding=2, normalize=True)  # Create a grid of images
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(grid, (1, 2, 0)))  # Display images
    plt.show()

# Generator's Helper Functions
class AdaptiveLayer(nn.Module):
    # Initializer for the adaptive layer which applies learned affine transformations.
    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.style = nn.Linear(style_dim, in_channel * 2)  # Creates a linear transformation for style codes
        # Initialize the affine transform parameters gamma to 1 (scale) and beta to 0 (shift)
        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    # Forward pass which applies the affine transformation to the input features
    def forward(self, input, style):
        gamma, beta = self.style(style).chunk(2, 1)  # Split style into gamma and beta components
        gamma, beta = gamma.unsqueeze(2).unsqueeze(3), beta.unsqueeze(2).unsqueeze(3)  # Adjust dimensions for feature map
        return gamma * input + beta  # Apply the affine transformation
    
class ResnetBlockCond(nn.Module):
    # Initializer for a conditional ResNet block which integrates time and style-based conditioning.
    def __init__(self, dim, norm_layer, temb_dim, z_dim):
        super(ResnetBlockCond, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),  # Padding for maintaining spatial dimensions after convolution
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),  # Standard convolution layer
            norm_layer(dim),  # Normalization layer
            nn.ReLU(inplace=False)  # Activation function
        ) 
        
        self.adaptive = AdaptiveLayer(dim, z_dim)  # Style-based adaptive layer
        
        self.conv_fin = nn.Sequential(
            nn.ReflectionPad2d(1),  # Additional padding for final convolution
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),  # Final convolution layer
            norm_layer(dim)  # Final normalization layer
        )
        self.dense_time = nn.Linear(temb_dim, dim)  # Linear layer for transforming time conditioning
        nn.init.zeros_(self.dense_time.bias)  # Initialize the bias for the time conditioning layer to zero
        self.style = nn.Linear(z_dim, dim * 2)  # Style transformation similar to the adaptive layer
        # Initialize gamma to 1 and beta to 0 for the style conditioning
        self.style.bias.data[:dim] = 1
        self.style.bias.data[dim:] = 0

    # Forward pass through the ResNet block with conditional inputs
    def forward(self, x, time_cond, z):
        time_input = self.dense_time(time_cond)  # Apply linear transformation to the time conditioning
        out = self.conv_block(x)  # Pass input through the convolutional block
        out = out + time_input[:, :, None, None]  # Add time conditioning to the features
        out = self.adaptive(out, z)  # Apply adaptive styling
        out = self.conv_fin(out)  # Final convolution to refine features
        out = x + out  # Add skip connections for better gradient flow
        return out
    
    
# Discriminator's Helper Function
class ConvBlock_cond(nn.Module):
    """Conditional convolution block with embedding integration for discriminator."""
    def __init__(self, in_channels, out_channels, embedding_dim, kernel_size=3, stride=1, padding=1, use_bias=True, norm_layer=nn.BatchNorm2d, downsample=True):
        super(ConvBlock_cond, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=use_bias) # Standard convolution layer
        self.norm = norm_layer(out_channels) # Normalization layer specified by norm_layer argument
        self.act = nn.LeakyReLU(0.2, inplace=True) # Activation function set to LeakyReLU for stable gradients
        self.downsample = downsample # Option to downsample the feature map for reducing spatial dimensions
        self.dense = nn.Linear(embedding_dim, out_channels) # Linear layer to transform the embedding dimension to match output channels

    def forward(self, x, t_emb):
        out = self.conv(x) # Apply convolution to the input
        out = out + self.dense(t_emb)[..., None, None] # Add transformed timestep embedding to the convolution output
        out = self.norm(out) # Normalize the output
        out = self.act(out) # Apply the activation function
        # Conditionally apply downsampling
        if self.downsample:
            out = nn.functional.avg_pool2d(out, kernel_size=2, stride=2)
        return out