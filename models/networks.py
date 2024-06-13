import torch
import torch.nn as nn
import torch.nn.functional as F
from models.helper_functions import *
import functools

""" Here we define the networks needed for our Diffusion Model. Below, we provide subnetworks' instances, used to run our diffusion model """

# Conditional ResNet Generator with Adaptive Conditioning to generate images at different time steps
class ResnetGenerator_cond(nn.Module):
    # Initialization of the conditional ResNet generator
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, n_blocks=9):
        super(ResnetGenerator_cond, self).__init__()
        
        # Ensuring the number of blocks is non-negative
        assert(n_blocks >= 0)
        # Determine if bias is needed based on the type of normalization layer
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        # Initial convolution module to process input image
        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),  # Padding before initial convolution
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),  # Initial convolution to transform input channel
            norm_layer(ngf),  # Normalization layer
            nn.ReLU(inplace=False)  # Activation function
        )
        
        self.ngf = ngf  # Number of generator filters
        
        # List of residual blocks with conditional inputs
        self.model_res = nn.ModuleList([])
        # Downsampling part of the model
        self.model_downsample = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4),
            nn.ReLU(inplace=False)
        )
        
        # Add multiple ResnetBlockCond instances for intermediate processing
        for i in range(n_blocks):
            self.model_res += [ResnetBlockCond(ngf * 4, norm_layer, temb_dim=4 * ngf, z_dim=4 * ngf)]
       
        # Upsampling part of the model to restore original image size
        self.model_upsample = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()  # Output activation to ensure output values are between -1 and 1
        )
        
        # Define a transformation for the latent vector z
        mapping_layers = [PixelNorm(),
                          nn.Linear(self.ngf * 4, self.ngf * 4),
                          nn.LeakyReLU(0.2)]
        self.z_transform = nn.Sequential(*mapping_layers)
        
        # Time embedding layers
        modules_emb = [nn.Linear(self.ngf, self.ngf * 4)]
        nn.init.zeros_(modules_emb[-1].bias)  # Initialize the bias to zero for stability
        modules_emb += [nn.LeakyReLU(0.2), nn.Linear(self.ngf * 4, self.ngf * 4)]
        nn.init.zeros_(modules_emb[-1].bias)  # Again, initialize the bias to zero
        modules_emb += [nn.LeakyReLU(0.2)]
        self.time_embed = nn.Sequential(*modules_emb)
                                
    # Define the forward pass with conditional inputs time_cond and z
    def forward(self, x, time_cond, z):
        z_embed = self.z_transform(z)  # Transform z before feeding it to the ResNet blocks
        temb = get_timestep_embedding(time_cond, self.ngf)  # Embedding the time steps
        time_embed = self.time_embed(temb)  # Applying the time embedding
        out = self.model(x)  # Initial processing of input
        out = self.model_downsample(out)  # Apply downsampling
        for layer in self.model_res:  # Apply each ResNet block sequentially
            out = layer(out, time_embed, z_embed)
        out = self.model_upsample(out)  # Final upsampling and output layer
        return out
    

# Discriminator with Conditional Convolution Blocks for processing input images
class NLayerDiscriminator_ncsn_new(nn.Module):
    """Discriminator that uses conditional convolution blocks to process input images."""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Initialize the discriminator with conditional convolution blocks."""
        super(NLayerDiscriminator_ncsn_new, self).__init__()
        # Determine if bias should be used based on the type of normalization layer
        use_bias = norm_layer == nn.InstanceNorm2d

        # List of modules that make up the main discriminator model
        self.model_main = nn.ModuleList()
        
        # First convolution block that processes the initial input layer
        self.model_main.append(
            ConvBlock_cond(input_nc, ndf, 4 * ndf, kernel_size=4, stride=1, padding=1, use_bias=use_bias))

        # Dynamically add intermediate convolution blocks with increasing feature depth
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.model_main.append(
                ConvBlock_cond(ndf * nf_mult_prev, ndf * nf_mult, 4 * ndf, kernel_size=4, stride=1, padding=1, use_bias=use_bias, norm_layer=norm_layer)
            )

        # Add the last convolution block without downsampling to maintain spatial dimensions
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.model_main.append(
            ConvBlock_cond(ndf * nf_mult_prev, ndf * nf_mult, 4 * ndf, kernel_size=4, stride=1, padding=1, use_bias=use_bias, norm_layer=norm_layer, downsample=False)
        )
        
        # Final convolution layer that outputs a single channel for discrimination
        self.final_conv = nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        # Time embedding layer that prepares the timestep embedding for integration into convolution blocks
        self.t_embed = TimestepEmbedding(
            embedding_dim=1,
            hidden_dim=4 * ndf,
            output_dim=4 * ndf,
            act=nn.LeakyReLU(0.2)
        )

    def forward(self, input, t_emb, input2=None):
        """Forward pass through the discriminator with optional dual inputs and timestep embedding."""
        t_emb = t_emb.float()  # Convert timestep embedding to float for processing
        t_emb = self.t_embed(t_emb)  # Apply embedding transformation
        # If a second input is provided, concatenate it with the first input
        out = torch.cat([input, input2], dim=1) if input2 is not None else input
        
        # Process each convolution block with the current output and timestep embedding
        for layer in self.model_main:
            out = layer(out, t_emb) if isinstance(layer, ConvBlock_cond) else layer(out)
        
        return self.final_conv(out)  # Apply the final convolution layer to produce the discriminator's output


# PatchSampleF aims to extract and processes patches from the feature maps generated by other parts of the model. 
class PatchSampleF(nn.Module):
    """ PatchsampleF is a class designed to sample and normalize patches from feature maps. 
    It can optionally use multi-layer perceptrons (MLPs) for further processing. During the forward pass, 
    it selects a specified number of patches from each feature map, optionally processes them through MLPs, 
    and applies L2 normalization. It also supports specifying custom patch indices or randomly selecting them if not provided """
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)  
        self.use_mlp = use_mlp
        self.nc = nc
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        """ Creation of MLP for further processing of features and patches """
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[0]
            mlp = nn.Sequential(
                nn.Linear(input_nc, self.nc),
                nn.LeakyReLU(0.2),
                nn.Linear(self.nc, self.nc)
            )
            mlp.cuda()
            setattr(self, f'mlp_{mlp_id}', mlp)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        """ The forward method returns the sampled (and possibly processed) patches and their indices. 
        If no patches are sampled, it reshapes the feature maps back to their original dimensions """
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)

        return_feats = []
        return_ids = []

        for feat_id, feat in enumerate(feats):
            # Add batch dimension if missing
            if len(feat.shape) == 3:
                feat = feat.unsqueeze(0)

            B, C, H, W = feat.shape
            feat_reshape = feat.permute(0, 2, 3, 1).reshape(B, -1, C)  # Reshape to [B, H*W, C]

            if num_patches > 0:
                if patch_ids is not None and len(patch_ids) > feat_id:
                    current_patch_ids = patch_ids[feat_id]
                else:
                    # Generate random patch indices if none provided
                    current_patch_ids = [torch.randperm(feat_reshape.shape[1])[:num_patches].to(feat.device) for _ in range(B)]
                current_patch_ids = [torch.tensor(pid, dtype=torch.long, device=feat.device) for pid in current_patch_ids]
                # Sampling patches
                x_sample = torch.cat([feat_reshape[b, pid, :] for b, pid in enumerate(current_patch_ids)], dim=0)
                return_ids.append(current_patch_ids)
            else:
                x_sample = feat_reshape.reshape(-1, C)
                current_patch_ids = [torch.tensor([], dtype=torch.long, device=feat.device) for _ in range(B)]
                return_ids.append(current_patch_ids)

            if self.use_mlp:
                mlp = getattr(self, f'mlp_{feat_id}')
                x_sample = mlp(x_sample)

            x_sample = self.l2norm(x_sample)

            return_feats.append(x_sample)

        # Since we add patches for each batch, we must handle the concatenation properly
        if num_patches == 0:
            return_feats = [f.view(B, H, W, -1).permute(0, 3, 1, 2) for f in return_feats]

        return return_feats, return_ids 
    
    
### Defining Used Subnetworks 
gen = ResnetGenerator_cond(input_nc=3, output_nc=3, ngf=64, n_blocks=9, norm_layer=nn.InstanceNorm2d).to(device) 
disc = NLayerDiscriminator_ncsn_new(input_nc=3, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d).to(device) 
netE = NLayerDiscriminator_ncsn_new(input_nc=3*4, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d).to(device)
netF = PatchSampleF(use_mlp=True, init_type='normal', init_gain=0.02, nc=256).to(device)