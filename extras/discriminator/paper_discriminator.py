import torch
import torch.nn as nn
from models.helper_functions import * 

# Paper's implementation of the discriminator

class NLayerDiscriminator_ncsn(nn.Module):
    """Defines a PatchGAN discriminator with conditional input"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator_ncsn, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        self.no_antialias = no_antialias
        self.model_main = nn.ModuleList()
        
        # First layer
        if no_antialias:
            self.model_main.append(
                nn.Sequential(
                    nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, True)
                )
            )
        else:
            self.model_main.append(
                ConvBlock_cond(input_nc, ndf, 4 * ndf, kernel_size=4, stride=1, padding=1, use_bias=use_bias)
            )

        # Intermediate layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if no_antialias:
                self.model_main.append(
                    nn.Sequential(
                        nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                        norm_layer(ndf * nf_mult),
                        nn.LeakyReLU(0.2, True)
                    )
                )
            else:
                self.model_main.append(
                    ConvBlock_cond(ndf * nf_mult_prev, ndf * nf_mult, 4 * ndf, kernel_size=4, stride=1, padding=1, use_bias=use_bias, norm_layer=norm_layer)
                )

        # Last layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.model_main.append(
            ConvBlock_cond(ndf * nf_mult_prev, ndf * nf_mult, 4 * ndf, kernel_size=4, stride=1, padding=1, use_bias=use_bias, norm_layer=norm_layer, downsample=False)
        )
        
        self.final_conv = nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        self.t_embed = TimestepEmbedding(
            embedding_dim=1,
            hidden_dim=4 * ndf,
            output_dim=4 * ndf,
            act=nn.LeakyReLU(0.2)
        )

    def forward(self, input, t_emb, input2=None):
        """Forward pass of the discriminator"""
        t_emb = t_emb.float()  # Ensure t_emb is of type float
        t_emb = self.t_embed(t_emb)
        if input2 is not None:
            out = torch.cat([input, input2], dim=1)
        else:
            out = input
        
        for layer in self.model_main:
            if isinstance(layer, ConvBlock_cond):
                out = layer(out, t_emb)
            else:
                out = layer(out)
        
        return self.final_conv(out)
