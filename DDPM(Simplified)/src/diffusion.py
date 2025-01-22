import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention,CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, 4*embed_dim)
        self.linear2 = nn.Linear(4*embed_dim, 4*embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear2(x)

        #(1,320)->(1,1280)
        return x
    

class SwitchSequential(nn.Sequential):


    def forward(self, x: torch.Tensor, context: torch.Tensor, timestep: torch.Tensor)->torch.Tensor:
        for layer in self:
            if isinstance(layer,UNet_AttentionBlock):
                x = layer(x,context)
            elif isinstance(layer,UNet_ResidualBlock):
                x = layer(x,timestep)
            else:
                x = layer(x)
        return x
    

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module([
            #(batch_size,4,Height/8,Width/8) -> (batch_size,320,Height/8,Width/8)
            SwitchSequential(nn.Conv2d(4,320,kernel_size=3,padding=1)),
            SwitchSequential(UNet_ResidualBlock(320,320),UNet_AttentionBlock(8,40)),
            SwitchSequential(UNet_ResidualBlock(320,320),UNet_AttentionBlock(8,40)),

            #(batch_size,320,Height/8,Width/8) -> (batch_size,320,Height/16,Width/16)
            SwitchSequential(nn.Conv2d(320,320,kernel_size=3,stride=2,padding=1)),
            SwitchSequential(UNet_ResidualBlock(320,640),UNet_AttentionBlock(8,80)),
            SwitchSequential(UNet_ResidualBlock(640,640),UNet_AttentionBlock(8,80)),

            #(batch_size,640,Height/16,Width/16) -> (batch_size,640,Height/32,Width/32)
            SwitchSequential(nn.Conv2d(640,640,kernel_size=3,stride=2,padding=1)),
            SwitchSequential(UNet_ResidualBlock(640,1280),UNet_AttentionBlock(8,160)),
            SwitchSequential(UNet_ResidualBlock(1280,1280),UNet_AttentionBlock(8,160)),

            #(batch_size,1280,Height/32,Width/32) -> (batch_size,1280,Height/64,Width/64)
            SwitchSequential(nn.Conv2d(1280,1280,kernel_size=3,stride=2,padding=1)),
            SwitchSequential(UNet_ResidualBlock(1280,1280)),
            SwitchSequential(UNet_ResidualBlock(1280,1280))
        ])

        self.bottleneck = SwitchSequential(
            UNet_ResidualBlock(1280,1280),
            UNet_AttentionBlock(8,160),
            UNet_ResidualBlock(1280,1280)
        )

        self.decoder = nn.Module([
            #(batch_size,2560,Height/64,Width/64) -> (batch_size,1280,Height/64,Width/64)
            SwitchSequential(UNet_ResidualBlock(2560,1280)),
        ])

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = UNet_Outputlayer(320,4)

        def forward(self, latent: torch.Tensor, context: torch.Tensor, timestep:torch.Tensor):
            #latent(batch_size,4,Height/8,Width/8)
            #conext(batch_size,seq_len,embed_dim)
            #timestep(1,320)

            #(1,320)->(1,1280)
            timestep = self.time_embedding(timestep)

            #(batch_size,4,Height/8,Width/8)->(batch_size,320,Height/8,Width/8)
            output = self.unet(latent, context, timestep)

            #(batch_size,320,Height/8,Width/8)->(batch_size,4,Height/8,Width/8)
            output = self.final(output)

            return output