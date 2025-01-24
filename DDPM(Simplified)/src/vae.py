import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1,channels)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #x: (Batch_size,Features,Height,Width)
        residue = x

        n,c,h,w = x.shape

        x = self.groupnorm(x)

        #(Batch_size,Features,Height,Width) -> (Batch_size,Features,Height*Width)
        x = x.view(n,c,h*w)

        #(Batch_size,Features,Height*Width) -> (Batch_size,Height*Width,Features)
        x = x.transpose(-1,-2)

        #(Batch_size,Height*Width,Features) -> (Batch_size,Height*Width,Features)
        x = self.attention(x)

        #(Batch_size,Height*Width,Features) -> (Batch_size,Features,Height*Width)
        x = x.transpose(-1,-2)

        #(Batch_size,Features,Height*Width) -> (Batch_size,Features,Height,Width)
        x = x.view(n,c,h,w)

        x += residue
        return x


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels!= out_channels:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1,padding=0)
        else:
            self.residual_layer = nn.Identity

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #x: (Batch_size,In_Channels,Height,Width)
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residue)


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            #(Batch_size,Channel,Height,Width) -> (Batch_size,128,Height,Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            #(Batch_size,128,Height,Width) -> (Batch_size,128,Height,Width)
            VAE_ResidualBlock(128,128),

            #(Batch_size,128,Height,Width) -> (Batch_size,128,Height,Width)
            VAE_ResidualBlock(128,128),

            #(Batch_size,128,Height,Width) -> (Batch_size,128,Height/2,Width/2)
            nn.Conv2d(128, 128, kernel_size=3,stride=2, padding=0),

            #(Batch_size,128,Height/2,Width/2) -> (Batch_size,256,Height/2,Width/2)
            VAE_ResidualBlock(128,256),

            #(Batch_size,256,Height/2,Width/2) -> (Batch_size,256,Height/2,Width/2)
            VAE_ResidualBlock(256,256),

            #(Batch_size,256,Height/2,Width/2) -> (Batch_size,256,Height/4,Width/4)
            nn.Conv2d(256, 256, kernel_size=3,stride=2, padding=0),

            #(Batch_size,256,Height/4,Width/4) -> (Batch_size,512,Height/4,Width/4)
            VAE_ResidualBlock(256,512),

            #(Batch_size,512,Height/4,Width/4) -> (Batch_size,512,Height/4,Width/4)
            VAE_ResidualBlock(512,512),

            #(Batch_size,512,Height/4,Width/4) -> (Batch_size,512,Height/8,Width/8)
            nn.Conv2d(512, 512, kernel_size=3,stride=2, padding=0),

            #(Batch_size,512,Height/8,Width/8) -> (Batch_size,512,Height/8,Width/8)
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),

            #(Batch_size,512,Height/8,Width/8) -> (Batch_size,512,Height/8,Width/8)
            VAE_AttentionBlock(512),

            #(Batch_size,512,Height/8,Width/8) -> (Batch_size,512,Height/8,Width/8)
            VAE_ResidualBlock(512,512),

            #(Batch_size,512,Height/8,Width/8) -> (Batch_size,512,Height/8,Width/8)
            nn.GroupNorm(32, 512),

            nn.SiLU(),
            #(Batch_size,512,Height/8,Width/8) -> (Batch_size,8,Height/8,Width/8)
            nn.Conv2d(512, 8, kernel_size=3,padding=1),

            #(Batch_size,8,Height/8,Width/8) -> (Batch_size,8,Height/8,Width/8)
            nn.Conv2d(8, 8, kernel_size=1,padding=0),
        )

    def forward(self, x:torch.Tensor,noise:torch.Tensor) -> torch.Tensor:
        #x: (Batch_size,Channel,Height,Width)
        #noise: (Batch_size,OutputChannel,Height/8,Width/8)

        for module in self:
            if getattr(module, 'stride', None) == (2,2):
                #(Padding_Left,Padding_Right,Padding_Top,Padding_Bottom)
                x = F.pad(x, (0,1,0,1))
            x = module(x)

        #(Batch_size,8,Height/8,Width/8) -> (Batch_size,4,Height/8,Width/8) * 2
        mean,log_variance = torch.chunk(x,2,dim=1)

        #(Batch_size,4,Height/8,Width/8) -> (Batch_size,4,Height/8,Width/8)
        log_variance = torch.clamp(log_variance, -30, 20)

        #(Batch_size,4,Height/8,Width/8) -> (Batch_size,4,Height/8,Width/8)
        variance = torch.exp(log_variance)

        #(Batch_size,4,Height/8,Width/8) -> (Batch_size,4,Height/8,Width/8)
        std = torch.sqrt(variance)

        #Z = N(0,1) -> N(mean,variance) = X
        #X = Z * std + mean
        x= mean + std * noise

        #scale the output
        x *= 0.18215

        return x
    



class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4,4,kernel_size=1,padding=0),

            nn.Conv2d(4,512,kernel_size=3,padding=1),

            VAE_ResidualBlock(512,512),

            VAE_AttentionBlock(512),
            
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),

            #(Batch_size,512,Height/8,Width/8)
            VAE_ResidualBlock(512,512),

            #(Batch_size,512,Height/8,Width/8) -> (Batch_size,512,Height/4,Width/4)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),

            #(Batch_size,512,Height/4,Width/4) -> (Batch_size,512,Height/2,Width/2)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            VAE_ResidualBlock(512,256),
            VAE_ResidualBlock(256,256),
            VAE_ResidualBlock(256,256),

            #(Batch_size,256,Height/2,Width/2) -> (Batch_size,256,Height,Width)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            VAE_ResidualBlock(256,128),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),

            nn.GroupNorm(32, 128),
            nn.SiLU(),

            #(Batch_size,128,Height,Width) -> images:(Batch_size,3,Height,Width)
            nn.Conv2d(128,3,kernel_size=3,padding=1),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #x: (Batch_size,4,Height/8,Width/8)
        x /= 0.18215

        for module in self:
            x = module(x)

        #(Batch_size,3,Height,Width)
        return x
