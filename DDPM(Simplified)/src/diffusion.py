import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import SelfAttention,CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, 4*embed_dim)
        self.linear_2 = nn.Linear(4*embed_dim, 4*embed_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)

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
    
class UpSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels,channels,kernel_size=3,padding=1)

    def forward(self, x):
        #(batch_size,channels,Height,Width)->(batch_size,channels,2*Height,2*Width)
        x = F.interpolate(x,scale_factor=2,mode='nearest')
        x = self.conv(x)
        return x

class UNet_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,n_time = 1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32,in_channels)
        self.conv_feature = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.linear = nn.Linear(n_time,out_channels)

        self.groupnorm_merged = nn.GroupNorm(32,out_channels)
        self.conv_merged = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)

        if in_channels!= out_channels:
            self.residual_layer = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, x: torch.Tensor, time: torch.Tensor)->torch.Tensor:

        #x(batch_size,in_channels,Height,Width)
        #time(1,n_time)

        residue = x
        x = self.groupnorm_feature(x)
        x = F.silu(x)

        #(batch_size,out_channels,Height,Width)
        x = self.conv_feature(x)
        time = F.silu(time)

        #time(1,n_time)->(1,out_channels)
        time = self.linear(time)
        #(batch_size,out_channels,Height,Width) + (1,out_channels,1,1)
        x = x + time.unsqueeze(-1).unsqueeze(-1)

        x = self.groupnorm_merged(x)
        x = F.silu(x)
        x = self.conv_merged(x)

        x = x + self.residual_layer(residue)

        return x   


class UNet_AttentionBlock(nn.Module):
    def __init__(self, n_head, d_head, d_context = 768):
        super().__init__()
        channels = n_head * d_head
        self.groupnorm = nn.GroupNorm(32,channels,eps=1e-6)
        self.conv_input = nn.Conv2d(channels,channels,kernel_size=1,padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1  = SelfAttention(channels,n_head,channels)
        self.layernorm_2  = nn.LayerNorm(channels)
        self.attention_2  = CrossAttention(n_head,channels,d_context,in_proj_bias = False)
        self.layernorm_3  = nn.LayerNorm(channels)

        self.linear_geglu_1 = nn.Linear(channels,4*channels*2)
        self.linear_geglu_2 = nn.Linear(4*channels,channels)
        self.conv_output  = nn.Conv2d(channels,channels,kernel_size=1,padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor)->torch.Tensor:
        #x(batch_size,channels,Height,Width)
        #context(batch_size,seq_len,embed_dim)

        residue_long = x
        x = self.groupnorm(x)
        x = self.conv_input(x)

        n,c,h,w = x.shape
        x = x.view(n,c,h*w).transpose(-1,-2)

        residue_short = x

        #LayerNorm + SelfAttention with skip connection
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x = x + residue_short

        residue_short = x

        #LayerNorm + CrossAttention with skip connection
        x = self.layernorm_2(x)
        x = self.attention_2(x,context)
        x = x + residue_short
        x = self.layernorm_3(x)

        x, gate = self.linear_geglu_1(x).chunk(2,dim=-1)
        x = F.gelu(gate) * x
        x = self.linear_geglu_2(x)
        x+= residue_short

        x = x.transpose(1,2).view(n,c,h,w)

        return self.conv_output(x)+residue_long


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
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

        self.decoders = nn.ModuleList([
            #(batch_size,2560,Height/64,Width/64) -> (batch_size,1280,Height/64,Width/64)
            SwitchSequential(UNet_ResidualBlock(2560,1280)),

            SwitchSequential(UNet_ResidualBlock(2560,1280)),

            SwitchSequential(UNet_ResidualBlock(2560,1280),UpSample(1280)),

            SwitchSequential(UNet_ResidualBlock(2560,1280),UNet_AttentionBlock(8,160)),

            SwitchSequential(UNet_ResidualBlock(2560,1280),UNet_AttentionBlock(8,160)),

            SwitchSequential(UNet_ResidualBlock(1920,1280),UNet_AttentionBlock(8,160),UpSample(1280)),
            
            SwitchSequential(UNet_ResidualBlock(1920,640),UNet_AttentionBlock(8,80)),

            SwitchSequential(UNet_ResidualBlock(1280,640),UNet_AttentionBlock(8,80)),

            SwitchSequential(UNet_ResidualBlock(960,640),UNet_AttentionBlock(8,80),UpSample(640)),

            SwitchSequential(UNet_ResidualBlock(960,320),UNet_AttentionBlock(8,40)),

            SwitchSequential(UNet_ResidualBlock(640,320),UNet_AttentionBlock(8,80)),

            SwitchSequential(UNet_ResidualBlock(640,320),UNet_AttentionBlock(8,40)),
        ])

    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x


class UNet_Outputlayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32,in_channels)
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)

    def forward(self, x):

        #(batch_size,in_channels,Height/8,Width/8)
        x = self.groupnorm(x)
        x = F.silu(x)

        #(batch_size,out_channels,Height/8,Width/8)
        x = self.conv(x)

        return x

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