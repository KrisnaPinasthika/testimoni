import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    efficientnet_b3, EfficientNet_B3_Weights,
    efficientnet_b5, EfficientNet_B5_Weights,
    efficientnet_b6, EfficientNet_B6_Weights, 
    
    efficientnet_v2_m, EfficientNet_V2_M_Weights, 
    efficientnet_v2_l, EfficientNet_V2_L_Weights,
)

from .attention_utils import (
    SWCATransformer, 
    positional_encoding
)

class EncoderBlock(nn.Module):
    """Some Information about EncoderBlock"""

    def __init__(self, backbone_name, freeze=False):
        super(EncoderBlock, self).__init__()
        self.backbone_name = backbone_name
        # Backbones: 
        # EfficientNet  : B1, B3, B5, B6
        # DenseNet      : 121, 169, 201
        # ResNet        : 18, 34, 50, 101
        backbones = {
            'eff_b3' : efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1),
            'eff_b5' : efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1),
            'eff_b6' : efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1),
            
            'eff_v2_m' : efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1),
            'eff_v2_l' : efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1),
        }
        self.backbone = backbones.get(backbone_name)
        
        if self.backbone == None:
            print('Check your backbone again ^.^')
            return None
            
        if freeze:
            for v in self.backbone.parameters():
                v.requires_grad = False

    def forward(self, x):
        features = [x]
        
        encoder = self.backbone.features
        
        for layer in encoder:
            features.append(layer(features[-1]))

        return features

class DecoderBLock(nn.Module):
    """Some Information about DecoderBLock"""

    def __init__(self, x_s_channels, desired_channels, 
                    num_layer, window_size, 
                    num_heads, qkv_bias=True, drop_prob=0.15, 
                    mlp_hidden_ratio=2.0,  device='cuda', 
                    explicit_hw=(480, 640)
                    ):
        super(DecoderBLock, self).__init__()
        self.explicit_hw = explicit_hw
        self.device = device
        self.desired_channels = desired_channels
        self.swca_blocks = SWCATransformer(
            skip_channels=desired_channels, 
            num_layer=num_layer, 
            window_size=window_size, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias,
            attn_drop_prob=drop_prob, 
            lin_drop_prob=drop_prob, 
            mlp_drop_prob=drop_prob,
            mlp_hidden_ratio=mlp_hidden_ratio, 
            device=device
        ).to(device)
        
        self.feed2msa = nn.Sequential(
            nn.Conv2d(in_channels=x_s_channels, out_channels=desired_channels*2, stride=1, kernel_size=1, padding='same'), 
            nn.BatchNorm2d(desired_channels*2),
            nn.GELU(),
        )
        
        self.post_process = nn.Sequential(
            nn.Conv2d(in_channels=desired_channels*2, out_channels=desired_channels, kernel_size=3, stride=1, padding='same'), 
            nn.BatchNorm2d(desired_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=desired_channels, out_channels=desired_channels, kernel_size=3, stride=1, padding='same'), 
            nn.BatchNorm2d(desired_channels),
            nn.GELU(),
        )
        

    def forward(self, skip, x):
        """
        Args: 
            skip    : B, C, H, W
            x       : B, 2C, H/2, W/2
            
            B - Batch Size
            C - Channels size
            H, W - height and width of the features
            D - Desired output channels
        """
        x = F.interpolate(
            x, 
            size=[self.explicit_hw[0], self.explicit_hw[1]], 
            mode='bilinear', 
            align_corners=True
        ) # B, 2C, H, W
        
        skip = F.interpolate(
            skip, 
            size=[self.explicit_hw[0], self.explicit_hw[1]], 
            mode='bilinear', 
            align_corners=True
        ) # B, 2C, H, W
        
        # print(f"Input x : {x.shape} | Skip : {skip.shape}")
        
        process_together = self.feed2msa(torch.concat([skip, x], dim=1))    # B, D, H, W
        process_together = self.add_abs_pe(process_together, self.device)   # B, D, H, W
        # print(f" Processed together : {process_together.shape}")
        
        processed_skip = process_together[:, :self.desired_channels, :, :]
        processed_x = process_together[:, self.desired_channels:, :, :]
        
        # print(f" Processed skip : {processed_skip.shape} | Processed x : {processed_x.shape}")
        
        out = self.swca_blocks(processed_skip, processed_x)                   # B, D, H, W
        
        out = self.post_process(out)
        # print(f" out : {out.shape}")
        # print('---'*20)
        
        return out

    def add_abs_pe(self, x, device): 
        """
        args:
            x : B, C, H/2, W/2
        return: 
            x : B, C, H/2, W/2
        """
        b, c, origin_h, origin_w = x.shape
        x = x.flatten(start_dim=2).permute(0, 2, 1) # B, HW/2, C
        b, hw, c = x.shape
        x = x + positional_encoding(max_len=hw, embed_dim=c, device=device) # B, HW/2, C
        x = x.reshape(b, origin_h, origin_w, c).permute(0, 3, 1, 2) # B, C, H/2, W/2
        
        return x


class PreTrainedUNetSWCA(nn.Module):
    """Some Information about UNetResNet"""

    def __init__(self, device, backbone_name:str, swca_blocks:int, swca_hidden_ratio:float, 
                    window_sizes:int=5, qkv_bias:bool=True, drop_prob:float=0.15, explicit_hw=(480, 640)):
        super(PreTrainedUNetSWCA, self).__init__()
        self.backbone_name = backbone_name.lower()
        self.encoder = EncoderBlock(self.backbone_name, freeze=False).to(device)
        
        self.block_idx, features, dec_heads = self.getFeatures(self.backbone_name)
        
        self.decoder = nn.ModuleList([
            DecoderBLock(
                x_s_channels=features[-1]+features[-2], desired_channels=features[-1]//4,
                num_layer=swca_blocks, num_heads=dec_heads[-1], window_size=window_sizes, 
                qkv_bias=qkv_bias, drop_prob=drop_prob, mlp_hidden_ratio=swca_hidden_ratio, device=device,
                explicit_hw=(explicit_hw[0]//32, explicit_hw[1]//32)
            ),
            
            
            DecoderBLock(
                x_s_channels=features[-1]//4+features[-3], desired_channels=features[-1]//4,
                num_layer=swca_blocks, num_heads=dec_heads[-1], window_size=window_sizes, 
                qkv_bias=qkv_bias, drop_prob=drop_prob, mlp_hidden_ratio=swca_hidden_ratio, device=device,
                explicit_hw=(explicit_hw[0]//16, explicit_hw[1]//16)
            ),
            
            DecoderBLock(
                x_s_channels=features[-1]//4+features[-4], desired_channels=features[-1]//8,
                num_layer=swca_blocks, num_heads=dec_heads[-1], window_size=window_sizes, 
                qkv_bias=qkv_bias, drop_prob=drop_prob, mlp_hidden_ratio=swca_hidden_ratio, device=device,
                explicit_hw=(explicit_hw[0]//8, explicit_hw[1]//8)
            ),
            
            DecoderBLock(
                x_s_channels=features[-1]//8+features[-5], desired_channels=features[-1]//8,
                num_layer=swca_blocks, num_heads=dec_heads[-1], window_size=window_sizes, 
                qkv_bias=qkv_bias, drop_prob=drop_prob, mlp_hidden_ratio=swca_hidden_ratio, device=device,
                explicit_hw=(explicit_hw[0]//4, explicit_hw[1]//4)
            ),
        ]).to(device)
        
        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels=features[-1]//8+features[-5], out_channels=1, kernel_size=1, stride=1, padding="same")
        ).to(device)
    
    def getFeatures(self, backbone_name): 
        # Todo: EfficientNet Attention Head dimension = 8
        if backbone_name == 'eff_b3':
            block_idx = [2, 3, 4, 6, 9]
            features = [24, 32, 48, 136, 1536]
            dec_heads = [4, 4, 8, 8]
        elif backbone_name == 'eff_b5':
            block_idx = [2, 3, 4, 6, 9]
            features = [24, 40, 64, 176, 2048]
            dec_heads = [4, 4, 8, 8]
        elif backbone_name == 'eff_b6':
            block_idx = [2, 3, 4, 6, 9]
            features = [32, 40, 72, 200, 2304]
            dec_heads = [4, 5, 9, 25]
        
        # EfficientNet V2
        elif backbone_name == 'eff_v2_m':
            block_idx = [2, 3, 4, 6, 9]
            features = [24, 48, 80, 176, 1280]
            dec_heads = [4, 8, 8, 16]
        elif backbone_name == 'eff_v2_l':
            block_idx = [2, 3, 4, 6, 9]
            features = [32, 64, 96, 224, 1280]
            dec_heads = [8, 8, 16, 16]
        
        else:
            print('Check your backbone again ^.^')
            return None
        
        return block_idx, features, dec_heads
    
    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.last_conv]
        for m in modules:
            yield from m.parameters()
            
    def forward(self, x):
        enc = self.encoder(x) 
        
        block1 = enc[self.block_idx[0]]
        block2 = enc[self.block_idx[1]]
        block3 = enc[self.block_idx[2]]
        block4 = enc[self.block_idx[3]]
        block5 = enc[self.block_idx[4]]
        
        u1 = self.decoder[0](block4, block5)
        u2 = self.decoder[1](block3, u1)
        u3 = self.decoder[2](block2, u2)
        
        upsampled_u3 = F.interpolate(
            u3, 
            size=[x.shape[2]//2, x.shape[3]//2], 
            mode='bilinear', 
            align_corners=True
        ) # B, 2C, H, W
        
        # print(f" block1_converted : {block1.shape} | u3 : {upsampled_u3.shape}")
        u4 = torch.concat([block1, upsampled_u3], dim=1)
        head = self.last_conv(u4)
        
        # print(head[0][0][0])
        
        return head

if __name__ == '__main__': 
    from torchsummary import summary
    # print(resnet34())
    img = torch.randn((2, 3, 416, 544)).to('cuda')
    model = PreTrainedUNetSWCA(
        device='cuda', 
        backbone_name='eff_b3',  
        window_sizes=5,
        swca_blocks=2,
        swca_hidden_ratio=2.
    ).to('cuda')
    
    print(model(img).shape)