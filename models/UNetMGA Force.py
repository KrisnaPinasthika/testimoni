import torch 
import torch.nn as nn
import torch.nn.functional as F 

from torchvision.models import (
    shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0, 
    ShuffleNet_V2_X0_5_Weights, ShuffleNet_V2_X1_0_Weights, ShuffleNet_V2_X1_5_Weights, ShuffleNet_V2_X2_0_Weights,
    
    efficientnet_b5, EfficientNet_B5_Weights
)
from .miniViT import mViT

class Encoder(nn.Module):
    """Some Information about Encoder"""
    def __init__(self, backbone):
        super(Encoder, self).__init__()
        self.backbone = backbone.lower()
        self.model = self.getBackbone(self.backbone)
        
    def getBackbone(self, backbone):
        model = {
            'x0_5': shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1),
            'x1_0': shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
            'x1_5': shufflenet_v2_x1_5(weights=ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1),
            'x2_0': shufflenet_v2_x2_0(weights=ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1),
            'eff_b5': efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
        }
        
        return model[backbone]
    
    def forward(self, x):
        features = [x]
        
        if self.backbone.split('_')[0] == 'eff':
            encoder = self.model.features
        else:
            encoder = list(self.model.children())[:-1]
            encoder = torch.nn.Sequential(*encoder)
        
        for layer in encoder:
            features.append(layer(features[-1]))
        
        return features 

class Decoder(nn.Module):
    """Some Information about Decoder
        input_features: Combination of Skip and X channels
        desired_output: 
    """
    def __init__(self, input_features, desired_output, skip_channels, x_channels, attention_type:str, device):
        super(Decoder, self).__init__()
        self.attention_type = attention_type.lower()
        self.sig = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU()
        
        self.process_together = nn.Sequential(
            nn.Conv2d(in_channels=input_features, out_channels=desired_output, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(desired_output),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=desired_output, out_channels=desired_output, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(desired_output)
        )
        
        if self.attention_type == 'weighted': 
            self.alpha1 = torch.tensor([0.5], dtype=torch.float32, requires_grad=True, device=device)
            self.alpha1 = nn.Parameter(self.alpha1)
            
        elif self.attention_type == 'multi_weighted': 
            self.alpha1 = torch.rand((1, desired_output, 1, 1), 
                                        dtype=torch.float32, 
                                        requires_grad=True, 
                                        device=device)
            self.alpha1 = nn.Parameter(self.alpha1)
        
        
        self.skip_norm = nn.Sequential(
            nn.Conv2d(in_channels=skip_channels,out_channels=desired_output, kernel_size=1, stride=1, padding='same'), 
            nn.BatchNorm2d(desired_output),
            nn.LeakyReLU(inplace=True),
        )
        
        
        
    def forward(self, skip, x):
        """ 
        args: 
            skip : (B, C_1, H, W)
            x    : (B, 2C_2, H/2, W/2)
        """  
        processed_x =  F.interpolate(x, size=[skip.size(2), skip.size(3)], mode='bilinear', align_corners=True) # (B, 2C_2, H, W)
        
        processed = torch.concat([skip, processed_x], dim=1)    # (B, C_1 + 2C_2, H, W) 
        processed = self.process_together(processed)            # (B, C_1, H, W) 

        if self.attention_type == 'normal': 
            out = self.lrelu(processed) 
        else:
            out = self.alpha1 * (self.skip_norm(skip) * self.sig(processed))

        return out


class UNetMGA(nn.Module):
    """Some Information about UNetMGA"""
    # UNetMGA: Multi Gate Attention - UNet
    def __init__(self, backbone:str, attention_type:str, n_bins=100, min_val=0.1, max_val=10, norm='linear', device='cuda'):
        super(UNetMGA, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        
        backbone = backbone.lower()
        attention_type = attention_type.lower()
        
        self.enc = Encoder(backbone)
        self.enc_idx, channels = self.getFeatures(backbone)
        
        if (attention_type != 'weighted') and (attention_type != 'multi_weighted') and (attention_type != 'normal') :
            print(f"Check the attention type ^w^\nOption available: normal, weighted, multi_weighted")
            return None
        else: 
            print(f"UNetMGA - {attention_type} | VALID")
        
        self.post_encoder = nn.Conv2d(channels[-1], channels[-1], kernel_size=1, stride=1, padding=1)
        
        self.dec = nn.ModuleList([
            Decoder(channels[-1]//1 + channels[-2], channels[-1]//4,  channels[-2], channels[-1]//1, attention_type, device),
            Decoder(channels[-1]//4 + channels[-3], channels[-1]//8,  channels[-3], channels[-1]//4, attention_type, device),
            Decoder(channels[-1]//8 + channels[-4], channels[-1]//16,  channels[-4], channels[-1]//8, attention_type, device),
            Decoder(channels[-1]//16 + channels[-5], channels[-1]//32, channels[-5], channels[-1]//16, attention_type, device),
        ])
        
        self.post_decoder = nn.Conv2d(in_channels=channels[-1]//32, out_channels=128, kernel_size=3, stride=1, padding='same')
        
        self.adaptive_bins_layer = mViT(
            in_channels=128, 
            n_query_channels=128, 
            patch_size=16,
            dim_out=n_bins,
            embedding_dim=128, 
            norm=norm
        )
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1)
        )
    
    def getFeatures(self, backbone): 
        channels = {
            'x0_5': [24, 24, 48, 96, 1024],
            'x1_0': [24, 24, 116, 232, 1024],
            'x1_5': [24, 24, 176, 352, 1024],
            'x2_0': [24, 24, 244, 488, 2048],
            'eff_b5': [24, 40, 64, 176, 2048]
        }
        
        idx_channels = {
            'x0_5': [1, 2, 3, 4, 6],
            'x1_0': [1, 2, 3, 4, 6],
            'x1_5': [1, 2, 3, 4, 6],
            'x2_0': [1, 2, 3, 4, 6],
            'eff_b5': [2, 3, 4, 6, 9]
        }
        
        return idx_channels[backbone], channels[backbone]
    
    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.enc.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.dec, self.adaptive_bins_layer, self.conv_out, self.post_decoder]
        for m in modules:
            yield from m.parameters()

    def forward(self, x):
        enc = self.enc(x)
        
        block1 = enc[self.enc_idx[0]]
        block2 = enc[self.enc_idx[1]]
        block3 = enc[self.enc_idx[2]]
        block4 = enc[self.enc_idx[3]]
        block5 = self.post_encoder(enc[self.enc_idx[4]])
        
        u1 = self.dec[0](block4, block5)
        u2 = self.dec[1](block3, u1)
        u3 = self.dec[2](block2, u2)
        u4 = self.dec[3](block1, u3)
        
        post_dec = self.post_decoder(u4)
        
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(post_dec)
        
        out = self.conv_out(range_attention_maps)
        
        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)
        
        return bin_edges, pred

if __name__ == '__main__': 
    device = torch.device('cuda')
    img = torch.randn(2, 3, 480, 640).to(device)
    attention_type = 'normal'
    model = UNetMGA(backbone='eff_b5', attention_type=attention_type, device=device).to(device)
    
    print(model(img)[0].shape, model(img)[1].shape)