import torch 
import torch.nn as nn
import torch.nn.functional as F 

from torchvision.models import (
    shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0, 
    ShuffleNet_V2_X0_5_Weights, ShuffleNet_V2_X1_0_Weights, ShuffleNet_V2_X1_5_Weights, ShuffleNet_V2_X2_0_Weights,
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
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
            'eff_b5': efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1), 
            'eff_v2_m': efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
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
        feature_input: A combination of Skip and X channels
        desired_output: Number of output channels
        skip_channels: list of channels from encoder-t to encoder-1 where t is the same number as decoder (from low to high level feature)
    """
    def __init__(self, desired_output, x_channel, skip_channels:list, attention_type:str, device):
        super(Decoder, self).__init__()
        self.attention_type = attention_type.lower()
        self.sig = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.skip_channels = skip_channels
        
        if self.attention_type == 'weighted': 
            self.alpha1 = torch.tensor([0.5], dtype=torch.float32, requires_grad=True, device=device)
            self.alpha1 = nn.Parameter(self.alpha1)
            
        elif self.attention_type == 'multi_weighted': 
            self.alpha1 = torch.rand((1, desired_output*(len(self.skip_channels) + 1), 1, 1), 
                                        dtype=torch.float32, 
                                        requires_grad=True, 
                                        device=device)
            self.alpha1 = nn.Parameter(self.alpha1)
        
        
        if len(self.skip_channels) > 0:
            self.process_all_skip = nn.ModuleList([])
            
            for i in range(len(self.skip_channels)): 
                if i >= 1: 
                    self.process_all_skip.append(
                        nn.Sequential(
                            nn.MaxPool2d(kernel_size=2**i, stride=2**i),
                            nn.Conv2d(self.skip_channels[i], desired_output, kernel_size=1, stride=1, padding='same'),
                            nn.BatchNorm2d(desired_output),
                        )
                    )
                else:
                    self.process_all_skip.append(
                        nn.Sequential(
                            nn.Conv2d(self.skip_channels[i], desired_output, kernel_size=1, stride=1, padding='same'),
                            nn.BatchNorm2d(desired_output),
                        )
                    )
                    
        self.post_process_x = nn.Sequential(
            nn.Conv2d(x_channel, desired_output, kernel_size=1, stride=1, padding='same'),
            nn.BatchNorm2d(desired_output),
        )
        self.out = nn.Sequential(
                nn.GELU(),
                nn.Conv2d(desired_output*(len(self.skip_channels) + 1), desired_output, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm2d(desired_output),
                nn.GELU(),
                nn.Conv2d(desired_output, desired_output, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm2d(desired_output),
                nn.GELU(),
        )
        
        
        
        
    def forward(self, skip, x):
        """ 
        args: 
            skip : (N, B, Cs, H, W)
            x    : (B, 2C, H/2, W/2)
        """  
        if len(skip) != len(self.skip_channels): 
            raise ValueError(f"The length of skip [{len(skip)}] and skip_channels [{len(self.skip_channels)}] doesn't match")
            
        processed_x =  F.interpolate(x, size=[skip[0].size(2), skip[0].size(3)], mode='bilinear', align_corners=True) # (B, 2C, H, W)
        processed_x = self.post_process_x(processed_x)  # (B, Desired, H, W)
        # print(f"Processed x : {processed_x.shape}")
        
        processed = []
        for skip_feat, skip_layer in zip(skip, self.process_all_skip): 
            skip_norm = skip_layer(skip_feat)
            # print(f"skip_feat : {skip_feat.shape} => {skip_norm.shape}")
            processed.append(skip_norm)                 # (B, Desired, H, W)
        
        processed.append(processed_x)                   
        processed = torch.concat(processed, dim=1)      # (B, skip_channels *  Desired, H, W)
        # print(f"processed : {processed.shape}")
        
        processed = processed * self.sig(self.alpha1)   # (B, skip_channels *  Desired, H, W)
        
        out = self.out(processed)                       # (B, Desired, H, W)

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
            Decoder(channels[-1]//2, channels[-1], channels[::-1][1:], attention_type, device),
            Decoder(channels[-1]//4, channels[-1]//2, channels[::-1][2:], attention_type, device),
            Decoder(channels[-1]//8, channels[-1]//4, channels[::-1][3:], attention_type, device),
            Decoder(channels[-1]//16, channels[-1]//8, channels[::-1][4:], attention_type, device),
        ])
        
        self.post_decoder = nn.Conv2d(in_channels=channels[-1]//16, out_channels=128, kernel_size=3, stride=1, padding='same')
        
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
            'eff_b5': [24, 40, 64, 176, 2048],
            'eff_v2_m': [24, 48, 80, 176, 1280]
        }
        
        idx_channels = {
            'x0_5': [1, 2, 3, 4, 6],
            'x1_0': [1, 2, 3, 4, 6],
            'x1_5': [1, 2, 3, 4, 6],
            'x2_0': [1, 2, 3, 4, 6],
            'eff_b5': [2, 3, 4, 6, 9], 
            'eff_v2_m': [2, 3, 4, 6, 9]
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
        
        # for i, block in enumerate([block1, block2, block3, block4, block5]): 
        #     print(f"Encoder {i} : {block.shape}")
        
        u1 = self.dec[0]([block4, block3, block2, block1], block5)
        # print(u1.shape)
        u2 = self.dec[1]([block3, block2, block1], u1)
        # print(u2.shape)
        u3 = self.dec[2]([block2, block1], u2)
        # print(u3.shape)
        u4 = self.dec[3]([block1], u3)
        # print(u4.shape)
        
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
    attention_type = 'multi_weighted'
    sample = torch.randn((1, 3, 480, 640)).to(device)
    model = UNetMGA(backbone='eff_b5', attention_type=attention_type, device=device).to(device)
    
    # skips = [torch.randn(2, 128, 30, 60).to(device),    
    #             torch.randn(2, 64, 60, 120).to(device),   
    #             torch.randn(2, 32, 120, 240).to(device),   
    #             torch.randn(2, 16, 240, 480).to(device),   
    #             ]
    
    # pred = model(sample)
    # print(pred[0].shape, pred[1].shape)
    
    from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
    count_parameters(model)