import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops

from torchvision.models import (
    efficientnet_v2_m, EfficientNet_V2_M_Weights, 
    efficientnet_b5, EfficientNet_B5_Weights,
    swin_s, Swin_S_Weights
)

from .swca_attention_utils import (
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
            'eff_b5' : efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1),
            'eff_v2_m' : efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1),
            'swin_b': swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)
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

class DeformableConv2d(nn.Module):
    def __init__(self,
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dilation=1,
                    bias=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(in_channels,
                            2 * kernel_size[0] * kernel_size[1],
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=self.padding,
                            dilation=self.dilation,
                            bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                            1 * kernel_size[0] * kernel_size[1],
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=self.padding,
                            dilation=self.dilation,
                            bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=self.padding,
                            dilation=self.dilation,
                            bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        # op = (n - (k * d - 1) + 2p / s)
        x = torchvision.ops.deform_conv2d(input=x,
                            offset=offset,
                            weight=self.regular_conv.weight,
                            bias=self.regular_conv.bias,
                            padding=self.padding,
                            mask=modulator,
                            stride=self.stride,
                            dilation=self.dilation)
        return x

class DecoderBLock(nn.Module):
    """Some Information about DecoderBLock
        feature_channels : 3 number of channel
    """

    def __init__(self, feature_channels, input_hw, desired_output_channels, num_layer, window_size, 
                    num_heads, qkv_bias, drop_prob, device):
        super(DecoderBLock, self).__init__()
        # print(f"Decoder channel feat : {feature_channels}")
        self.device = device
        self.feature_channels = feature_channels
        self.swca_blocks = SWCATransformer(
            channels=desired_output_channels, 
            num_layer=num_layer, 
            window_size=window_size, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias,
            drop_prob=drop_prob,
            device=device
        ).to(device)
        
        self.feat1 = nn.Sequential(
            nn.Conv2d(feature_channels[0], desired_output_channels, stride=1, kernel_size=1, padding='same'),
            nn.GELU(),
            nn.Conv2d(desired_output_channels, desired_output_channels, stride=1, kernel_size=3, padding='same'),
        )
        self.feat2 = nn.Sequential(
            nn.Conv2d(feature_channels[1], desired_output_channels, stride=2, kernel_size=2),
            nn.GELU(),
            nn.Conv2d(desired_output_channels, desired_output_channels, stride=1, kernel_size=3, padding='same'),
        )
        
        self.process_together = nn.Sequential(
            nn.Conv2d(desired_output_channels, desired_output_channels, stride=1, kernel_size=3, padding='same'),
            nn.GELU(),
            nn.Conv2d(desired_output_channels, desired_output_channels, stride=1, kernel_size=3, padding='same'),
            DeformableConv2d(desired_output_channels, desired_output_channels),
            nn.GELU(),
        )
        
        
    def forward(self, x, skip):
        """
        Args: 
            X       = B, C1, H/2, W/2     (high level feature) from deeper feature
            Skip    = B, C2, H, W         (low level feature) from Skip connection
        """
        # skip_ori = skip.clone()
        b, c, dest_h, dest_w = skip.shape 
        
        # Todo: X_up for combining at the last step
        # x_up = F.interpolate(x, size=[dest_h, dest_w], mode='bilinear', align_corners=True) # B, D, H, W
        
        # Todo: extracting high level feature and generalize the feature channels
        x = self.feat1(x)               # B, D, H/2, W/2 
        skip = self.feat2(skip)         # B, D, H/2, W/2 
        
        # Todo: add the fixed positional encoding for msa calculation
        x = self.add_abs_pe(x, self.device)   # B, D, H/2, W/2 
        skip = self.add_abs_pe(skip, self.device)   # B, D, H/2, W/2 
        
        swca_calc, attn_w8 = self.swca_blocks(x, x, skip)  # B, D, H/2, W/2
        
        # out = self.process_together(torch.concat([swca_calc, skip, x_up], dim=1)) # B, D, H, W
        out = self.process_together(swca_calc)  # B, D, H, W
        out = F.interpolate(out, size=[dest_h, dest_w], mode='bilinear', align_corners=True) # B, D, H, W
        
        return out, attn_w8

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

class UNetSWCA(nn.Module):
    """Some Information about UNetResNet"""

    def getFeatures(self, backbone_name): 
        # Todo: EfficientNet Attention Head dimension = 8
        if backbone_name == 'eff_v2_s':
            block_idx = [2, 3, 4, 6, 8]
            features = [24, 48, 64, 160, 1280]
            dec_heads = [4, 8, 8, 16]
        elif backbone_name == 'eff_v2_m':
            block_idx = [2, 3, 4, 6, 9]
            features = [24, 48, 80, 176, 1280]
            dec_heads = [1, 1, 1, 1]
        elif backbone_name == 'eff_v2_l':
            block_idx = [2, 3, 4, 6, 9]
            features = [32, 64, 96, 224, 1280]
            dec_heads = [8, 8, 16, 16]
            
        elif backbone_name == 'eff_b5':
            block_idx = [2, 3, 4, 6, 9]
            features = [24, 40, 64, 176, 2048]
            dec_heads = [8, 8, 16, 16]
            
        elif backbone_name == 'swin_b':
            block_idx = [2, 4, 6, 8]
            features = [96, 192, 384, 768]
            dec_heads = [8, 8, 8, 8]
        
        else:
            print('Check your backbone again ^.^')
            return None
        
        return block_idx, features, dec_heads
    
    def __init__(self, 
        device, 
        input_hw:tuple,
        backbone_name:str, 
        window_sizes:int,
        swca_blocks:int,
        drop_prob:float=0.35, 
        qkv_bias:bool=True):
        
        super(UNetSWCA, self).__init__()
        self.backbone_name = backbone_name.lower()
        self.input_hw = input_hw
        
        self.encoder = EncoderBlock(self.backbone_name, freeze=False).to(device)
        
        self.block_idx, features, dec_heads = self.getFeatures(self.backbone_name)
        
        self.decoder = nn.ModuleList([
            DecoderBLock(
                [features[-1], features[-2]], (input_hw[0]//32, input_hw[1]//32), 
                desired_output_channels=features[-1]//4,
                num_layer=swca_blocks[-1], num_heads=dec_heads[-1], 
                window_size=window_sizes, qkv_bias=qkv_bias, drop_prob=drop_prob, device=device
            ),
            
            DecoderBLock(
                [features[-1]//4, features[-3]], (input_hw[0]//16, input_hw[1]//16), 
                desired_output_channels=features[-1]//8,
                num_layer=swca_blocks[-2], num_heads=dec_heads[-2], 
                window_size=window_sizes, qkv_bias=qkv_bias, drop_prob=drop_prob, device=device
            ),
            DecoderBLock(
                [features[-1]//8, features[-4]], (input_hw[0]//8, input_hw[1]//8), 
                desired_output_channels=features[-1]//8,
                num_layer=swca_blocks[-3], num_heads=dec_heads[-3], 
                window_size=window_sizes, qkv_bias=qkv_bias, drop_prob=drop_prob, device=device
            ),
            
            DecoderBLock(
                [features[-1]//8, features[-5]], (input_hw[0]//4, input_hw[1]//4), 
                desired_output_channels=features[-1]//8,
                num_layer=swca_blocks[-3], num_heads=dec_heads[-4], 
                window_size=window_sizes, qkv_bias=qkv_bias, drop_prob=drop_prob, device=device
            ),
        ]).to(device)
        
        self.non_attention_head = nn.Conv2d(features[-1]//8, 1, kernel_size=1, stride=1, padding="same")
        
    def forward(self, x):
        ori_b, ori_c, ori_h, ori_w = x.shape
        
        x = F.interpolate(x, size=[self.input_hw[0], self.input_hw[1]], mode='bilinear', align_corners=True)
        
        enc = self.encoder(x) 
        
        block0 = enc[self.block_idx[0]]
        block1 = enc[self.block_idx[1]]
        block2 = enc[self.block_idx[2]]
        block3 = enc[self.block_idx[3]]
        block4 = enc[self.block_idx[4]]
        
        # print(f"Block 0 : {block0.shape}")
        # print(f"Block 1 : {block1.shape}")
        # print(f"Block 2 : {block2.shape}")
        # print(f"Block 3 : {block3.shape}")
        # print(f"Block 4 : {block4.shape}")
        
        u1, msa_counter1 = self.decoder[0](block4, block3)
        # print(f"U1 : {u1.shape}")
        
        u2, msa_counter2 = self.decoder[1](u1, block2)
        # print(f"U2 : {u2.shape}")
        
        u3, msa_counter3 = self.decoder[2](u2, block1)
        # print(f"U3 : {u3.shape}")
        
        u4, msa_counter4 = self.decoder[3](u3, block0)
        # print(f"U4 : {u4.shape}")
        
        out = F.interpolate(u4, size=[ori_h, ori_w], mode='bilinear', align_corners=True)
        
        head = self.non_attention_head(out)
        
        cumulative_msa = msa_counter1 + msa_counter2 + msa_counter3 + msa_counter4
        
        return head, cumulative_msa
    
    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.non_attention_head]
        for m in modules:
            yield from m.parameters()

if __name__ == '__main__': 
    from torchsummary import summary
    # print(resnet34())
    device = 'cuda'
    img = torch.randn((1, 3, 481, 640)).to(device)
    
    model = UNetSWCA(
        device=device, 
        input_hw=(480, 640),
        backbone_name='eff_b5', 
        window_sizes=5,
        swca_blocks=[2, 2, 2]
    ).to(device)
    
    pred, msa = model(img)
    print(pred.shape, msa)
    
    from prettytable import PrettyTable
    def count_parameters(model):
        
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        
        decoder = 0 
        decoder_0 = 0
        decoder_1 = 0
        decoder_2 = 0
        decoder_3 = 0
        encoder = 0
        
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
            
            if name.split('.')[0] == 'decoder': 
                decoder += params
                if name.split('.')[1] == '0': 
                    decoder_0 += params
                elif name.split('.')[1] == '1': 
                    decoder_1 += params
                if name.split('.')[1] == '2': 
                    decoder_2 += params
                elif name.split('.')[1] == '3': 
                    decoder_3 += params
            else:
                encoder += params
                
        print(table)
        print(f"Total Trainable Params: {total_params:,}")
        print(f"Total Encoder Params: {encoder:,}")
        print(f"Total Decoder Params: {decoder:,}")
        print(f"Total Decoder 0 : {decoder_0:,} | 1 : {decoder_1:,} | 2 : {decoder_2:,} | 3 : {decoder_3:,}")
        
        return total_params
    
    count_parameters(model)