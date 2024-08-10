import torch 
import torch.nn as nn
import torch.nn.functional as F 
from einops import rearrange
import numpy as np

from torchvision.models import (
    shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0, 
    ShuffleNet_V2_X0_5_Weights, ShuffleNet_V2_X1_0_Weights, ShuffleNet_V2_X1_5_Weights, ShuffleNet_V2_X2_0_Weights,
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
    efficientnet_b5, EfficientNet_B5_Weights
)

class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(2, 3))
    
def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask

def get_relative_distances(window_size):
    indices = torch.tensor(
        np.array([[x, y] for x in range(window_size) for y in range(window_size)]), 
        dtype=torch.long)
    distances = indices[None, :, :] - indices[:, None, :]
    return distances

class MultiHeadSWCA(nn.Module):
    """Some Information about MultiHeadCrossAttention"""
    def __init__(self, input_features:int, cyclic_shift:bool, window_size:int, num_heads:int, 
                    qkv_bias=True, attn_drop_prob=0.0, lin_drop_prob=0.0, device='cuda'):
        super(MultiHeadSWCA, self).__init__()
        self.device = device
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dims = input_features // num_heads 
        self.cyclic_shift = cyclic_shift
        
        if cyclic_shift:
            displacement = window_size // 2
            self.cyclic_propagate = CyclicShift(-displacement)
            self.cyclic_revert = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(
                create_mask(window_size=window_size, 
                                displacement=displacement,
                                upper_lower=True, 
                                left_right=False), 
                requires_grad=False
            )
            self.left_right_mask = nn.Parameter(
                create_mask(window_size=window_size, 
                                displacement=displacement,
                                upper_lower=False, 
                                left_right=True), 
                requires_grad=False
            )

        self.relative_indices = get_relative_distances(window_size) + window_size - 1
        self.pe = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        # self.pe = nn.Parameter(torch.randn(window_size**2, window_size**2), requires_grad=True)
        
        self.q = nn.Linear(in_features=self.head_dims, out_features=self.head_dims, bias=qkv_bias)
        self.k = nn.Linear(in_features=self.head_dims, out_features=self.head_dims, bias=qkv_bias)
        self.v = nn.Linear(in_features=self.head_dims, out_features=self.head_dims, bias=qkv_bias)
        
        self.lin = nn.Linear(in_features=self.head_dims, out_features=self.head_dims, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_prob)
        self.lin_drop = nn.Dropout(lin_drop_prob)
    
    def reshape_features2token(self, data): 
        """
            Input: B, C, H, W 
            Return : B, H*W, C || (batch_size, n_tokens, dim), and The origin
        """
        b, c, h, w = data.shape
        data = data.reshape(b, c, h*w)
        data = data.permute(0, 2, 1)
        
        return data, (b, c, h, w)
    
    def reshape_token2features(self, token, original_shape):
        
        """
            Input: batch_size, n_tokens, dim
            Return : B, C, H, W
        """
        token = token.permute(0, 2, 1)
        token = token.reshape(original_shape)
        
        return token
    
    def forward(self, features, original_shape):
        """
        Args: 
            features : batch_size, n_tokes, dim
            original_shape : (B, C, H, W)
        Return:
            out     : batch_size, n_tokes, dim
        """
        # Todo: Reshaping from batch_size, n_tokes, dim -> B, C, H, W to make a cyclic shift or calculating window size partition
        features = self.reshape_token2features(features, original_shape)
        
        if self.cyclic_shift:
            features = self.cyclic_propagate(features)
            
        b, c, h, w = features.shape
        n_h, n_w = h//self.window_size, w//self.window_size
        window_squared = self.window_size*self.window_size
        
        # Reshape x and skip to [b, num_head, n_h*n_w, windows*window, head_dim]
        # print(x.shape)
        features = features.reshape(b, self.num_heads, self.head_dims, n_h, self.window_size, n_w, self.window_size)
        features = features.permute(0, 1, 3, 5, 4, 6, 2) # b, num_head, n_h, n_w, window, window, head_dim
        features = features.reshape(b, self.num_heads, n_h*n_w, window_squared, self.head_dims)
        
        q = self.q(features)    # b, num_head, n_h*n_w, window_squared, head_dim
        k = self.k(features)    # b, num_head, n_h*n_w, window_squared, head_dim
        v = self.v(features)    # b, num_head, n_h*n_w, window_squared, head_dim
        
        # qk = b, num_head, n_h*n_w, window_squared, window_squared
        qk = ( torch.matmul(q, k.transpose(3, 4)) ) / np.sqrt(self.head_dims)
        qk += self.pe[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        
        if self.cyclic_shift:
            qk[:, :, -n_w:] += self.upper_lower_mask
            qk[:, :, n_w-1::n_w] += self.left_right_mask
        
        attn_weight = self.attn_drop(torch.softmax(qk, dim=-1)) # b, num_head, n_h*n_w, window_squared, window_squared
        out = torch.matmul(attn_weight, v)                      # b, num_head, n_h*n_w, window_squared, head_dim
        out = self.lin_drop(self.lin(out))                      # b, num_head, n_h*n_w, window_squared, head_dim
        
        # out ==> [b, num_head, n_h*n_w, window_squared, head_dim] to [b, e, h, w]
        out = out.permute(0, 1, 4, 2, 3).reshape(b, c, n_h, n_w, self.window_size, self.window_size)
        out = out.permute(0, 1, 2, 4, 3, 5).reshape(b, c, h, w)
        
        if self.cyclic_shift:
            out = self.cyclic_revert(out) # B, C, H, W
        
        out, original_size = self.reshape_features2token(out) # B, C, H, W -> batch_size, n_tokes, dim
        
        return out

class MLP(nn.Module):
    """Some Information about MLP"""
    def __init__(self, in_features, hidden_features, out_features, drop_prob=0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.drop = nn.Dropout(p=drop_prob)
        self.act = nn.GELU()

    def forward(self, x):
        """
        input:
            x = (batch_size, n_tokes, embed_dim)
        return:
            out = (batch_size, n_tokes, embed_dim)
        """
        x = self.drop(self.act(self.fc1(x))) # (batch_size, n_tokes, hidden_features)
        x = self.drop(self.act(self.fc2(x))) # (batch_size, n_tokes, hidden_features)
        
        return x # (batch_size, n_tokes, embed_dim)

class MiniSwinEx(nn.Module):
    """Some Information about RegularTransformer"""
    def __init__(self, embed_dim, num_heads, qkv_bias=True, num_tblock=3, window_size=10,
                    n_bins=100, attn_drop_prob=0.0, lin_drop_prob=0.0, mlp_hidden_ratio=4., 
                    mlp_drop_prob=0.0, patch_size=16, norm='linear', device='cuda'):
        super(MiniSwinEx, self).__init__()
        self.device = device
        self.num_tblock = num_tblock
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.norm = norm
        self.msa = nn.ModuleList([])
        self.mlp = nn.ModuleList([]) 
        
        for _ in range(num_tblock): 
            self.msa.append(nn.ModuleList([
                MultiHeadSWCA(embed_dim, False, window_size, num_heads, qkv_bias, attn_drop_prob, lin_drop_prob, device), 
                MultiHeadSWCA(embed_dim, True, window_size, num_heads, qkv_bias, attn_drop_prob, lin_drop_prob, device), 
            ]))

            self.mlp.append(nn.ModuleList([
                MLP(embed_dim, int(embed_dim*mlp_hidden_ratio), embed_dim, mlp_drop_prob), 
                MLP(embed_dim, int(embed_dim*mlp_hidden_ratio), embed_dim, mlp_drop_prob)
            ]))
        
        self.norm1 = nn.ModuleList([
            nn.LayerNorm(normalized_shape=embed_dim)
            for _ in range(num_tblock)
        ]).to(device)
        
        self.norm2 = nn.ModuleList([
            nn.LayerNorm(normalized_shape=embed_dim)
            for _ in range(num_tblock)
        ]).to(device)
        
        self.simple_token_converter = nn.Conv2d(embed_dim, embed_dim,
                                            kernel_size=patch_size, stride=patch_size, padding=0)
        
        self.conv3x3 = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, stride=1, padding='same')
        self.regressor = nn.Sequential(nn.Linear(embed_dim, 256),
                                        nn.LeakyReLU(),
                                        nn.Linear(256, 256),
                                        nn.LeakyReLU(),
                                        nn.Linear(256, n_bins))

    def forward(self, x):
        """
        Input: 
            x = B, C, H, W
        Return:
            second_calc = B, C, H, W
        """
        keys = self.conv3x3(x) # batch_size, embed_dim, H, W
        
        b, c, h, w = x.shape    
        # Todo: Patching the input X
        x = self.simple_token_converter(x)  # batch_size, embed_dim, H/16, W/16
        
        x = x.reshape(b, c, int((h/self.patch_size)*(w/self.patch_size)))    #  B, C, n_tokens where "n_tokens = H/16*W/16"
        x = x.permute(0, 2, 1)      #  B, n_tokens, C    
        
        # Todo: add positional encoding 
        b_, max_len, embed_dim = x.shape
        x = x + self.positional_encoding(max_len, embed_dim, self.device)
        
        for i in range(self.num_tblock): 
            # Todo: Non-cyclic swin transformer calculation 
            x_11 = self.msa[i][0](self.norm1[i](x), (b, c, (h//self.patch_size), (w//self.patch_size)))     # B, n_tokens, C
            x_11 = x_11 + x
            x_12 = self.mlp[i][0](x_11)                                   # B, n_tokens, C
            x_12 = x_12 + x_11
            
            x_21 = self.msa[i][1](self.norm2[i](x_12), (b, c, (h//self.patch_size), (w//self.patch_size)))    # B, n_tokens, C
            x_21 = x_21 + x_12
            x_22 = self.mlp[i][1](x_21)                                   # B, n_tokens, C
            x_22 = x_22 + x_21 
            # Todo: change the variable x for the next loop
            x = x_22.clone()
        
        regression_head = x[:, 0, :]                         # [B, n_tokens[0], dim]  
        queries = x[:, 1:self.embed_dim + 1, :]              # [B, n_tokens[1 -> embed_dim], dim]
        
        range_attention_maps = torch.matmul(keys.reshape(b, c, h*w).permute(0, 2, 1), queries)   # B, C, H*W
        range_attention_maps = range_attention_maps.reshape(b, c, h, w)         # B, C, H, W
        
        y = self.regressor(regression_head)     # [batch_size, dim]  
        
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)
            
        y = y / y.sum(dim=1, keepdim=True)
        
        return y, range_attention_maps

    def positional_encoding(self, max_len, embed_dim, device):
        # initialize a matrix angle_rads of all the angles
        angle_rads = np.arange(max_len)[:, np.newaxis] / np.power(
            10_000, (2 * (np.arange(embed_dim)[np.newaxis, :] // 2)) / np.float32(embed_dim)
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return torch.tensor(pos_encoding, dtype=torch.float32, device=device, requires_grad=False)

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
    def __init__(self, backbone:str, attention_type:str, n_bins=100, min_val=0.1, max_val=10, 
                    norm='linear', H_x_W=(240, 320),device='cuda'):
        super(UNetMGA, self).__init__()
        self.H_x_W = H_x_W
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
        
        self.adabin_swin_extractor = MiniSwinEx(
            embed_dim=128, 
            num_heads=4, 
            qkv_bias=True, 
            num_tblock=4,
            window_size=5,
            n_bins=n_bins, 
            attn_drop_prob=0.1, 
            lin_drop_prob=0.1, 
            mlp_hidden_ratio=8., 
            mlp_drop_prob=0.1, 
            patch_size=2,
            norm=norm, 
            device=device)
        
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
        modules = [self.dec, self.adabin_swin_extractor, self.conv_out, self.post_decoder, self.post_encoder]
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
        post_dec = F.interpolate(post_dec, size=(self.H_x_W[0], self.H_x_W[1]), mode='bilinear', align_corners=True)
        
        bin_widths_normed, range_attention_maps = self.adabin_swin_extractor(post_dec)
        
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

    
    pred = model(sample)
    print(pred[0].shape, pred[1].shape)
    
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
        print(f"Total Trainable Params: {total_params:,}")
        return total_params
    
    count_parameters(model)