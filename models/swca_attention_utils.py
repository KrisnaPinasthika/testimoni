import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F
from einops import rearrange

def positional_encoding(max_len, embed_dim, device):
    # initialize a matrix angle_rads of all the angles
    angle_rads = np.arange(max_len)[:, np.newaxis] / np.power(
        10_000, (2 * (np.arange(embed_dim)[np.newaxis, :] // 2)) / np.float32(embed_dim)
    )
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return torch.tensor(pos_encoding, dtype=torch.float32, device=device, requires_grad=False)


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

class MultiheadAttention(nn.Module):
    """Some Information about MultiheadAttention"""
    def __init__(self, embed_dim, num_heads, qkv_bias=False, attn_drop_prob=0.0, lin_drop_prob=0.0, device='cuda'):
        super(MultiheadAttention, self).__init__()
        self.device = device
        self.embed_dim = embed_dim 
        self.num_heads = num_heads 
        self.head_dims = embed_dim // num_heads
        self.qkv = nn.Linear(in_features=embed_dim, out_features=embed_dim*3, bias=qkv_bias)
        self.lin = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        
        self.att_drop = nn.Dropout(p=attn_drop_prob)
        self.lin_drop = nn.Dropout(p=lin_drop_prob)
        
        
    def forward(self, x):
        """
        input: 
            x = batch_size, n_tokes, embed_dim
        output:
            out = batch_size, n_tokes, embed_dim
        """
        # org_batch_size, org_channel, org_height, org_width = x.shape
        # x = x.reshape(org_batch_size, org_channel, org_height*org_width).permute(0, 2, 1)
        batch_size, n_tokens, dim = x.shape 

        pe = positional_encoding(max_len=n_tokens, embed_dim=dim, device=self.device)
        x += pe
        # print(dim, self.embed_dim)
        if dim != self.embed_dim:
            print('[ERROR MHSA] : dim != embeddim') 
            raise ValueError
        
        # qkv = (batch_size, n_tokes, embed_dim * 3)
        qkv = self.qkv(x)
        
        # reshaped qkv = (batch_size, n_tokes, 3, num_heads, head_dims)
        # permuted qkv = (3, batch_size, num_heads, n_tokes, head_dims)
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.num_heads, self.head_dims)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # q, k, and v = (batch_size, num_heads, n_tokes, head_dims)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        
        qk_transposed = torch.matmul(q, k.transpose(2, 3)) / np.sqrt(self.head_dims) # (batch_size, num_heads, n_tokens, n_tokens)
        attention_weights = torch.softmax(qk_transposed, dim=-1) # (batch_size, num_heads, n_tokens, n_tokens)
        attention_weights = self.att_drop(attention_weights)
        
        weighted_avg = torch.matmul(attention_weights, v) # (batch_size, num_heads, n_tokes, head_dims)
        weighted_avg = weighted_avg.transpose(1, 2).flatten(start_dim=2) # (batch_size, n_tokes, num_heads * head_dims)
        
        out = self.lin(weighted_avg) # (batch_size, n_tokes, embed_dim)
        out = self.lin_drop(out) # (batch_size, n_tokes, embed_dim)
        
        # Todo: embed_dim ~ channel
        # out = out.reshape(org_batch_size, org_height, org_width, org_channel) # (batch_size, height, width, channel)
        # out = out.permute(0, 3, 1, 2) # (batch_size, channel, height, width)
        
        return out
    
class RegularTransformer(nn.Module):
    """Some Information about RegularTransformer"""
    def __init__(self, embed_dim, num_heads, qkv_bias=False, num_tblock=4,
                    attn_drop_prob=0.0, lin_drop_prob=0.0, 
                    mlp_hidden_ratio=4., mlp_drop_prob=0.0, device='cuda'):
        super(RegularTransformer, self).__init__()
        
        self.msa = nn.ModuleList([
            MultiheadAttention( 
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                qkv_bias=qkv_bias, 
                attn_drop_prob=attn_drop_prob, 
                lin_drop_prob=lin_drop_prob, 
                device=device
            ) 
            for _ in range(num_tblock)
        ]).to(device)
        
        self.mlp = nn.ModuleList([
            MLP(
                in_features=embed_dim, 
                hidden_features=int(embed_dim*mlp_hidden_ratio), 
                out_features=embed_dim, 
                drop_prob=mlp_drop_prob
            )
            for _ in range(num_tblock)
        ]).to(device)
        
        self.norm1 = nn.ModuleList([
            nn.LayerNorm(normalized_shape=embed_dim)
            for _ in range(num_tblock)
        ]).to(device)
        
        self.norm2 = nn.ModuleList([
            nn.LayerNorm(normalized_shape=embed_dim)
            for _ in range(num_tblock)
        ]).to(device)
        
        self.num_tblock = num_tblock

    def forward(self, x):
        """
        Input: 
            x = B, C, H, W
        Return:
            second_calc = B, C, H, W
        """
        org_batch_size, org_channel, org_height, org_width = x.shape
        x = x.reshape(org_batch_size, org_channel, org_height*org_width).permute(0, 2, 1) # batch_size, n_tokens, dim
        
        for i in range(self.num_tblock): 
            first_calc = self.msa[i]( self.norm1[i](x) )        # batch_size, n_tokens, dim
            first_calc = first_calc + x                         # batch_size, n_tokens, dim
            second_calc = self.mlp[i]( self.norm2[i](first_calc) ) # batch_size, n_tokens, dim
            second_calc = second_calc + first_calc              # batch_size, n_tokens, dim
            
            x = second_calc.clone()
        
        second_calc = second_calc.reshape(org_batch_size, org_height, org_width, org_channel) # (batch_size, height, width, channel)
        second_calc = second_calc.permute(0, 3, 1, 2) # (batch_size, channel, height, width)
        
        return second_calc
    
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
    def __init__(self, skip_channels, cyclic_shift, window_size, num_heads, 
                    qkv_bias=False, attn_drop_prob=0.0, lin_drop_prob=0.0, device='cuda'):
        super(MultiHeadSWCA, self).__init__()
        self.device = device
        self.skip_channels = skip_channels
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dims = skip_channels // num_heads 
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
        
    def reshaping(self, feature, original_sizes):
        """
            Input = B, H*W, C or B, n_token, embed_dims
            Output = B, C, H, W
        """
        b, c, h, w = original_sizes
        feature = feature.permute(0, 2, 1)      # B, C, H*W
        feature = feature.reshape(b, c, h, w)   # B, C, H, W
        
        return feature 
    
    def shaping(self, feature): 
        """
            Input = B, C, H, W
            Output = B, H*W, C or B, n_token, embed_dims
        """
        b, c, h, w = feature.shape
        feature = feature.reshape(b, c, h*w)
        feature = feature.permute(0, 2, 1)
        
        return feature, (b, c, h, w)
    
    def forward(self, q_feat, k_feat, v_feat, original_sizes):
        """
        Args: 
            q_feat    : b, h*w, c
            k_feat    : b, h*w, c
            v_feat    : b, h*w, c
        Return:
            out     : b, h*w, c
        """
        q_feat = self.reshaping(q_feat, original_sizes) # b, c, h, w
        k_feat = self.reshaping(k_feat, original_sizes) # b, c, h, w
        v_feat = self.reshaping(v_feat, original_sizes) # b, c, h, w
        
        if self.cyclic_shift:
            q_feat = self.cyclic_propagate(q_feat)
            k_feat = self.cyclic_propagate(k_feat)
            v_feat = self.cyclic_propagate(v_feat)
            
        b, c, h, w = q_feat.shape
        n_h, n_w = h//self.window_size, w//self.window_size
        window_squared = self.window_size*self.window_size
        
        # Reshape features to [b, num_head, n_h*n_w, windows*window, head_dim]
        q_feat = q_feat.reshape(b, self.num_heads, self.head_dims, n_h, self.window_size, n_w, self.window_size)
        q_feat = q_feat.permute(0, 1, 3, 5, 4, 6, 2) # b, num_head, n_h, n_w, window, window, head_dim
        q_feat = q_feat.reshape(b, self.num_heads, n_h*n_w, window_squared, self.head_dims)
        
        
        k_feat = k_feat.reshape(b, self.num_heads, self.head_dims, n_h, self.window_size, n_w, self.window_size)
        k_feat = k_feat.permute(0, 1, 3, 5, 4, 6, 2) # b, num_head, n_h, n_w, window, window, head_dim
        k_feat = k_feat.reshape(b, self.num_heads, n_h*n_w, window_squared, self.head_dims)
        
        
        v_feat = v_feat.reshape(b, self.num_heads, self.head_dims, n_h, self.window_size, n_w, self.window_size)
        v_feat = v_feat.permute(0, 1, 3, 5, 4, 6, 2) # b, num_head, n_h, n_w, window, window, head_dim
        v_feat = v_feat.reshape(b, self.num_heads, n_h*n_w, window_squared, self.head_dims)
        
        
        q = self.q(q_feat)       # b, num_head, n_h*n_w, window_squared, head_dim
        k = self.k(k_feat)       # b, num_head, n_h*n_w, window_squared, head_dim
        v = self.v(v_feat)    # b, num_head, n_h*n_w, window_squared, head_dim
        
        # qk = b, num_head, n_h*n_w, window_squared, window_squared
        qk = ( torch.matmul(q, k.transpose(3, 4)) ) / np.sqrt(self.head_dims)
        qk += self.pe[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        
        if self.cyclic_shift:
            qk[:, :, -n_w:] += self.upper_lower_mask
            qk[:, :, n_w-1::n_w] += self.left_right_mask
        
        attn_weight = self.attn_drop(torch.softmax(qk, dim=-1)) # b, num_head, n_h*n_w, window_squared, window_squared
        out = torch.matmul(attn_weight, v)  # b, num_head, n_h*n_w, window_squared, head_dim
        out = self.lin_drop(self.lin(out))  # b, num_head, n_h*n_w, window_squared, head_dim
        
        # out ==> [b, num_head, n_h*n_w, window_squared, head_dim] to [b, e, h, w]
        out = out.permute(0, 1, 4, 2, 3).reshape(b, c, n_h, n_w, self.window_size, self.window_size)
        out = out.permute(0, 1, 2, 4, 3, 5).reshape(b, c, h, w)
        
        if self.cyclic_shift:
            out = self.cyclic_revert(out) # B, C, H, W
        
        out, out_sizes = self.shaping(out) # B, C, H, W -> # B, H*W, C
        
        return out, attn_weight

class BrainConv(nn.Module):
    """Some Information about MLP"""
    def __init__(self, in_features, out_features, drop_prob=0.0):
        super(BrainConv, self).__init__()
        self.c1 = nn.Conv2d(in_features, in_features*2, kernel_size=1, stride=1, padding='same')
        self.c2 = nn.Conv2d(in_features*2, out_features, kernel_size=1, stride=1, padding='same')
        self.drop = nn.Dropout(p=drop_prob)
        self.act = nn.GELU()

    def forward(self, x):
        """
        input:
            x = (B, 2C, H, W)
        return:
            out = (B, C, H, W)
        """
        x = self.drop(self.act(self.c1(x))) # (B, D, H, W)
        x = self.drop(self.act(self.c2(x))) # (B, D, H, W)
        
        return x 

class SWCATransformer(nn.Module):
    """Some Information about SWCATransformer"""
    def __init__(self, channels, num_layer, window_size, num_heads, 
                    qkv_bias=True, drop_prob=0.2, device='cuda'):
        super(SWCATransformer, self).__init__()
        
        self.num_layer = num_layer
        self.channels = channels
        
        if self.num_layer % 2 != 0:
            print('[ERROR Num layer] : Num layer must be even number') 
            raise ValueError
        
        self.msa = nn.ModuleList([])
        for _ in range(num_layer): 
            self.msa.append( 
                nn.ModuleList([
                    MultiHeadSWCA(
                        skip_channels=channels, cyclic_shift=False, window_size=window_size, 
                        num_heads=num_heads, qkv_bias=qkv_bias, attn_drop_prob=drop_prob, 
                        lin_drop_prob=drop_prob, device=device
                    ), 
                    MultiHeadSWCA(
                        skip_channels=channels, cyclic_shift=True, window_size=window_size, 
                        num_heads=num_heads, qkv_bias=qkv_bias, attn_drop_prob=drop_prob, 
                        lin_drop_prob=drop_prob, device=device
                    )
                ])
            )
            
        self.mlp = nn.ModuleList([])
        for _ in range(num_layer): 
            self.mlp.append(MLP(channels, channels*4, channels, drop_prob))
        
        self.norm1_w_q = nn.ModuleList([
                nn.LayerNorm(normalized_shape=channels)
            for _ in range(num_layer//2)
        ])
        self.norm1_w_k = nn.ModuleList([
                nn.LayerNorm(normalized_shape=channels)
            for _ in range(num_layer//2)
        ])
        self.norm1_w_v = nn.ModuleList([
                nn.LayerNorm(normalized_shape=channels)
            for _ in range(num_layer//2)
        ])
        
        
        self.norm1_sw = nn.ModuleList([
            nn.LayerNorm(normalized_shape=channels)
            for _ in range(num_layer//2)
        ])
        # self.norm2_sw = nn.ModuleList([
        #     nn.LayerNorm(normalized_shape=channels)
        #     for _ in range(num_layer//2)
        # ])
        
        # self.final_norm = nn.LayerNorm(normalized_shape=channels)

    
    def forward(self, q_feat, k_feat, v_feat):
        """
        Input: 
            features    = B, D, H, W
        Return:
            Out         = B, D, H, W
        """
        q_feat, q_feat_sizes = self.shaping(q_feat)    # B, H*W, D
        k_feat, k_feat_sizes = self.shaping(k_feat)    # B, H*W, D
        v_feat, v_feat_sizes = self.shaping(v_feat)    # B, H*W, D
        
        msa_counter = 0
        
        for i in range(self.num_layer//2): 
            
            # Todo: First part
            norm_q = self.norm1_w_q[i](q_feat)
            norm_k = self.norm1_w_k[i](k_feat)
            norm_v = self.norm1_w_v[i](v_feat)
            
            w_msa, attn_weight = self.msa[i][0](norm_q, norm_k, norm_v, q_feat_sizes)  # B, H*W, D
            w_msa = w_msa + (q_feat + k_feat)
            
            msa_counter += torch.sum(attn_weight**2)
            
            # Todo: Second part
            sw_norm1_v = self.norm1_sw[i](w_msa)                                     # B, H*W, D
            sw_msa, attn_weight = self.msa[i][1](norm_q, norm_k, sw_norm1_v, q_feat_sizes)
            sw_msa = sw_msa + w_msa
            
            msa_counter += torch.sum(attn_weight**2)
            
            mlp = self.mlp[i](sw_msa)
            
            q_feat = sw_msa.clone() + q_feat
            k_feat = sw_msa.clone() + k_feat
            v_feat = sw_msa.clone() + v_feat
            
            
        # final_calc = self.final_norm(q_feat + k_feat + v_feat)
        # Todo: Reshaping post calculation features from [B, H*W, D] to [B, C, H, W]
        sw_msa = self.reshaping(mlp, v_feat_sizes)   # B, D, H, W 
        
        return sw_msa, msa_counter


    def shaping(self, feature): 
        """
            Input = B, C, H, W
            Output = B, H*W, C or B, n_token, embed_dims
        """
        b, c, h, w = feature.shape
        feature = feature.reshape(b, c, h*w)
        feature = feature.permute(0, 2, 1)
        
        return feature, (b, c, h, w)

    def reshaping(self, feature, original_sizes):
        """
            Input = B, H*W, C or B, n_token, embed_dims
            Output = B, C, H, W
        """
        b, c, h, w = original_sizes
        feature = feature.permute(0, 2, 1)
        feature = feature.reshape(b, c, h, w)
        
        return feature 
    
if __name__ == '__main__': 
    device = torch.device('cuda')
    skip, x = torch.randn((5, 32, 24, 24)).to('cuda'), torch.randn((5, 32, 24, 24)).to('cuda')
    model = SWCATransformer(
        skip_channels=32, 
        num_layer=4, 
        window_size=8, 
        num_heads=2, 
        qkv_bias=False,
        attn_drop_prob=0.0, 
        lin_drop_prob=0.0, 
        mlp_hidden_ratio=2.0, 
        mlp_drop_prob=0.0,
        device='cuda'
    ).to('cuda')
    
    print(model(skip, x).shape)
    
    
    # sample = torch.randn((4, 128, 32, 32)).to(device)
    # trans = RegularTransformer(
    #     embed_dim=128, num_heads=8, qkv_bias=False, num_tblock=4,
    #     attn_drop_prob=0.0, lin_drop_prob=0.0, 
    #     mlp_hidden_ratio=4., mlp_drop_prob=0.0, device='cuda').to(device)
    
    # print(trans(sample).shape)