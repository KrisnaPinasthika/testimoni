import torch 
import torch.nn as nn 
from transformers import SwinModel, SwinConfig, UperNetConfig, UperNetForSemanticSegmentation

class BinRangeModule(nn.Module):
    """Some Information about BinRangeModule"""
    def __init__(self, embedding_dim, n_bins, norm='linear'):
        super(BinRangeModule, self).__init__()
        self.norm = norm
        self.conv_ram = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1) # range attention maps
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                        nn.LeakyReLU(),
                                        nn.Linear(256, 256),
                                        nn.LeakyReLU(),
                                        nn.Linear(256, n_bins))
    def forward(self, x):
        """
            Input:
                X : B, C, H, W
            Output: 
                Bins, Attention Range = (1, bins), (B, C, H, W)
        """
        range_attention_maps = self.conv_ram(x)  # B, C, H, W
        
        y = x.mean(dim=[2, 3])  # B, C
        y = self.regressor(y)   # B, C
        
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

class SwinAdaBin(nn.Module):
    """Some Information about SwinAdabin"""
    def __init__(self,  n_bins=100, min_val=0.1, max_val=10, norm='linear', device='cuda'):
        super(SwinAdaBin, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        
        backbone = SwinModel.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
        backbone_config = backbone.config
        backbone_config.out_features = ["stage1", "stage2", "stage3", "stage4"]

        config = UperNetConfig(backbone_config=backbone_config)
        config.num_labels = 128
        self.model = UperNetForSemanticSegmentation(config).to('cuda')
        
        self.adaptive_bins_layer = BinRangeModule(
            embedding_dim=128, 
            n_bins=n_bins,
            norm=norm
        )
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1)
        )
        
    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.model.backbone.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.model.decode_head, self.model.auxiliary_head, self.adaptive_bins_layer, self.conv_out]
        for m in modules:
            yield from m.parameters()

    def forward(self, x):
        # Todo: change list to dict
        x = {'pixel_values': x}
        
        enc_dec_pred = self.model(**x)
        enc_dec_pred = enc_dec_pred.logits
        
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(enc_dec_pred)
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
    model = SwinAdaBin(n_bins=100, min_val=0.1, max_val=10, norm='linear', device=device).to(device)
    sample = torch.randn((4, 3, 480, 640)).to(device)
    
    bin, pred = model(sample)
    
    print(bin.shape)
    print(pred.shape)
    
    print(model.get_1x_lr_params())