import torch
import torch.nn as nn
from collections import OrderedDict

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class mim_decoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.embed_dim = config['embed_dim']
        self.self_attn = nn.MultiheadAttention(self.embed_dim, self.embed_dim//64, batch_first=True)
        self.cross_fushion = Transformer(width=self.embed_dim,layers=4,heads=self.embed_dim//64)
        self.decoder_norm1 = nn.LayerNorm(self.embed_dim)
        self.decoder_norm2 = nn.LayerNorm(self.embed_dim)
        self.mim_head = nn.Sequential(
            OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                        ('gelu', QuickGELU()),
                        ('ln', nn.LayerNorm(self.embed_dim)),
                        ('fc', nn.Linear(self.embed_dim, 3*384*96))]))
        self.init_decoder_params()

    def init_decoder_params(self):
        scale = self.cross_fushion.width**-0.5
        proj_std = scale * ((2 * self.cross_fushion.layers)**-0.5)
        attn_std = scale
        fc_std = (2 * self.cross_fushion.width)**-0.5
        nn.init.normal_(self.self_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.self_attn.out_proj.weight, std=proj_std)
        for block in self.cross_fushion.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        nn.init.normal_(self.mim_head.dense.weight, std=fc_std)
        nn.init.normal_(self.mim_head.fc.weight, std=proj_std)

    def forward(self, image_feats, text_feats):
        image_feats = image_feats.unsqueeze(1)
        text_feats = text_feats.unsqueeze(1)
        image_feats = self.self_attn(image_feats, 
                                     text_feats, 
                                     text_feats, 
                                     need_weights=False)[0]
        fushion_feats = self.decoder_norm1(image_feats)

        x = self.cross_fushion(fushion_feats)

        x = self.decoder_norm2(x)
        x = self.mim_head(x)
        x = x.squeeze(1)
        return x