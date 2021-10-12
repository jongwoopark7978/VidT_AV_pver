import torch
from torch import nn, einsum
import numpy as np

from einops import rearrange, repeat

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)) #use 2 layers insdie feedforward. one of them is lin layer with mlp_dim
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class VidT(nn.Module):
    def __init__(self, *, num_patches, num_acc, num_frames, patch_dim, dim, depth, heads, mlp_dim, cls_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0., device='cuda:0'):
    #changed num_classes to num_acc, added num_frames
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Conv1d(in_channels=patch_dim, out_channels = dim, kernel_size = 1, stride=1) #patch_dim = flatten patch dim xp. here it is the dim after resent34 == 512. out_channels = 128 here. (maybe too narrow) 
        #try higher dim. x=(16,16,3), xp=768, D=1028, Dh(q,k,v) = 128, 8heads ???
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames*num_patches + 1, dim))#positional encoding
        self.segment_embedding = nn.Embedding(num_frames,dim)#added segment embedding layer
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Linear(dim, cls_dim)

        self.mlp_head = nn.Linear(cls_dim, num_acc)
        
        self.segment_template = torch.tensor(np.arange(num_frames)).repeat_interleave(num_patches) #added segment embedding
        self.device = device
        
    

    def forward(self, resnet_emb):
        x = self.to_patch_embedding(resnet_emb.permute(0,2,1)).permute(0,2,1)
        b, n, _ = x.shape #b:batch size n:seq length
        segments_tiled = torch.cat((torch.tensor(0).repeat(b,1), self.segment_template.repeat(b,1)), dim=-1).to(self.device) #create BERT style segment embedding 
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1) #expand seq_len + 1
        x += self.pos_embedding[:, :(n + 1)]
        x += self.segment_embedding(segments_tiled) #add segment embedding to embedding
        
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.dropout(x)

        x = self.to_latent(x)
        x = self.dropout(x)
        return self.mlp_head(x)
