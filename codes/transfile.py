import torch
from torch import nn

class DropPath(nn.Module):
    """Drop path

    Randomly drop the input (i.e., output zero) with some probability, per sample.
    """

    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.dropout_p = dropout_p

    def forward(self, x):

        if self.dropout_p == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.dropout_p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 0 / 1

        # Discussion: https://github.com/rwightman/pytorch-image-models/discussions/895
        output = x / keep_prob * random_tensor
        return output


class MLP(nn.Module):
    """MLP layer, usually used in Transformer"""

    def __init__(
        self,
        in_feat: int,
        mlp_ratio: float = 1.5,
        out_feat: int = 0,
        dropout_p: float = 0.0,
        act_layer: nn.Module = nn.ReLU,
        init_std: float = 1e-5
    ):
        super().__init__()

        mid_feat = int(in_feat * mlp_ratio)
        out_feat = out_feat or in_feat

        self.act = nn.Sigmoid()

        self.linear1 = nn.Linear(in_feat, mid_feat)
        nn.init.normal_(self.linear1.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.linear1.bias, mean=0.0, std=init_std)
        self.drop1 = nn.Dropout(dropout_p)

        self.linear2 = nn.Linear(mid_feat, out_feat)
        nn.init.normal_(self.linear2.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.linear2.bias, mean=0.0, std=init_std)
        self.drop2 = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.linear1(x)
        x = self.drop1(x * self.act(x))
        x = self.drop2(self.linear2(x))
        return x


class AttentionLayer(nn.Module):
    """Multi-head scaled self-attension layer"""

    def __init__(self, num_feat: int, num_heads: int = 8, qkv_bias: bool = False, dropout_p: float = 0.0, init_std: float = 1e-5):
        super().__init__()

        assert num_feat % num_heads == 0

        self.num_feat = num_feat
        self.num_heads = num_heads
        self.head_dim = num_feat // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(self.num_feat, self.num_feat * 3, bias=qkv_bias)

        nn.init.normal_(self.qkv.weight, mean=0.0, std=init_std)
        if qkv_bias:
            nn.init.normal_(self.qkv.bias, mean=0.0, std=init_std)

        self.attn_drop = nn.Dropout(dropout_p)

    def forward(self, x, mask):
        B, L, C = x.shape
        assert C == self.num_feat
        
        qkv = self.qkv(x)  # [B, L, num_feat * 3]
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)  # [B, L, 3, num_heads, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, L, head_dim]
        q, k, v = qkv.unbind(0)  # [B, num_heads, L, head_dim] * 3

        attn_mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1).unsqueeze(-1).float()
        attn_mask = attn_mask @ attn_mask.transpose(-2, -1)

        attn = q @ k.transpose(-2, -1)  # [B, num_heads, L, L]
        attn = self.attn_drop(attn)

        attn = attn * self.scale
        attn = attn.masked_fill(attn_mask==0, -1e15)
        #attn = attn.softmax(dim=-1)
        #softmax_one
        attn_max = torch.max(attn, dim=-1)[0].unsqueeze(-1)
        attn = torch.exp(attn - attn_max)
        partition = torch.sum(attn, dim=-1).unsqueeze(-1) + 0.01
        attn = attn / partition

        x = (attn @ v).transpose(1, 2)  # [B, L, num_heads, head_dim]
        x = x.reshape(B, L, self.num_feat)  # [B, L, num_feat]

        #x = self.proj(x)
        #x = self.proj_drop(x)

        return x



class TransformerBlock(nn.Module):
    def __init__(
        self,
        in_feat: int,
        out_feat: int = 0,
        num_heads: int = 8,
        head_dim: int = 0,
        qkv_bias: bool = False,
        mlp_ratio: int = 4,
        dropout_p: float = 0.0,
        droppath_p: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        init_std: float = 1e-5
    ):
        super().__init__()

        out_feat = out_feat or in_feat

        self.droppath = DropPath(droppath_p)

        self.norm1 = norm_layer(in_feat)
        self.norm2 = norm_layer(in_feat)

        self.attn = AttentionLayer(num_feat=in_feat, num_heads=num_heads, qkv_bias=qkv_bias, dropout_p=dropout_p, init_std=init_std)
        # self.attn = ReducedAttentionLayer(num_feat=in_feat, head_dim=head_dim, num_heads=num_heads, qkv_bias=qkv_bias, dropout_p=dropout_p)

        self.mlp = MLP(in_feat=in_feat, mlp_ratio=mlp_ratio, out_feat=out_feat, dropout_p=dropout_p, act_layer=act_layer, init_std=init_std)

    def forward(self, x, mask):
        attn = x + self.droppath(self.attn(self.norm1(x), mask))
        #print('after attn:', x[0,:,:])
        opt = x + self.droppath(self.mlp(self.norm2(attn)))
        #print('output:', x[0,:,:])
        #raise ValueError('stop')
        return opt 