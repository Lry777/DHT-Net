import torch
import torch.nn as nn
from torch.nn import init
from dhtnet.network_architecture.Adaptive_Conv import DARConv3d

class DWConv3d_IN(nn.Module):
    """Depthwise Separable Convolution with IN module."""
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        norm_layer=nn.InstanceNorm3d,
        act_layer=nn.LeakyReLU,
    ):
        super().__init__()

        # dw
        self.dwconv = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            (kernel_size - 1) // 2,
            groups=out_ch,
            bias=False,
        )
        # pw-linear
        self.pwconv = nn.Conv3d(out_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = norm_layer(out_ch)
        self.act = act_layer(inplace=True) if act_layer is not None else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

    def forward(self, x):
        """
        foward function
        """
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

class ResBlock(nn.Module):
    """Residual block for convolutional local feature."""
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.LeakyReLU,
        norm_layer=nn.InstanceNorm3d,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels=in_features,out_channels=hidden_features,kernel_size=3,padding=1),
                                   nn.LeakyReLU(inplace=True),
                                   )
        self.dwconv = nn.Conv3d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=False,
            groups=hidden_features,
        )
        self.norm = norm_layer(hidden_features)
        self.act = act_layer(inplace=True)
        self.conv2 = nn.Sequential(nn.Conv3d(in_channels=in_features,out_channels=hidden_features,kernel_size=3,padding=1),
                                   nn.Identity(),
                                   )

    def forward(self, x):
        """foward function"""
        identity = x
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.norm(feat)
        feat = self.act(feat)
        feat = self.conv2(feat)

        return identity + feat

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, D ,H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.LeakyReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.norm = nn.LayerNorm(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, D, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, D, H, W) # conv3d 提供位置信息
        x = self.act(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            # self.sr = DARConv3d(dim, dim, kernel_size=sr_ratio ,region_num=4,stride=sr_ratio)
            # self.sr = IACB(dim=dim, N=1, K=3)
            self.norm = nn.LayerNorm(dim)


    def forward(self, x, D, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, D, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.LeakyReLU, norm_layer=nn.LayerNorm, sr_ratio=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, D ,H, W):
        # x = x + self.attn(self.norm1(x), D, H, W)
        x = x + self.attn(self.norm1(x), D, H, W)
        # x = self.attn(self.norm1(x), D, H, W)
        x = x + self.mlp(self.norm2(x), D, H, W)

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, in_chans=320, embed_dim=320,kernel_size=3,padding = 1,  is_pool = False,dropout = 0.):
        super().__init__()
        if is_pool:
            self.proj = nn.Sequential(
                                      nn.MaxPool3d(2),
                                      # nn.Conv3d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1),
                                      # DWConv3d_IN(in_ch=in_chans, out_ch= embed_dim,kernel_size=3, stride=2),
                                      DARConv3d(in_chans, embed_dim, kernel_size=kernel_size, region_num=8, stride=1, padding=padding),
                                      nn.LeakyReLU(inplace=True),
                                      nn.InstanceNorm3d(embed_dim)
                                      )
        else:
            self.proj = DARConv3d(in_chans, embed_dim, kernel_size=3, region_num=8, stride=1, padding=1)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        # x:(B, C, D ,H, W)
        x = self.proj(x)
        _, _, D, H, W = x.shape
        # x = x.flatten(2).transpose(1, 2) # x:(B, N ,C)
        x = self.dropout(x)

        return x

class DHTransformer(nn.Module):
    def __init__(self, in_chans=1, out_chans=32, paths = 3,
                 num_heads=8, mlp_ratios=2, is_pool = True, qkv_bias=False, qk_scale=None,
                 attn_drop_rate=0., norm_layer=nn.LayerNorm,
                 sr_ratios=4):
        super().__init__()
        self.paths = paths
        self.num_heads = num_heads
        self.sr_ratios = sr_ratios
        embed_dims = out_chans//2

        # Patch embeddings.
        # k = [1,3,5]
        # p = [0,1,2]
        self.patch_embed_stages = nn.ModuleList([
            OverlapPatchEmbed( in_chans=in_chans,embed_dim=embed_dims, kernel_size=idx*2+1,padding = idx , is_pool = is_pool) for idx in range(self.paths)
        ])
        # Attn
        self.attn = nn.ModuleList([
            Block(dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop_rate, norm_layer=norm_layer,
                sr_ratio=sr_ratios) for idx in range(self.paths)
        ])
        self.norm = nn.LayerNorm(embed_dims)
        # concat
        self.local_feature = ResBlock(in_features=embed_dims, out_features=embed_dims)
        self.cat = nn.Sequential(nn.Conv3d(embed_dims * (paths + 1),out_chans,kernel_size=3,stride=1,padding=1),
                                 nn.InstanceNorm3d(out_chans),
                                 nn.LeakyReLU(inplace=True)
                                 )

    def forward_features(self, x):
        emb_outputs = []
        for idx in range(self.paths):
            emb_output = self.patch_embed_stages[idx](x)
            emb_outputs.append(emb_output)

        att_outputs = [self.local_feature(emb_outputs[-1])]
        for x_e ,attn in zip(emb_outputs,self.attn):
            B, _,D, H, W = x_e.shape
            x_e = x_e.flatten(2).transpose(1, 2)
            x_e = attn(x_e, D, H, W)
            x_e = self.norm(x_e)
            x_e = x_e.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
            att_outputs.append(x_e)

        out_concat = torch.cat(att_outputs,dim=1)
        x = self.cat(out_concat)
        return x


    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x


if __name__ == "__main__":
    # input_arr = torch.rand((2,320, 8, 8)).cuda()
    # input_arr = input_arr.permute(0, 2, 3, 1).contiguous()

    input_arr1 = torch.rand((2, 320, 8, 8, 8)).cuda()
    input_arr1 = input_arr1.permute(0, 2, 3, 4, 1).contiguous()
    # conv = nn.Conv3d(320,320,kernel_size=(3,3,3),padding=1)
    # out = conv(input_arr)
    # print(out.shape)
    print('input',input_arr1.shape)
    # trans = MPTransformer(in_chans=320,out_chans=320,paths=3).cuda()
    # focal_b = FocalModulation3d(dim=320).cuda()
    # out = trans(input_arr)
    input_arr1 = input_arr1.flatten(2).transpose(1, 2)
    # out = focal_b(input_arr1,8,8,8)
    # print(out.shape)
