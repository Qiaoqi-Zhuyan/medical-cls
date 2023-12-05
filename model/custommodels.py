# custom models
import copy

# modify from: https://github.com/sail-sg/poolformer/blob/main/models/poolformer.py
# modify from: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
import einops


@staticmethod
def depthwise_conv(in_channel, out_channel, kernel_size, stride=1, padding=0, bias=False):
    return nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=in_channel)



class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: [B, C, H, W] -> output: [B, C, H, W]
    """

    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class PatchEmbed(nn.Module):
    """
    input: [B, C, H, W] -> output: [B, C, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class PatchEmbed_swin(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        y = self.proj(x)
        print(y.shape)
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class Stem(nn.Module):
    '''
        input: [B, C, H, W] -> output: [B, C, H/stride, W/stride]
        conv -> LayerNorm
    '''

    def __init__(self, in_channels=3, patch_size=4, stride=4, padding=0, embed_dim=96, norm_layer=nn.LayerNorm):
        super(Stem, self).__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = einops.rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        x = einops.rearrange(x, "b h w c-> b c h w ")
        return x


class Downsample(nn.Module):
    '''
        input: [B, C, H, W] -> output: [B, C * 2, H // 2, W // 2]
    '''
    def __init__(self, in_channel, norm_layer=nn.LayerNorm, type="linear"):
        super(Downsample, self).__init__()
        self.norm = norm_layer(4 * in_channel)
        self.fc = nn.Linear(4 * in_channel, 2 * in_channel)
        self.conv = nn.Conv2d(in_channels=4 * in_channel, out_channels=2 * in_channel, kernel_size=1, stride=1)
        self.type = type

    def forward(self, x):
        _, _, H, W = x.shape

        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        assert type == "linear" or type == "conv", "worry type"

        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2] # -> [B, C, H//2, W//2]

        x = torch.cat([x0, x1, x2, x3], 1)

        x = einops.rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)

        if self.type == "linear":
            x = self.fc(x)
            return einops.rearrange(x, "b h w c -> b c h w")

        elif self.type == "conv":
            x = einops.rearrange(x, "b h w c -> b c h w")
            return self.conv(x)


class ConvFFN(nn.Module):
    '''
     input: [B C_in H W] -> [B C_out H W]
    '''
    def __init__(self, in_channel, out_channel, expan_ratio=4):
        super(ConvFFN, self).__init__()
        hidden_dim = int(expan_ratio * in_channel)
        self.pwconv1 = nn.Conv2d(in_channels=in_channel, out_channels=hidden_dim, kernel_size=1)
        self.pwconv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=out_channel, kernel_size=1)
        self.dwconv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, groups=in_channel)
        self.act_layer = nn.GELU()
        self.norm = nn.LayerNorm(in_channel)
        self.apply(self._init_wights)

    def _init_wights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.dwconv(x)
        x = einops.rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        x = einops.rearrange(x, "b h w c -> b c h w")
        x = self.pwconv1(x)
        x = self.act_layer(x)
        x = self.pwconv2(x)

        return x


class DWSEblock(nn.Module):
    '''
        input: [B C H W] -> hidden dim [B C/r 1 1] -> output: [B C 1 1]
    '''
    def __init__(self, in_channel, scale_ratio=4):
        super(DWSEblock, self).__init__()
        # se layer blocks implement with DWConv
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden_dim = int(in_channel * scale_ratio)
        # dw
        self.se_conv1 = nn.Conv2d(in_channels=in_channel, out_channels=hidden_dim, kernel_size=3, padding=1, groups=in_channel)
        # pw
        self.se_conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=in_channel, kernel_size=1)
        #self.se_conv3 = nn.Conv2d(in_channels=hidden_dim, out_channels=out_channel, kernel_size=1)

        self.act_layer = nn.GELU()
        #self.sigmoid = nn.Sigmoid()

        self.apply(self._init_wights)

    def _init_wights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.avg_pool(x) # [b c h w] -> [b c 1 1]
        x = self.se_conv1(x)
        x = self.act_layer(x)
        x = self.se_conv2(x)
        #x = self.sigmoid(x)
        return x

# test-> maybe never be used
class SEBlock(nn.Module):
    '''
        input: [B C H W] -> hidden [B 1 1 C//r] -> out: [B C 1 1]
    '''
    def __init__(self, in_channel, scale_ratio=4, drop=0.0):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden_dim = int(in_channel * scale_ratio)
        self.fc1 = nn.Linear(in_channel, hidden_dim)
        self.act_layer = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_channel)
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.avg_pool(x) # [b c h w] -> [b c 1 1]
        x = einops.rearrange(x, "b c h w -> b h w c")
        x = self.fc1(x)
        x = self.drop(x)
        x = self.act_layer(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = einops.rearrange(x, "b h w c->b c h w")
        return self.sigmoid(x)


class FuseMBConv(nn.Module):
    def __init__(self, in_channel, scale_ratio=4):
        super(FuseMBConv, self).__init__()
        hidden_dim = int ( scale_ratio * in_channel)

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=hidden_dim, kernel_size=3)
        self.seblock = DWSEblock(in_channel=hidden_dim, scale_ratio=scale_ratio)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=in_channel, kernel_size=1)


    def _init_wights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.seblock(x)
        x = self.conv2(x)

        x = x + residual

        return x

class ConvFormerBlock(nn.Module):
    def __init__(self, in_channel, out_channel, expand_ratio=4, norm_layer=nn.LayerNorm, drop=0.0, drop_path=0.0, use_layer_scale=True, layer_scale_init_value=1e-5, mode="conv"):
        super(ConvFormerBlock, self).__init__()
        assert mode == "conv" or mode == "attn"

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.expand_ratio = expand_ratio
        self.norm_layer = norm_layer(in_channel)

        self.use_layer_scale = use_layer_scale
        self.layer_scale_init_value = layer_scale_init_value
        self.mode = mode

        self.drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.conv = depthwise_conv(in_channel=in_channel, out_channel=in_channel, kernel_size=3)
        # ffn
        self.ffn = ConvFFN(in_channel=out_channel, out_channel=out_channel, expan_ratio=4)

    def _attn_mixer(self, x):
        pass

    def _conv_mixer(self, x):

        MBconv = FuseMBConv(in_channel=self, scale_ratio=self.expand_ratio)
        x = MBconv(x)

        return x


    def forward(self, x):

        x = x + self.drop_path(
            self._conv_mixer(self.norm_layer(self.conv(x) + x))
        )

        x = x + self.drop_path(
            self.ffn(self.norm_layer(x))
        )

        return x

def basic_blocks(dim, idx, layers,
                 mode="conv", expand_ratio=4, norm_layer=nn.LayerNorm,
                 drop=0.0, drop_path=0.0, use_layer_scale=True, layer_scale_init_value=1e-5):

    blocks = []
    for block_idx in range(layers[idx]):
        block_dp = drop_path * (block_idx + sum(layers[:idx])) / (sum(layers) - 1)

        blocks.append(ConvFormerBlock(
            in_channel=dim, out_channel=dim ,mode=mode, expand_ratio=expand_ratio, norm_layer=norm_layer,
            drop=drop, drop_path=drop_path, use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value
        ))

    blocks = nn.Sequential(*blocks)

    return blocks


class ConvFormer(nn.Module):
    def __init__(self, layers, embed_dims=None, hidden_dims=None, downsamples=None,
                 mode=None, norm_layer=nn.LayerNorm, act_layer=nn.GELU, num_classes=5,
                 in_patch_size=4, in_stride=4, in_pad=2,
                 down_patch_size=4, down_stride=2, down_pad=1,
                 drop=0.0, drop_path=0.0, use_layer_scale=True, layer_scale_init_value=1e-5, cfg = None):
        super(ConvFormer, self).__init__()

        '''
        def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        '''

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size,
            stride=in_stride,
            padding=in_pad,
            in_chans=3,
            embed_dim=embed_dims[0],
            norm_layer=nn.GroupNorm,

        )

        '''
        def basic_blocks(dim, idx, layers, 
                 mode="conv", expand_ratio=4, norm_layer=nn.LayerNorm,
                 drop=0.0, drop_path=0.0, user_layer_scale=True, layer_scale_init_value=1e-5):
        '''

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(
                dim=embed_dims[i], idx=i, layers=layers, mode=mode, expand_ratio=4,
                drop=drop, drop_path=drop_path, use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value,
            )

            network.append(stage)
            if i >= len(layers) - 1:
                break

            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling
                network.append(PatchEmbed(
                    patch_size=down_patch_size, stride=down_stride,
                    padding=down_pad,
                    in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]
                ))

        self.network = nn.ModuleList(network)

        # classifier head
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(
            embed_dims[-1], num_classes
        ) if num_classes > 0 else nn.Identity()

        self.cfg = copy.deepcopy(cfg)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []

        for idx, block in enumerate(self.network):
            x = block(x)

        return x


    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        # for image classification
        return cls_out
        

if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)
    norm = PatchEmbed(patch_size=4, stride=4, embed_dim=96)
    norm_swin = PatchEmbed_swin(patch_size=4, embed_dim=96)
    embed = Stem()
    downsample = Downsample(3)
    mconv = FuseMBConv(in_channel=3)
    y = mconv(x)
    print(f"input x : {x.shape}")
    print(f"mbconv module: {y.shape}")


