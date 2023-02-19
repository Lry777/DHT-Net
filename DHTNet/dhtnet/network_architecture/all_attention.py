import torch
import torch.nn as nn


class Spartial_Attention3d(nn.Module):

    def __init__(self, kernel_size):
        super(Spartial_Attention3d, self).__init__()

        assert kernel_size % 2 == 1, "kernel_size = {}".format(kernel_size)
        padding = (kernel_size - 1) // 2

        self.__layer = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = torch.cat([avg_mask, max_mask], dim=1)

        mask = self.__layer(mask)
        return x * mask


class Channel_Attention3d(nn.Module):

    def __init__(self, channel, r):
        super(Channel_Attention3d, self).__init__()

        self.__avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.__max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))

        self.__fc = nn.Sequential(
            nn.Conv3d(channel, channel//r, 1, bias=False),
            nn.LeakyReLU(True),
            nn.Conv3d(channel//r, channel, 1, bias=False),
        )
        self.__sigmoid = nn.Sigmoid()


    def forward(self, x):
        y1 = self.__avg_pool(x)
        y1 = self.__fc(y1)

        y2 = self.__max_pool(x)
        y2 = self.__fc(y2)

        y = self.__sigmoid(y1+y2)
        return x * y

class Gate(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Gate, self).__init__()
        self._w = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm3d(out_channels)
        )
        self.relu = nn.LeakyReLU(inplace=True)
        self.psi = nn.Sequential(
            nn.Conv3d(out_channels, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        w1 = self._w(x1)
        w2 = self._w(x2)
        psi = self.relu(w1 + w2)
        psi = self.psi(psi)
        return x2 * psi


class FocalModulation(nn.Module):
    """ Focal Modulation

    Args:
        dim (int): Number of input channels.
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        focal_factor (int, default=2): Step to increase the focal window
        use_postln (bool, default=False): Whether use post-modulation layernorm
    """

    def __init__(self, dim, proj_drop=0., focal_level=2, focal_window=7, focal_factor=2, use_postln=False):

        super().__init__()
        self.dim = dim

        # specific args for focalv3
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor
        self.use_postln = use_postln

        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=True)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        if self.use_postln:
            self.ln = nn.LayerNorm(dim)

        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim,
                              padding=kernel_size // 2, bias=False),
                    nn.GELU(),
                )
            )

    def forward(self, x):
        """ Forward function.

        Args:
            x: input features with shape of (B, H, W, C)
        """
        B, nH, nW, C = x.shape
        print(x.shape)
        x = self.f(x)
        print(x.shape)
        x = x.permute(0, 3, 1, 2).contiguous()
        print(x.shape)
        q, ctx, gates = torch.split(x, (C, C, self.focal_level + 1), 1)
        print(q.shape, ctx.shape, gates.shape)

        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * gates[:, l:l + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]

        x_out = q * self.h(ctx_all)
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln:
            x_out = self.ln(x_out)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out






