import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange



# helpers functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# small helper modules
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=9):
        super().__init__()
        # self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.proj = nn.Conv1d(dim, dim_out, kernel_size, padding=kernel_size // 2)  # 修改卷积核大小为9
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, kernel_size=9):
        super().__init__()
        self.block1 = Block(dim, dim_out, kernel_size=kernel_size)  # 修改卷积核大小为9
        self.block2 = Block(dim_out, dim_out, kernel_size=kernel_size)  # 修改卷积核大小为9

        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):

        scale_shift = None
        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

        
class Gate(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_wv = nn.Conv1d(dim, hidden_dim * 2, kernel_size=7, padding=7 // 2, bias = False)
        self.sigmoid = nn.Sigmoid()
        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        wv = self.to_wv(x).chunk(2, dim = 1)
        w, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), wv)

        w = w.softmax(dim = -1)

        out = w*v

        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)
   

# model
class Cnet1D(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        learned_variance = False,
        attn_dim_head = 32,
        attn_heads = 4
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)
        # 例如，如果 dim = 64，init_dim = 128，dim_mults = (1, 2, 4, 8)，则：
        # dims = [128, 64 * 1, 64 * 2, 64 * 4, 64 * 8]
        # dims = [128, 64, 128, 256, 512]
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # 例如，使用前面的 dims 计算 in_out：
        # in_out = list(zip([128, 64, 128, 256], [64, 128, 256, 512]))
        # in_out = [(128, 64), (64, 128), (128, 256), (256, 512)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([

                ResnetBlock(dim_in, dim_in, kernel_size=9),  # 修改卷积核大小为9
                ResnetBlock(dim_in, dim_in, kernel_size=9),  # 修改卷积核大小为9
                Residual(PreNorm(dim_in, Gate(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 9, padding=4)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, kernel_size=9)  # 修改卷积核大小为9
        self.mid_gate = Residual(PreNorm(mid_dim, Gate(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, kernel_size=9)  # 修改卷积核大小为9

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([

                ResnetBlock(dim_out + dim_in, dim_out, kernel_size=9),  # 修改卷积核大小为9
                ResnetBlock(dim_out + dim_in, dim_out, kernel_size=9),  # 修改卷积核大小为9
                Residual(PreNorm(dim_out, Gate(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 9, padding=4)  # 修改卷积核大小为9
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ResnetBlock(dim * 2, dim, kernel_size=9)  # 修改卷积核大小为9
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x):

        x = self.init_conv(x)
        r = x.clone()

        h = []
        # 通过多层卷积和注意力机制进行特征提取和下采样，
        # 然后在解码器部分通过上采样和特征融合重建原始数据，适合于处理时间序列数据的生成或重建任务。
        for block1, block2, gate, downsample in self.downs:
            x = block1(x)
            h.append(x)

            x = block2(x)
            x = gate(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_gate(x)
        x = self.mid_block2(x)

        for block1, block2, gate, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x)
            x = gate(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x)

        return self.final_conv(x)



