from math import ceil
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import ELU
from torch.nn import BatchNorm1d as BN
from torch.nn import Conv1d
from torch.nn import Linear as L
from torch.nn import Sequential as S


class Reshape(torch.nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        """"""  # noqa: D419
        x = x.view(*self.shape)
        return x

    def __repr__(self) -> str:
        shape = ', '.join([str(dim) for dim in self.shape])
        return f'{self.__class__.__name__}({shape})'


class XConv(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, dim: int = 3,
                 kernel_size: int = 32, hidden_channels: Optional[int] = None,
                 dilation: int = 1, bias: bool = True, num_workers: int = 1):
        super().__init__()

        self.in_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels // 4
        assert hidden_channels > 0
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.num_workers = num_workers

        C_in, C_delta, C_out = in_channels, hidden_channels, out_channels
        D, K = dim, kernel_size

        self.mlp1 = S(
            L(dim, C_delta),
            ELU(),
            BN(C_delta),
            L(C_delta, C_delta),
            ELU(),
            BN(C_delta),
            Reshape(-1, K, C_delta),
        )

        self.mlp2 = S(
            L(D * K, K**2),
            ELU(),
            BN(K**2),
            Reshape(-1, K, K),
            Conv1d(K, K**2, K, groups=K),
            ELU(),
            BN(K**2),
            Reshape(-1, K, K),
            Conv1d(K, K**2, K, groups=K),
            BN(K**2),
            Reshape(-1, K, K),
        )

        C_in = C_in + C_delta
        depth_multiplier = int(ceil(C_out / C_in))
        self.conv = S(
            Conv1d(C_in, C_in * depth_multiplier, K, groups=C_in),
            Reshape(-1, C_in * depth_multiplier),
            L(C_in * depth_multiplier, C_out, bias=bias),
        )

    def forward(self, x: Optional[Tensor], pos: Tensor):
        """
        Args:
            x: (B, N, F_in): Node features.
            pos: (B, N, D): Node positions.
        Returns:
            (B, N, F_out): Transformed node features.
        """
        B, N, D = pos.size()

        # (1) Compute the k-NN graph using topk.
        pw_dist = torch.cdist(pos, pos)  # (B, N, N)
        knn_idx = torch.topk(pw_dist, self.kernel_size * self.dilation, largest=False)[1]  # (B, N, K * dilation)

        if self.dilation > 1:
            knn_idx = knn_idx[:, :, ::self.dilation]  # (B, N, K)

        pos_i = pos[:, :, None, :]
        pos_j = pos[:, None, :, :].repeat(1, N, 1, 1)  # (B, N, N, D)
        pos_j = torch.gather(pos_j, 2, knn_idx[:, :, :, None].expand(-1, -1, -1, D))
        d = pos_i - pos_j  # (B, N, K, D)

        x_star = self.mlp1(d.reshape(B * N * self.kernel_size, -1)).reshape(B, N, self.kernel_size, -1)  # (B, N, K, C_delta)
        if x is not None:
            x = x[:, None, :, :].repeat(1, N, 1, 1)  # (B, N, 1, F_in)
            x = torch.gather(x, 2, knn_idx[:, :, :, None].expand(-1, -1, -1, self.in_channels))
            x_star = torch.cat([x_star, x], dim=-1)  # (B, N, K, C_delta + F_in)

        x_star = x_star.reshape(B * N, self.kernel_size, -1)
        x_star = x_star.transpose(1, 2).contiguous()  # (B * N, C_delta + C_in, K)

        transform_matrix = self.mlp2(d.reshape(B * N, self.kernel_size * D))  # (B * N, K, K)

        x_transformed = torch.matmul(x_star, transform_matrix)  # (B * N, C_delta + C_in, K)

        out = self.conv(x_transformed)  # (B * N, C_out)
        out = out.view(B, N, -1).contiguous()  # (B, N, C_out)
        return out


class PointCNNEncoder(torch.nn.Module):
    def __init__(self, zdim, input_dim=3):
        super().__init__()
        assert input_dim == 3, "Only 3D point clouds are supported."
        self.conv1 = XConv(0, 48, dim=3, kernel_size=8, hidden_channels=32)
        self.conv2 = XConv(48, 96, dim=3, kernel_size=12, hidden_channels=64,
                           dilation=2)
        self.conv3 = XConv(96, 192, dim=3, kernel_size=16, hidden_channels=128,
                           dilation=2)
        self.conv4 = XConv(192, 384, dim=3, kernel_size=16,
                           hidden_channels=256, dilation=2)

        self.lin1 = L(384, 256)
        self.lin2 = L(256, 128)
        self.lin3 = L(128, zdim * 2)

    def forward(self, pos):
        x = F.relu(self.conv1(None, pos))

        randidx = torch.randperm(x.size(1))[:x.size(1) // 2]
        x = x[:, randidx, :]
        pos = pos[:, randidx, :]

        x = F.relu(self.conv2(x, pos))

        randidx = torch.randperm(x.size(1))[:x.size(1) // 2]
        x = x[:, randidx, :]
        pos = pos[:, randidx, :]

        x = F.relu(self.conv3(x, pos))
        x = F.relu(self.conv4(x, pos))

        x = x.mean(dim=1)  # (B, C)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        m, v = x.chunk(2, dim=-1)
        return m, v


if __name__ == '__main__':
    pc = torch.randn(2, 1024, 3)
    net = PointCNNEncoder(128)
    out = net(pc)
    print(out)
