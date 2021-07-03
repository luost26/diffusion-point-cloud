import torch
import numpy as np
import math
import random
import numbers
import random
from itertools import repeat


class Center(object):
    r"""Centers node positions around the origin."""

    def __init__(self, attr):
        self.attr = attr

    def __call__(self, data):
        for key in self.attr:
            data[key] = data[key] - data[key].mean(dim=-2, keepdim=True)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class NormalizeScale(object):
    r"""Centers and normalizes node positions to the interval :math:`(-1, 1)`.
    """

    def __init__(self, attr):
        self.center = Center(attr=attr)
        self.attr = attr

    def __call__(self, data):
        data = self.center(data)

        for key in self.attr:
            scale = (1 / data[key].abs().max()) * 0.999999
            data[key] = data[key] * scale

        return data


class FixedPoints(object):
    r"""Samples a fixed number of :obj:`num` points and features from a point
    cloud.
    Args:
        num (int): The number of points to sample.
        replace (bool, optional): If set to :obj:`False`, samples fixed
            points without replacement. In case :obj:`num` is greater than
            the number of points, duplicated points are kept to a
            minimum. (default: :obj:`True`)
    """

    def __init__(self, num, replace=True):
        self.num = num
        self.replace = replace
        # warnings.warn('FixedPoints is not deterministic')

    def __call__(self, data):
        num_nodes = data['pos'].size(0)
        data['dense'] = data['pos']

        if self.replace:
            choice = np.random.choice(num_nodes, self.num, replace=True)
        else:
            choice = torch.cat([
                torch.randperm(num_nodes)
                for _ in range(math.ceil(self.num / num_nodes))
            ], dim=0)[:self.num]

        for key, item in data.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes and key != 'dense':
                data[key] = item[choice]

        return data

    def __repr__(self):
        return '{}({}, replace={})'.format(self.__class__.__name__, self.num,
                                           self.replace)


class LinearTransformation(object):
    r"""Transforms node positions with a square transformation matrix computed
    offline.
    Args:
        matrix (Tensor): tensor with shape :math:`[D, D]` where :math:`D`
            corresponds to the dimensionality of node positions.
    """

    def __init__(self, matrix, attr):
        assert matrix.dim() == 2, (
            'Transformation matrix should be two-dimensional.')
        assert matrix.size(0) == matrix.size(1), (
            'Transformation matrix should be square. Got [{} x {}] rectangular'
            'matrix.'.format(*matrix.size()))

        self.matrix = matrix
        self.attr = attr

    def __call__(self, data):
        for key in self.attr:
            pos = data[key].view(-1, 1) if data[key].dim() == 1 else data[key]

            assert pos.size(-1) == self.matrix.size(-2), (
                'Node position matrix and transformation matrix have incompatible '
                'shape.')

            data[key] = torch.matmul(pos, self.matrix.to(pos.dtype).to(pos.device))

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.matrix.tolist())


class RandomRotate(object):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval.
    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If :obj:`degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axis (int, optional): The rotation axis. (default: :obj:`0`)
    """

    def __init__(self, degrees, attr, axis=0):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis
        self.attr = attr

    def __call__(self, data):
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if self.axis == 0:
            matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        elif self.axis == 1:
            matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        else:
            matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        return LinearTransformation(torch.tensor(matrix), attr=self.attr)(data)

    def __repr__(self):
        return '{}({}, axis={})'.format(self.__class__.__name__, self.degrees,
                                        self.axis)


class AddNoise(object):

    def __init__(self, std=0.01, noiseless_item_key='clean'):
        self.std = std
        self.key = noiseless_item_key

    def __call__(self, data):
        data[self.key] = data['pos']
        data['pos'] = data['pos'] + torch.normal(mean=0, std=self.std, size=data['pos'].size())
        return data


class AddRandomNoise(object):

    def __init__(self, std_range=[0, 0.10], noiseless_item_key='clean'):
        self.std_range = std_range
        self.key = noiseless_item_key

    def __call__(self, data):
        noise_std = random.uniform(*self.std_range)
        data[self.key] = data['pos']
        data['pos'] = data['pos'] + torch.normal(mean=0, std=noise_std, size=data['pos'].size())
        return data


class AddNoiseForEval(object):

    def __init__(self, stds=[0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15]):
        self.stds = stds
        self.keys = ['noisy_%.2f' % s for s in stds]

    def __call__(self, data):
        data['clean'] = data['pos']
        for noise_std in self.stds:
            data['noisy_%.2f' % noise_std] = data['pos'] + torch.normal(mean=0, std=noise_std, size=data['pos'].size())
        return data


class IdentityTransform(object):
    
    def __call__(self, data):
        return data


class RandomScale(object):
    r"""Scales node positions by a randomly sampled factor :math:`s` within a
    given interval, *e.g.*, resulting in the transformation matrix
    .. math::
        \begin{bmatrix}
            s & 0 & 0 \\
            0 & s & 0 \\
            0 & 0 & s \\
        \end{bmatrix}
    for three-dimensional positions.
    Args:
        scales (tuple): scaling factor interval, e.g. :obj:`(a, b)`, then scale
            is randomly sampled from the range
            :math:`a \leq \mathrm{scale} \leq b`.
    """

    def __init__(self, scales, attr):
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales
        self.attr = attr

    def __call__(self, data):
        scale = random.uniform(*self.scales)
        for key in self.attr:
            data[key] = data[key] * scale
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.scales)


class RandomTranslate(object):
    r"""Translates node positions by randomly sampled translation values
    within a given interval. In contrast to other random transformations,
    translation is applied separately at each position.
    Args:
        translate (sequence or float or int): Maximum translation in each
            dimension, defining the range
            :math:`(-\mathrm{translate}, +\mathrm{translate})` to sample from.
            If :obj:`translate` is a number instead of a sequence, the same
            range is used for each dimension.
    """

    def __init__(self, translate, attr):
        self.translate = translate
        self.attr = attr

    def __call__(self, data):
        (n, dim), t = data['pos'].size(), self.translate
        if isinstance(t, numbers.Number):
            t = list(repeat(t, times=dim))
        assert len(t) == dim

        ts = []
        for d in range(dim):
            ts.append(data['pos'].new_empty(n).uniform_(-abs(t[d]), abs(t[d])))

        for key in self.attr:
            data[key] = data[key] + torch.stack(ts, dim=-1)

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.translate)


class Rotate(object):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval.
    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If :obj:`degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axis (int, optional): The rotation axis. (default: :obj:`0`)
    """

    def __init__(self, degree, attr, axis=0):
        self.degree = degree
        self.axis = axis
        self.attr = attr

    def __call__(self, data):
        degree = math.pi * self.degree / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if self.axis == 0:
            matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        elif self.axis == 1:
            matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        else:
            matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        return LinearTransformation(torch.tensor(matrix), attr=self.attr)(data)

    def __repr__(self):
        return '{}({}, axis={})'.format(self.__class__.__name__, self.degrees,
                                        self.axis)
