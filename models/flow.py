import types
import torch
import torch.nn as nn
import torch.nn.functional as F


class CouplingLayer(nn.Module):

    def __init__(self, d, intermediate_dim, swap=False):
        nn.Module.__init__(self)
        self.d = d - (d // 2)
        self.swap = swap
        self.net_s_t = nn.Sequential(
            nn.Linear(self.d, intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_dim, (d - self.d) * 2),
        )

    def forward(self, x, logpx=None, reverse=False):

        if self.swap:
            x = torch.cat([x[:, self.d:], x[:, :self.d]], 1)

        in_dim = self.d
        out_dim = x.shape[1] - self.d

        s_t = self.net_s_t(x[:, :in_dim])
        scale = torch.sigmoid(s_t[:, :out_dim] + 2.)
        shift = s_t[:, out_dim:]

        logdetjac = torch.sum(torch.log(scale).view(scale.shape[0], -1), 1, keepdim=True)

        if not reverse:
            y1 = x[:, self.d:] * scale + shift
            delta_logp = -logdetjac
        else:
            y1 = (x[:, self.d:] - shift) / scale
            delta_logp = logdetjac

        y = torch.cat([x[:, :self.d], y1], 1) if not self.swap else torch.cat([y1, x[:, :self.d]], 1)

        if logpx is None:
            return y
        else:
            return y, logpx + delta_logp


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, logpx=None, reverse=False, inds=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                x = self.chain[i](x, reverse=reverse)
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, logpx, reverse=reverse)
            return x, logpx


def build_latent_flow(args):
    chain = []
    for i in range(args.latent_flow_depth):
        chain.append(CouplingLayer(args.latent_dim, args.latent_flow_hidden_dim, swap=(i % 2 == 0)))
    return SequentialFlow(chain)


##################
## SpectralNorm ##
##################

POWER_ITERATION_FN = "spectral_norm_power_iteration"


class SpectralNorm(object):
    def __init__(self, name='weight', dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        self.eps = eps

    def compute_weight(self, module, n_power_iterations):
        if n_power_iterations < 0:
            raise ValueError(
                'Expected n_power_iterations to be non-negative, but '
                'got n_power_iterations={}'.format(n_power_iterations)
            )

        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim, * [d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        weight_mat = weight_mat.reshape(height, -1)
        with torch.no_grad():
            for _ in range(n_power_iterations):
                # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                # are the first left and right singular vectors.
                # This power iteration produces approximations of `u` and `v`.
                v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
        setattr(module, self.name + '_u', u)
        setattr(module, self.name + '_v', v)

        sigma = torch.dot(u, torch.matmul(weight_mat, v))
        weight = weight / sigma
        setattr(module, self.name, weight)

    def remove(self, module):
        weight = getattr(module, self.name)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight))

    def get_update_method(self, module):
        def update_fn(module, n_power_iterations):
            self.compute_weight(module, n_power_iterations)

        return update_fn

    def __call__(self, module, unused_inputs):
        del unused_inputs
        self.compute_weight(module, n_power_iterations=0)

        # requires_grad might be either True or False during inference.
        if not module.training:
            r_g = getattr(module, self.name + '_orig').requires_grad
            setattr(module, self.name, getattr(module, self.name).detach().requires_grad_(r_g))

    @staticmethod
    def apply(module, name, dim, eps):
        fn = SpectralNorm(name, dim, eps)
        weight = module._parameters[name]
        height = weight.size(dim)

        u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
        v = F.normalize(weight.new_empty(int(weight.numel() / height)).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a
        # buffer, which will cause weight to be included in the state dict
        # and also supports nn.init due to shared storage.
        module.register_buffer(fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)

        setattr(module, POWER_ITERATION_FN, types.MethodType(fn.get_update_method(module), module))

        module.register_forward_pre_hook(fn)
        return fn


def inplace_spectral_norm(module, name='weight', dim=None, eps=1e-12):
    r"""Applies spectral normalization to a parameter in the given module.
    .. math::
         \mathbf{W} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})} \\
         \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}
    Spectral normalization stabilizes the training of discriminators (critics)
    in Generaive Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.
    See `Spectral Normalization for Generative Adversarial Networks`_ .
    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectal norm
        dim (int, optional): dimension corresponding to number of outputs,
            the default is 0, except for modules that are instances of
            ConvTranspose1/2/3d, when it is 1
        eps (float, optional): epsilon for numerical stability in
            calculating norms
    Returns:
        The original module with the spectal norm hook
    Example::
        >>> m = spectral_norm(nn.Linear(20, 40))
        Linear (20 -> 40)
        >>> m.weight_u.size()
        torch.Size([20])
    """
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, dim=dim, eps=eps)
    return module


def remove_spectral_norm(module, name='weight'):
    r"""Removes the spectral normalization reparameterization from a module.
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
    Example:
        >>> m = spectral_norm(nn.Linear(40, 10))
        >>> remove_spectral_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))


def add_spectral_norm(model, logger=None):
    """Applies spectral norm to all modules within the scope of a CNF."""

    def apply_spectral_norm(module):
        if 'weight' in module._parameters:
            if logger: logger.info("Adding spectral norm to {}".format(module))
            inplace_spectral_norm(module, 'weight')

    def find_coupling_layer(module):
        if isinstance(module, CouplingLayer):
            module.apply(apply_spectral_norm)
        else:
            for child in module.children():
                find_coupling_layer(child)

    find_coupling_layer(model)


def spectral_norm_power_iteration(model, n_power_iterations=1):

    def recursive_power_iteration(module):
        if hasattr(module, POWER_ITERATION_FN):
            getattr(module, POWER_ITERATION_FN)(n_power_iterations)

    model.apply(recursive_power_iteration)

