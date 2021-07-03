import torch
from torch.nn import Module

from .common import *
from .encoders import *
from .diffusion import *
from .flow import *


class FlowVAE(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(args.latent_dim)
        self.flow = build_latent_flow(args)
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )

    def get_loss(self, x, kl_weight, writer=None, it=None):
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        batch_size, _, _ = x.size()
        # print(x.size())
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        
        # H[Q(z|X)]
        entropy = gaussian_entropy(logvar=z_sigma)      # (B, )

        # P(z), Prior probability, parameterized by the flow: z -> w.
        w, delta_log_pw = self.flow(z, torch.zeros([batch_size, 1]).to(z), reverse=False)
        log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(dim=1, keepdim=True)   # (B, 1)
        log_pz = log_pw - delta_log_pw.view(batch_size, 1)  # (B, 1)

        # Negative ELBO of P(X|z)
        neg_elbo = self.diffusion.get_loss(x, z)

        # Loss
        loss_entropy = -entropy.mean()
        loss_prior = -log_pz.mean()
        loss_recons = neg_elbo
        loss = kl_weight*(loss_entropy + loss_prior) + neg_elbo

        if writer is not None:
            writer.add_scalar('train/loss_entropy', loss_entropy, it)
            writer.add_scalar('train/loss_prior', loss_prior, it)
            writer.add_scalar('train/loss_recons', loss_recons, it)
            writer.add_scalar('train/z_mean', z_mu.mean(), it)
            writer.add_scalar('train/z_mag', z_mu.abs().max(), it)
            writer.add_scalar('train/z_var', (0.5*z_sigma).exp().mean(), it)

        return loss

    def sample(self, w, num_points, flexibility, truncate_std=None):
        batch_size, _ = w.size()
        if truncate_std is not None:
            w = truncated_normal_(w, mean=0, std=1, trunc_std=truncate_std)
        # Reverse: z <- w.
        z = self.flow(w, reverse=True).view(batch_size, -1)
        samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility)
        return samples
