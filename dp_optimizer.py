import torch
import numpy as np
from torch.optim import Optimizer

class DPMFSGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, weight_decay=1, l2_norm_clip=1, noise_multiplier=1,
                 batch_size=500, iterations_number=100, b_min_sep=10, band_width=1, factorization_type='band', MF_coef=None, use_amplification=False, device='cpu'):
        # Initialize parameters
        defaults = dict(lr=lr)
        super(DPMFSGD, self).__init__(params, defaults)
        self.device = device
        self.momentum = momentum
        self.weight_decay = weight_decay

        """ Differential privacy parameters """
        self.l2_norm_clip = l2_norm_clip
        print(self.l2_norm_clip)
        self.noise_multiplier = noise_multiplier
        self.batch_size = batch_size
        self.running_noise_buffer = []

        """ Multi-Epoch participation"""
        self.iterations_number = iterations_number
        self.b_min_sep = b_min_sep
        self.k = (self.iterations_number + self.b_min_sep - 1) // self.b_min_sep

        """ Matrix Factorization"""
        self.matrix_factorization_coefficients = torch.zeros(band_width, device=device)
        self.C_sens = None
        self.band_width = band_width
        self.factorization_type = factorization_type
        self.generate_matrix_factorization(MF_coef)
        print("C matrix sensitivity", self.C_sens)
        if use_amplification:
          self.C_sens = 1

        # Initialize first moment (m) and second moment (v) for all parameters

        for group in self.param_groups:
          noise_buffer_size = self.band_width - (self.factorization_type == 'band')
          group['accum_grads'] = [torch.zeros_like(param.data, device=device) if param.requires_grad else None for param in group['params']]
          group['grad_momentum'] = [torch.zeros_like(param.data, device=device) if param.requires_grad else None for param in group['params']]
          group['noise_buffer'] = [torch.zeros([noise_buffer_size] + list(param.data.shape), device=device) if param.requires_grad else None for param in group['params']]


    def generate_matrix_factorization(self, MF_coef):

      def compute_square_root(x, n):
        y = torch.zeros_like(x)
        y[0] = torch.sqrt(x[0])
        for k in range(1, n):
            y[k] = (x[k] -torch.dot(y[1:k], y[1:k].flip(0))) / (2 * y[0])
        return y

      def sequence_inverse(r, n):
        y = torch.zeros(n, device=self.device)
        y[0] = 1 / r[0]
        for i in range(1, n):
          y[i] = -(y[:i].flip(0)[:min(i, len(r) - 1)] * r[1:min(i + 1, len(r))]).sum() / r[0]
        return y

      def compute_sensitivity(c, b, k, n):
        c_clone = np.pad(np.array(c.detach().cpu()), (0, n - len(c)), 'constant')
        c_sum = np.zeros_like(c_clone)
        for i in range(k):
          c_sum[b * i:] += c_clone[:(len(c_clone) - b * i)]
        sens = np.sqrt((c_sum ** 2).sum())
        return float(sens)

      if MF_coef is not None:
        self.matrix_factorization_coefficients = torch.tensor(MF_coef, device=self.device)
      else:
        self.matrix_factorization_coefficients = compute_square_root(torch.tensor([(self.weight_decay ** (k + 1) - self.momentum ** (k + 1)) / (self.weight_decay - self.momentum) for k in range(self.iterations_number)], device=self.device), self.iterations_number)

      if self.factorization_type == 'band-inv':
        if MF_coef is None:
          self.matrix_factorization_coefficients = sequence_inverse(self.matrix_factorization_coefficients, self.iterations_number)[:self.band_width]
        else:
          self.matrix_factorization_coefficients = self.matrix_factorization_coefficients[:self.band_width]
        self.C_sens = compute_sensitivity(sequence_inverse(self.matrix_factorization_coefficients, self.iterations_number), self.b_min_sep, self.k, self.iterations_number)
      else:
        self.matrix_factorization_coefficients = self.matrix_factorization_coefficients[:self.band_width]
        self.C_sens = compute_sensitivity(self.matrix_factorization_coefficients, self.b_min_sep, self.k, self.iterations_number)

    def generate_noise(self, noise_buffer, grads, coeff):
      """ Add noise when C is banded"""
      new_noise = self.l2_norm_clip * self.C_sens * self.noise_multiplier * torch.randn_like(grads, device=self.device) / self.batch_size
      if len(noise_buffer) == 0 or self.band_width == 1:
        return new_noise, noise_buffer

      if self.factorization_type == 'band':
        inner_shape = noise_buffer.shape[1:]
        corr_new_noise =  1 / coeff[0] * (new_noise - (coeff[1:] @ noise_buffer.view(len(noise_buffer), -1)).view(*inner_shape))
        noise_buffer = torch.cat([corr_new_noise.unsqueeze(0), noise_buffer[:-1]], dim=0)
        return corr_new_noise, noise_buffer

      noise_buffer = torch.cat([new_noise.unsqueeze(0), noise_buffer[:-1]], dim=0)
      inner_shape = noise_buffer.shape[1:]
      return (coeff @ noise_buffer.view(len(noise_buffer), -1)).view(*inner_shape), noise_buffer

    def zero_microbatch_grad(self):
      for group in self.param_groups:
          for param in group['params']:
              if param.grad is not None:
                  param.grad.zero_()
          
              param.grad_sample = None

    def microbatch_step(self):
        total_norm = None

        # Calculate the total L2 norm of gradients
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad and param.grad is not None:
                    grad_samples = param.grad_sample 

                    batch_size = grad_samples.shape[0]

                    if total_norm is None:
                        total_norm = torch.zeros(batch_size, device=self.device)

                    # Compute per-sample norm
                    total_norm += torch.norm(grad_samples.view(batch_size, -1), dim=1) ** 2

        total_norm = total_norm ** 0.5
        clip_coef = (self.l2_norm_clip / (total_norm + 1e-6)).clamp(max=1.0)

        for group in self.param_groups:
          for param, accum_grad in zip(group['params'], group['accum_grads']):
            if not param.requires_grad:
                continue
                
            grad_samples = param.grad_sample
            clipped_grads = grad_samples * clip_coef.view(-1, *([1] * (grad_samples.dim() - 1)))

            # Average across samples and accumulate
            accum_grad.add_(clipped_grads.sum(dim=0))


    @torch.no_grad()
    def step(self, *args, **kwargs):
        """Update parameters based on accumulated gradients with added noise for privacy."""

        for group in self.param_groups:
            lr = group['lr']

            for ind, param in enumerate(group['params']):
                if param.requires_grad:
                    # Get state for this parameter
                    accum_grads = group['accum_grads'][ind] / self.batch_size

                    # Add noise to the accumulated gradients
                    noise1, group['noise_buffer'][ind] = self.generate_noise(group['noise_buffer'][ind], accum_grads, self.matrix_factorization_coefficients)
                    assert np.all(noise1.shape == accum_grads.shape)

                    noisy_grad = accum_grads + noise1
                    group['grad_momentum'][ind] = group['grad_momentum'][ind] * self.momentum + (1 - self.momentum) * noisy_grad

                    # Update parameters
                    param.data = param.data * self.weight_decay - group['grad_momentum'][ind] * lr

                    # Zero out the accumulated gradients after each minibatch step
                    group['accum_grads'][ind].zero_()
