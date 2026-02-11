import jax_privacy
from jax_privacy.dpftrl_mechanisms import toeplitz
from jax_privacy.dpftrl_mechanisms import buffered_toeplitz
import jax.numpy as jnp
import numpy as np
import functools
import matplotlib.pyplot as plt
import scipy
from jax import lax
import jax

def compute_square_root(x, n):
    y = np.zeros(n)
    y[0] = np.sqrt(x[0])
    for k in range(1, n):
        y[k] = (x[k] -np.dot(y[1:k], y[1:k][::-1])) / (2 * y[0])
    return y

def Toeplitz_inverse(r):
  n = len(r)
  y = np.zeros(n)
  y[0] = 1 / r[0]
  for i in range(1, n):
    y[i] = -(y[:i] * r[i:0:-1]).sum() / r[0]
  return y

def Toeplitz_product(s1, s2):
  return np.convolve(s1, s2)[:len(s1)]

# BandInvMF https://arxiv.org/abs/2505.12128
def expected_mean_error_BandInvMF_with_momentum(inv_coef, n: int, k: int, workload_coef: jnp.array) -> float:
  inv_coef = jnp.pad(inv_coef, (0, n - inv_coef.size))
  B_norm_squared = toeplitz.mean_error(noising_coef=inv_coef, n=n, workload_coef=workload_coef, skip_checks=True)

  coef = toeplitz.inverse_coef(inv_coef)
  min_sep = n // k # assume divisible

  sensitivity_squared = toeplitz.minsep_sensitivity_squared(coef, min_sep, k, n, skip_checks=True)

  return sensitivity_squared * B_norm_squared

def init_BandInvMF_with_momentum(n, p, alpha=1, beta=0):
  x = jnp.array([1, -alpha - beta, alpha * beta] + [0]*(n-3))
  return jnp.array(compute_square_root(x, n)[:p])

def BandInv_matrix_factorization_BandInvMF_with_momentum(n, b, k, p, beta=0, alpha=1, steps=10):
    M = jnp.array(Toeplitz_product(np.array([alpha ** k for k in range(n)]), np.array([beta ** k for k in range(n)])))
    C_inv_init = init_BandInvMF_with_momentum(n, p, alpha, beta)
    C_inv_opt = toeplitz.optimize_banded_toeplitz(
      n=n,
      bands=p,
      strategy_coef=C_inv_init,
      loss_fn=functools.partial(expected_mean_error_BandInvMF_with_momentum, k=k, workload_coef=M),
      max_optimizer_steps=steps,
    )

    return np.array(C_inv_opt  / C_inv_opt[0])


# BandMF https://arxiv.org/abs/2405.15913
def expected_mean_error_BandMF_with_momentum(coef, n: int, k: int, workload_coef: jnp.array) -> float:
  coef = jnp.pad(coef, (0, n - coef.size))
  inv_coef = toeplitz.inverse_coef(coef)
  B_norm_squared = toeplitz.mean_error(noising_coef=inv_coef, n=n, workload_coef=workload_coef, skip_checks=True)

  sensitivity_squared = (coef ** 2).sum() * k

  return sensitivity_squared * B_norm_squared

def init_BandMF_with_momentum(n, p, alpha=1, beta=0):
  x = jnp.ones(n)
  return jnp.array(compute_square_root(x, n)[:p])

def Band_matrix_factorization_with_momentum(n, b, k, p, beta=0, alpha=1, steps=10):
    M = jnp.array(Toeplitz_product(np.array([alpha ** k for k in range(n)]), np.array([beta ** k for k in range(n)])))
    C_init = init_BandMF_with_momentum(n, p)
    C_opt = toeplitz.optimize_banded_toeplitz(
      n=n,
      bands=p,
      strategy_coef=C_init,
      loss_fn=functools.partial(expected_mean_error_BandMF_with_momentum, k=k, workload_coef=M),
      max_optimizer_steps=steps,
    )

    return np.array(C_opt / C_opt[0])

# Buffered Linear Toeplitz (BLT) https://aclanthology.org/2024.emnlp-industry.64.pdf
def optimize_buffered_toeplitz(n, sep, participations, buf_size, objective='mean'):
    with jax.experimental.enable_x64():
        strategy = buffered_toeplitz.optimize(
            n=n, min_sep=sep, max_participations=participations, error=objective,  min_buffers=buf_size, max_buffers=buf_size
        )
        loss_fn = buffered_toeplitz.LossFn.build_min_sep(
            n, objective, sep, participations,
        )
        loss = loss_fn.loss(strategy)
    loss = float(jnp.sqrt(loss))
    return torch.Tensor(strategy['buf_decay']), torch.Tensor(strategy['output_scale']), loss

# Function to compute c based on buffer-decay parameters
def compute_c(theta_seq, omega_seq, n):
    d = theta_seq.size(0)
    c = torch.ones(n, device=theta_seq.device)
    for i in range(1, n):
        c[i] = sum([omega_seq[j] * (theta_seq[j]**(i-1)) for j in range(d)])
    return c

def compute_BLT_factorization(num_iterations, b_min_sep, epoch):
    theta, omega, loss = optimize_buffered_toeplitz(num_iterations, b_min_sep, epoch, 4)
    return compute_c(theta, omega, num_iterations)
