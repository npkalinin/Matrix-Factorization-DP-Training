# Differentially Private Matrix Factorization with BSR and BISR

This repository implements a differentially private optimizer based on matrix factorization for stochastic gradient descent (SGD) with momentum and weight decay. It supports two efficient and scalable methods: **BSR** and **BISR**.

**Matrix Factorization for Private SGD.**  
In differentially private training, matrix factorization provides a principled way to reduce the noise needed for privacy. We consider a matrix of all *intermediate* models as a product of the **workload matrix** $A \in \mathbb{R}^{n \times n}$, which encodes the structure of SGD with **momentum and weight decay** across training steps, and the private gradients as the matrix $X \in \mathbb{R}^{n \times d}$, where each row corresponds to the gradient from one data point. The goal is to compute a private version of the update $AX$. Rather than adding noise directly to $AX$, we factor $A = BC$ and apply noise to $CX$ first, then postprocess it with matrix $B$ without violating privacy:

$$
\hat{AX} = B(CX + Z) \quad \text{or equivalently} \quad \widehat{AX} = A(X + C^{-1}Z)
$$

where $Z \sim \mathcal{N}(0, sI)$ is appropriately scaled Gaussian noise. In this repository, we consider two explicit factorizations:

- **[BSR (Banded Square Root)](https://arxiv.org/pdf/2405.13763)** enforces a **banded structure** for matrix $C$, which allows efficient computation and closed-form expressions with momentum and weight decay.

- **[BISR (Banded Inverse Square Root)](https://arxiv.org/pdf/2505.12128)** instead applies the **banded structure to $C^{-1}$**, improving the error in the small-memory regime.

These matrix factorizations are tailored for training deep models efficiently under differential privacy.



## 📂 Contents

- `dp_optimizer.py`: Contains the `DPMFSGD` class, implementing:
  - Matrix-factorized DP-SGD with momentum and weight decay
  - Support for `factorization_type='band'` (BSR) and `factorization_type='band-inv'` (BISR)
  - Optional Toeplitz matrix via `MF_coef`
  - Gradient clipping with Opacus' `GradSampleModule`

- `CIFAR_10_training.ipynb`: Example notebook for training on CIFAR-10 using:
  - DP-SGD
  - BSR and BISR (no amplification yet)

---
