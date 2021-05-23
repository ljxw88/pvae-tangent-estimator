# A Remedy for Negative Monte Carlo Estimated Values of KL-Divergence
This repo is code for a PyTorch implementation of Poincare Variational Auto-Encoder. KL-divergence inside Evidence Lower-Bound is replaced by Tangent/Absolute-KL-divergence.

## Prerequisites
- Install Prerequisite packages: `pip install -r -U requirements.txt`

## Models

### The Poincare-VAE (Mathieu et al (2019) [[Paper](https://arxiv.org/abs/1901.06033)], [[Repo](https://github.com/emilemathieu/pvae)]):
- Curvature (`--c`): 1.0
- Prior distribution (`--prior`): `WrappedNormal` or `RiemannianNormal`
- Posterior distribution (`--posterior`): `WrappedNormal` or `RiemannianNormal`
- Decoder architecture (`--dec`):
    - `Linear` (MLP)
    - `Wrapped` (logarithm map followed by MLP),
    - `Geo` (first layer is based on geodesic distance to hyperplanes, followed by MLP)
    - `Mob` (based on Hyperbolic feed-forward layers from Ganea et al (2018))
- Encoder architecture (`--enc`): `Wrapped` or `Mob`
- Estimator (`--est`): `tan`, `abs` or `naive` (default).

## Directory structure

```bash
README.md
data
   |-- .gitkeep
experiments
   |-- .gitkeep
pvae
   |-- __init__.py
   |-- datasets
   |   |-- __init__.py
   |   |-- datasets.py
   |-- distributions
   |   |-- __init__.py
   |   |-- ars.py
   |   |-- hyperbolic_radius.py
   |   |-- hyperspherical_uniform.py
   |   |-- riemannian_normal.py
   |   |-- wrapped_normal.py
   |-- main.py
   |-- manifolds
   |   |-- __init__.py
   |   |-- euclidean.py
   |   |-- poincareball.py
   |-- models
   |   |-- __init__.py
   |   |-- architectures.py
   |   |-- mnist.py
   |   |-- tabular.py
   |   |-- vae.py
   |-- objectives.py
   |-- ops
   |   |-- __init__.py
   |   |-- manifold_layers.py
   |-- utils.py
   |-- vis.py
requirements.txt
run_357.sh
run_all.sh
run_vae_40_1.sh
run_vae_40_2.sh
run_vae_40_3.sh
run_vae_40_4.sh
run_vae_60_1.sh
run_vae_60_2.sh
run_vae_60_3.sh
run_vae_60_4.sh
run_vae_80_1.sh
run_vae_80_2.sh
run_vae_80_3.sh
run_vae_80_4.sh
tests
   |-- __init__.py
   |-- test_hyperbolic_radius.py
   |-- test_hyperspherical_uniform.py

```

## Trainning

### MNIST dataset
- curvature=0.1, latent_dim=40: `./run_vae_40_1.sh`
- curvature=0.2, latent_dim=40: `./run_vae_40_2.sh`
- curvature=0.7, latent_dim=40: `./run_vae_40_3.sh`
- curvature=1.4, latent_dim=40: `./run_vae_40_4.sh`
- curvature=0.1, latent_dim=60: `./run_vae_60_1.sh`
- curvature=0.2, latent_dim=60: `./run_vae_60_2.sh`
- curvature=0.7, latent_dim=60: `./run_vae_60_3.sh`
- curvature=1.4, latent_dim=60: `./run_vae_60_4.sh`
- curvature=0.1, latent_dim=80: `./run_vae_80_1.sh`
- curvature=0.2, latent_dim=80: `./run_vae_80_2.sh`
- curvature=0.7, latent_dim=80: `./run_vae_80_3.sh`
- curvature=1.4, latent_dim=80: `./run_vae_80_4.sh`

### Custom dataset via csv file (placed in `/data`, no header, integer labels on last column)
`
python3 pvae/main.py --model csv --data-param CSV_NAME --data-size NB_FEATURES
`

## Acknowledgement
Special thanks to Mr. Zhenyue Qin in Australian National University; Mrs. Yang Liu and Dr. Saeed Anwar in Data61 CSIRO; Dr. Pan Ji in OPPO US Research Center.

## Additional Links
- Geoopt: Riemannian Optimization in PyTorch: [[Link](https://github.com/geoopt/geoopt)]
- Monte Carlo theory, methods and examples: [[Link](https://statweb.stanford.edu/~owen/mc/)]