# A Remedy for Negative Monte Carlo Estimated Values of KL-Divergence

## Prerequisites
`pip install -r -U requirements.txt` or `python3 setup.py install --user`

## Models

### The Poincare-VAE (`--manifold PoincareBall`):
- Curvature (`--c`): 1.0
- Prior distribution (`--prior`): `WrappedNormal` or `RiemannianNormal`
- Posterior distribution (`--posterior`): `WrappedNormal` or `RiemannianNormal`
- Decoder architecture (`--dec`):
    - `Linear` (MLP)
    - `Wrapped` (logarithm map followed by MLP),
    - `Geo` (first layer is based on geodesic distance to hyperplanes, followed by MLP)
    - `Mob` (based on Hyperbolic feed-forward layers from Ganea et al (2018))
- Encoder architecture (`--enc`): `Wrapped` or `Mob`


## Run experiments

### MNIST dataset
- curvature=0.1, latent_dim=40: ./run_vae_40_1.sh

### Custom dataset via csv file (placed in `/data`, no header, integer labels on last column)
```
python3 pvae/main.py --model csv --data-param CSV_NAME --data-size NB_FEATURES
```

