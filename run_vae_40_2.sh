#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=20GB

module load pytorch/1.4.0-py36-cuda90
module load torchvision/0.5.0-py36

#pip install geoopt
#pip install git+https://github.com/geoopt/geoopt.git
python3 pvae/main.py --model mnist --manifold PoincareBall --c 0.2  --latent-dim 40 --hidden-dim 600 --prior WrappedNormal --posterior WrappedNormal --dec Geo     --enc Wrapped --lr 5e-4 --epochs 80 --save-freq 80 --batch-size 128 --iwae-samples 5000

