#!/bin/bash

#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=20G
#SBATCH --time=23:59:00
#SBATCH --output=./slurm_out/mdcrl-%j.out
#SBATCH --error=./slurm_err/mdcrl-%j.err

module load miniconda/3
conda activate bb

export WANDB_API_KEY=1406ef3255ef2806f2ecc925a5e845e7164b5eef
wandb login

export LD_PRELOAD=/home/mila/s/sayed.mansouri-tehrani/MD-CRL/hack.so
# export WANDB_MODE=offline

# for runs more than a day, use: 1-11:59:00 (day-hour)


# ------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------- #
# --------------------------------- Synthethic Mixing --------------------------------- #
# ------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------- #
# ---------------------------- Linear Mixing, Linear Model ---------------------------- #
# ------------------------------------------------------------------------------------- #

# num_domains = 2
# python run_training.py ckpt_path=null model.optimizer.lr=0.1,0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1,0.01,0.001 datamodule.dataset.z_dim=4,6,8,10,16,20 --multirun

# num_domains = 4

# z_dim = 4
# python run_training.py ckpt_path=null model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=4 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["mila","4-domain","fixed"] --multirun

# z_dim = 8
# python run_training.py ckpt_path=null model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=8 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["mila","4-domain","fixed"] --multirun

# z_dim = 12
# python run_training.py ckpt_path=null model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=12 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["mila","4-domain","fixed"] --multirun

# z_dim = 16
# python run_training.py ckpt_path=null model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=16 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["mila","4-domain","fixed"] --multirun

# z_dim = 20
python run_training.py ckpt_path=null model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=20 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["mila","4-domain","fixed"] --multirun

# ------------------------------------------------------------------------------------- #
# -------------------------- Linear Mixing, Non-Linear Model -------------------------- #
# ------------------------------------------------------------------------------------- #

# python run_training.py ckpt_path=null model/autoencoder=fc_mlp_ae_synthetic model/optimizer=adam model.optimizer.lr=0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=4 model.wait_steps=0 model.linear_steps=1 logger.wandb.tags=["mila","non-linear-model"] --multirun


conda deactivate
module purge
