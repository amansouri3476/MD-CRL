#!/bin/bash

#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:59:00
#SBATCH --output=./slurm_out/%j.out
#SBATCH --error=./slurm_err/%j.err

module load miniconda/3
conda activate mdcrl

export WANDB_API_KEY=1406ef3255ef2806f2ecc925a5e845e7164b5eef
wandb login

export LD_PRELOAD=/home/mila/s/sayed.mansouri-tehrani/MD-CRL/hack.so
# export WANDB_MODE=offline



# for runs more than a day, use: 1-11:59:00 (day-hour)

# ------------------------------------------------------------------------------------- #
# --------------------------------- Synthetic Mixing ---------------------------------- #
# ------------------------------------------------------------------------------------- #

# ------------------------- Beta VAE Linear Mixing, Uniform --------------------------- #
# d32, domain16
# python run_training.py ckpt_path=null trainer.accelerator="cpu" trainer.devices="auto" model=beta_vae model.beta=1.0 model.optimizer.lr=0.001 model.r2_fit_intercept=False datamodule=mixing datamodule.batch_size=1024 datamodule.dataset.linear=True datamodule.dataset.z_dim=32 model.z_dim_invariant_data=16 datamodule.dataset.num_domains=16 logger.wandb.tags=["linear-uniform-beta"] ~callbacks.early_stopping ~callbacks.visualization_callback model/autoencoder=linear_ae_synthetic model.save_encoded_data=False datamodule.dataset.correlated_z=False seed=1235,4256,49685,7383,9271 --multirun

# ------------------------ Beta VAE Linear Mixing, Correlated ------------------------- #
# d32, domain16
# python run_training.py ckpt_path=null trainer.accelerator="cpu" trainer.devices="auto" model=beta_vae model.beta=1.0 model.optimizer.lr=0.001 model.r2_fit_intercept=False datamodule=mixing datamodule.batch_size=1024 datamodule.dataset.linear=True datamodule.dataset.z_dim=32 model.z_dim_invariant_data=16 datamodule.dataset.num_domains=16 logger.wandb.tags=["linear-corr-beta"] ~callbacks.early_stopping ~callbacks.visualization_callback model/autoencoder=linear_ae_synthetic model.save_encoded_data=False datamodule.dataset.correlated_z=True datamodule.dataset.corr_prob=0.5 seed=1235,4256,49685,7383,9271 --multirun


# ------------------------- Beta VAE Polynomial Mixing, Uniform --------------------------- #
# d14, domain16, p3
# python run_training.py ckpt_path=null trainer.accelerator="cpu" trainer.devices="auto" model=beta_vae model.beta=1.0 model.optimizer.lr=0.001 model.z_dim_invariant_data=7 model/autoencoder=linear_ae_nc datamodule=mixing_encoded datamodule.batch_size=1024 logger.wandb.tags=["poly-uniform-beta","p3"] ~callbacks.early_stopping ~callbacks.visualization_callback model.save_encoded_data=False run_path="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/multiruns/autoencoder_synthetic_mixing_linear_False_16_14_p3/2023-10-09_18-58-08/datamodule.dataset.polynomial_degree=3,datamodule.dataset.z_dim=14,model/autoencoder=poly_ae,model=mixing_synthetic/'" seed=1235,4256,49685,7383,9271 --multirun

# ------------------------ Beta VAE Polynomial Mixing, Correlated ------------------------- #
# d14, domain16, p3
# python run_training.py ckpt_path=null trainer.accelerator="cpu" trainer.devices="auto" model=beta_vae model.beta=1.0 model.optimizer.lr=0.001 model.z_dim_invariant_data=7 model/autoencoder=linear_ae_nc datamodule=mixing_encoded datamodule.batch_size=1024 logger.wandb.tags=["poly-corr-beta","p3"] ~callbacks.early_stopping ~callbacks.visualization_callback model.save_encoded_data=False run_path="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/multiruns/autoencoder_synthetic_mixing_linear_False_16_14_p3/2023-10-09_20-40-39/datamodule.dataset.polynomial_degree=3,datamodule.dataset.z_dim=14,model/autoencoder=poly_ae,model=mixing_synthetic/'" seed=1235,4256,49685,7383,9271 --multirun


# ------------------------------------------------------------------------------------- #
# --------------------------------------- Balls --------------------------------------- #
# ------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------- #
# --------------------- Beta-VAE Disentanglement with encoded images ------------------ #

# ------------------------------- iv=1,sp=1, 16 domains ------------------------------- #

# resnet18 128 bn-enc non-overlapping dataset, uniform
# python run_training.py trainer.accelerator="cpu" trainer.devices="auto" model=beta_vae model.beta=1.0 model.optimizer.lr=0.001 model.z_dim_invariant_data=2 datamodule=balls_encoded datamodule.batch_size=1024 logger.wandb.tags=["balls-uniform-beta"] ~callbacks.early_stopping ~callbacks.visualization_callback run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_md_balls_128_iv_1_sp_1/2023-10-05_07-21-05/" model.z_dim=128 model/autoencoder=mlp_ae_balls ckpt_path=null model.save_encoded_data=False seed=1235,4256,49685,7383,9271 --multirun

# resnet18 128 bn-enc non-overlapping dataset, correlated
# python run_training.py trainer.accelerator="cpu" trainer.devices="auto" model=beta_vae model.beta=1.0 model.optimizer.lr=0.001 model.z_dim_invariant_data=2 datamodule=balls_encoded datamodule.batch_size=1024 logger.wandb.tags=["balls-corr-beta"] ~callbacks.early_stopping ~callbacks.visualization_callback run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_md_balls_128_iv_1_sp_1/2023-10-09_07-31-12/" model.z_dim=128 model/autoencoder=mlp_ae_balls ckpt_path=null model.save_encoded_data=False seed=1235,4256,49685,7383,9271 --multirun


conda deactivate
module purge
