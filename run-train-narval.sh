#!/bin/bash

#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=./slurm_out/mdcrl-%j.out
#SBATCH --error=./slurm_err/mdcrl-%j.err

module load StdEnv/2020
module load coinmp
module load python/3.8
module load httpproxy
source /home/aminm/mbd/bin/activate

export WANDB_API_KEY=1406ef3255ef2806f2ecc925a5e845e7164b5eef
wandb login

export LD_PRELOAD=/home/aminm/mechanism-based-disentanglement/disentanglement_by_mechanisms/hack.so
export WANDB_MODE=offline

# for runs more than a day, use: 1-11:59:00 (day-hour)

# ------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------- #
# --------------------------------- Synthethic Mixing --------------------------------- #
# ------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------- #
# ---------------------------- Linear Mixing, Linear Model ---------------------------- #
# ------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------- #
# --------------------------------------- MNIST --------------------------------------- #
# ------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------- #
# -------------------------------- just reconstruction -------------------------- #

python run_training.py ckpt_path=null model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist datamodule.dataset.num_domains=8 model=default model.z_dim=256 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" model.autoencoder.num_channels=3 logger.wandb.tags=["mila","mnist","intrepolation"] ~callbacks.early_stopping trainer.gpus=1 model.autoencoder.upsampling_interpolation="trilinear"

# python run_training.py ckpt_path=null model.optimizer.lr=0.001,0.01 datamodule/dataset=multi_domain_mnist datamodule.dataset.num_domains=8 model=default model.z_dim=8,16,32,64 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" model.autoencoder.num_channels=3 logger.wandb.tags=["mila","mnist","reconstruction"] ~callbacks.early_stopping --multirun
# python run_training.py ckpt_path=null model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist datamodule.dataset.num_domains=8 model=default model.z_dim=512 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" model.autoencoder.num_channels=3 logger.wandb.tags=["mila","mnist","reconstruction"] ~callbacks.early_stopping
# python run_training.py ckpt_path=null model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist datamodule.dataset.num_domains=8 model=mnist_md_autoencoder model.autoencoder.num_channels=3 model.z_dim=8 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["mila","mnist","8-domain","autoencoder","test"] ~callbacks.early_stopping

# ------------------------------------------------------------------------------------- #
# -------------------------------- Disentanglement -------------------------- #


# python run_training.py ckpt_path=null model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist datamodule.dataset.num_domains=8 model=mnist_md_autoencoder model.z_dim=256 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" model.autoencoder.num_channels=3 logger.wandb.tags=["mila","mnist","intrepolation"] ~callbacks.early_stopping trainer.gpus=1 model.wait_steps=10000 model.linear_steps=1000 model.z_dim_invariant=64
# python run_training.py ckpt_path=null model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist datamodule.dataset.num_domains=8 model=mnist_md_autoencoder model.z_dim=256 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" model.autoencoder.num_channels=3 logger.wandb.tags=["mila","mnist","reconstruction"] ~callbacks.early_stopping trainer.gpus=1 model.wait_steps=10000 model.linear_steps=1000 model.z_dim_invariant=64

deactivate
module purge
