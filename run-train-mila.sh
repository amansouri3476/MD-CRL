#!/bin/bash

#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=02:59:00
#SBATCH --output=./slurm_out/%j.out
#SBATCH --error=./slurm_err/%j.err

module load miniconda/3
conda activate bb

export WANDB_API_KEY=1406ef3255ef2806f2ecc925a5e845e7164b5eef
wandb login

export LD_PRELOAD=/home/mila/s/sayed.mansouri-tehrani/MD-CRL/hack.so
# export WANDB_MODE=offline



# for runs more than a day, use: 1-11:59:00 (day-hour)


# -------------------------- Synthetic Mixing -------------------------- #

python run_training.py ckpt_path=null model.optimizer.lr=0.01 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=0.00001
python3 run_training.py model.additional_logger.logging_interval=400 ~callbacks.visualization_callback ~callbacks.early_stopping callbacks.model_checkpoint.monitor="train_loss" logger.wandb.tags=["mila"] ckpt_path=null trainer.max_epochs=2000


python run_training.py ckpt_path=null model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=default model.autoencoder.num_channels=3 model.z_dim=32
python run_training.py ckpt_path=null model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=multi_domain_autoencoder model.autoencoder.num_channels=3 model.z_dim=32
python run_training.py ckpt_path=null model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=default model.autoencoder.num_channels=3

# ------------------------------------------------------------------------------------- #
# --------------------------------------- MNIST --------------------------------------- #
# ------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------- #
# -------------------------- Disentanglement with encoded images ---------------------- #
python run_training.py ckpt_path=null model.optimizer.lr=0.001 datamodule=mnist_encoded model=mnist_md_encoded_autoencoder model.z_dim=64 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["mila","mnist","separate"] ~callbacks.early_stopping ~callbacks.visualization_callback trainer.gpus=1 run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_multi_domain_mnist_8_256/2023-09-09_07-59-59"

deactivate
module purge
