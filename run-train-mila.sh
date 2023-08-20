#!/bin/bash

#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=30G
#SBATCH --time=23:59:00
#SBATCH --output=./slurm_out/%j.out
#SBATCH --error=./slurm_err/%j.err

module load miniconda/3
conda activate bb

export WANDB_API_KEY=1406ef3255ef2806f2ecc925a5e845e7164b5eef
wandb login

export LD_PRELOAD=/home/mila/s/sayed.mansouri-tehrani/MD-CRL/hack.so
export WANDB_MODE=offline



# for runs more than a day, use: 1-11:59:00 (day-hour)


# ---------- SA-Mesh ---------- #

python3 run_training.py model.additional_logger.logging_interval=400 ~callbacks.visualization_callback ~callbacks.early_stopping callbacks.model_checkpoint.monitor="train_loss" logger.wandb.tags=["mila"] ckpt_path=null trainer.max_epochs=2000


deactivate
module purge
