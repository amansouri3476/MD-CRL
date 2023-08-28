#!/bin/bash

#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=23:59:00
#SBATCH --output=./slurm_out/mdcrl-%j.out
#SBATCH --error=./slurm_err/mdcrl-%j.err

module load StdEnv/2020
module load coinmp
module load python/3.8
module load scipy-stack
module load httpproxy
source /home/aminm/mbd/bin/activate

export WANDB_API_KEY=1406ef3255ef2806f2ecc925a5e845e7164b5eef
wandb login

export LD_PRELOAD=/home/aminm/mechanism-based-disentanglement/disentanglement_by_mechanisms/hack.so
# export WANDB_MODE=offline


# ------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------- #
# --------------------------------- Synthethic Mixing --------------------------------- #

# num_domains = 4
python run_training.py ckpt_path=null model/optimizer=adam,adamw model.optimizer.lr=0.1,0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1,0.01 datamodule.dataset.z_dim=4,8,12,16,20 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["mila","4-domain","fixed"] --multirun
# python run_training.py ckpt_path=null model.optimizer.lr=0.01 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=0.1 datamodule.dataset.z_dim=8 model.wait_steps=500 model.linear_steps=1000 logger.wandb.tags=["narval","more-diversity","penalty-warmup"]
# python run_training.py ckpt_path=null model.optimizer.lr=0.01 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=0.1 datamodule.dataset.z_dim=20 logger.wandb.tags=["narval","4-domains"]



# python run_training.py ckpt_path=null model/optimizer=adam model.optimizer.lr=0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=0.1 datamodule.dataset.z_dim=20 model.wait_steps=0 model.linear_steps=1 logger.wandb.tags=["test"]
python run_training.py ckpt_path=null model/autoencoder=fc_mlp_ae_synthetic model/optimizer=adam model.optimizer.lr=0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=0.1 datamodule.dataset.z_dim=20 model.wait_steps=0 model.linear_steps=1 logger.wandb.tags=["test"]

deactivate
module purge
