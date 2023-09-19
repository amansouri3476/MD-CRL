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
# --------------------------------------- Balls --------------------------------------- #
# ------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------- #
# -------------------------------- Reconstruction Only -------------------------------- #

# resnet

# iv=1,sp=1
# python run_training.py trainer.accelerator='cpu' ckpt_path=null model.optimizer.lr=0.001 datamodule=md_balls datamodule.dataset.n_balls_invariant=1 datamodule.dataset.n_balls_spurious=1 model=balls model.z_dim=64 model/autoencoder=cnn_ae_balls model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["narval","balls","test"] ~callbacks.early_stopping
# python run_training.py trainer.gradient_clip_val=0.1 trainer.accelerator='gpu' trainer.devices=1 ckpt_path=null model.optimizer.lr=0.001 datamodule=md_balls datamodule.dataset.n_balls_invariant=1 datamodule.dataset.n_balls_spurious=1 model=balls model.z_dim=64 model/autoencoder=cnn_ae_balls model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["narval","balls","test"] ~callbacks.early_stopping
# iv=1,sp=2
# python run_training.py trainer.accelerator='gpu' trainer.devices=1 ckpt_path=null model.optimizer.lr=0.001 datamodule=md_balls datamodule.dataset.n_balls_invariant=1 datamodule.dataset.n_balls_spurious=2 model=balls model.z_dim=64 model/autoencoder=cnn_ae_balls model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["narval","balls","test"] ~callbacks.early_stopping
# iv=1,sp=3
# python run_training.py trainer.accelerator='gpu' trainer.devices=1 ckpt_path=null model.optimizer.lr=0.001 datamodule=md_balls datamodule.dataset.n_balls_invariant=1 datamodule.dataset.n_balls_spurious=3 model=balls model.z_dim=64 model/autoencoder=cnn_ae_balls model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["narval","balls","test"] ~callbacks.early_stopping
# iv=2,sp=1
# python run_training.py trainer.accelerator='gpu' trainer.devices=1 ckpt_path=null model.optimizer.lr=0.001 datamodule=md_balls datamodule.dataset.n_balls_invariant=2 datamodule.dataset.n_balls_spurious=1 model=balls model.z_dim=64 model/autoencoder=cnn_ae_balls model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["narval","balls","test"] ~callbacks.early_stopping
# iv=2,sp=2
# python run_training.py trainer.accelerator='gpu' trainer.devices=1 ckpt_path=null model.optimizer.lr=0.001 datamodule=md_balls datamodule.dataset.n_balls_invariant=2 datamodule.dataset.n_balls_spurious=2 model=balls model.z_dim=64 model/autoencoder=cnn_ae_balls model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["narval","balls","test"] ~callbacks.early_stopping
# iv=2,sp=3
# python run_training.py trainer.accelerator='gpu' trainer.devices=1 ckpt_path=null model.optimizer.lr=0.001 datamodule=md_balls datamodule.dataset.n_balls_invariant=2 datamodule.dataset.n_balls_spurious=3 model=balls model.z_dim=64 model/autoencoder=cnn_ae_balls model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["narval","balls","test"] ~callbacks.early_stopping
# python run_training.py trainer.accelerator='cpu' ckpt_path=null model.optimizer.lr=0.001 datamodule=md_balls model=balls model.z_dim=64 model/autoencoder=resnet18_ae_balls model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["narval","balls","test"] ~callbacks.early_stopping

# ------------------------------------------------------------------------------------- #
# --------------------- Loading from ckpt to get encoded datasets --------------------- #
# iv=1,sp=1
# python run_training.py trainer.accelerator='cpu' model.optimizer.lr=0.001 datamodule=md_balls datamodule.dataset.n_balls_invariant=1 datamodule.dataset.n_balls_spurious=1 model=balls model.z_dim=64 model/autoencoder=cnn_ae_balls model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["narval","balls","test"] ~callbacks.early_stopping ckpt_path="/home/aminm/scratch/logs/training/runs/autoencoder_md_balls_64/2023-09-15_13-25-59/checkpoints/autoencoder_md_balls_64-epoch\=294-val_loss\=0.00-val_r2_hz_z\=0.00-val_r2_hz_z\=0.00.ckpt"
# python run_training.py trainer.accelerator='gpu' trainer.devices=1 ckpt_path=null model.optimizer.lr=0.001 datamodule=md_balls datamodule.dataset.n_balls_invariant=1 datamodule.dataset.n_balls_spurious=1 model=balls model.z_dim=64 model/autoencoder=cnn_ae_balls model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["narval","balls","test"] ~callbacks.early_stopping

# ------------------------------------------------------------------------------------- #
# -------------------------- Disentanglement with encoded images ---------------------- #

# -------------------------- min-max penalty, no hinge loss ---------------------- #

# iv=1,sp=1
# cpu
# python run_training.py ckpt_path=null trainer.accelerator='cpu' model.optimizer.lr=0.001 datamodule=balls_encoded model=balls_md_encoded_autoencoder model.z_dim=4 model.z_dim_invariant_fraction=0.5 model.hinge_loss_weight=0.0 model.penalty_criterion="minmax" model.penalty_weight=1.0 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["narval","balls-encoded"] ~callbacks.early_stopping ~callbacks.visualization_callback run_path="/home/aminm/scratch/logs/training/runs/autoencoder_md_balls_64_iv_1_sp_1/2023-09-16_08-23-15" ckpt_path=null
# python run_training.py ckpt_path=null trainer.accelerator='cpu' model.optimizer.lr=0.001 datamodule=balls_encoded model=balls_md_encoded_autoencoder model.z_dim=4 model.z_dim_invariant_fraction=0.5 model.hinge_loss_weight=0.0 model.penalty_criterion="minmax" model.penalty_weight=1.0 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["narval","balls-encoded"] ~callbacks.early_stopping ~callbacks.visualization_callback run_path="/home/aminm/scratch/logs/training/runs/autoencoder_md_balls_64_iv_1_sp_1/2023-09-16_13-54-39" ckpt_path=null

# ------------------------------------------------------------------------------------- #
# --------------------------------------- MNIST --------------------------------------- #
# ------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------- #
# -------------------------------- just reconstruction -------------------------- #

# python run_training.py ckpt_path=null model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist datamodule.dataset.num_domains=8 model=default model.z_dim=256 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" model.autoencoder.num_channels=3 logger.wandb.tags=["mila","mnist","intrepolation"] ~callbacks.early_stopping trainer.gpus=1 model.autoencoder.upsampling_interpolation="trilinear"

# python run_training.py ckpt_path=null model.optimizer.lr=0.001,0.01 datamodule/dataset=multi_domain_mnist datamodule.dataset.num_domains=8 model=default model.z_dim=8,16,32,64 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" model.autoencoder.num_channels=3 logger.wandb.tags=["mila","mnist","reconstruction"] ~callbacks.early_stopping --multirun
# python run_training.py ckpt_path=null model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist datamodule.dataset.num_domains=8 model=default model.z_dim=512 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" model.autoencoder.num_channels=3 logger.wandb.tags=["mila","mnist","reconstruction"] ~callbacks.early_stopping
# python run_training.py ckpt_path=null model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist datamodule.dataset.num_domains=8 model=mnist_md_autoencoder model.autoencoder.num_channels=3 model.z_dim=8 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["mila","mnist","8-domain","autoencoder","test"] ~callbacks.early_stopping

# ------------------------------------------------------------------------------------- #
# -------------------------------- Disentanglement -------------------------- #


# python run_training.py ckpt_path=null model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist datamodule.dataset.num_domains=8 model=mnist_md_autoencoder model.z_dim=256 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" model.autoencoder.num_channels=3 logger.wandb.tags=["mila","mnist","intrepolation"] ~callbacks.early_stopping trainer.gpus=1 model.wait_steps=10000 model.linear_steps=1000 model.z_dim_invariant=64
# python run_training.py ckpt_path=null model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist datamodule.dataset.num_domains=8 model=mnist_md_autoencoder model.z_dim=256 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" model.autoencoder.num_channels=3 logger.wandb.tags=["mila","mnist","reconstruction"] ~callbacks.early_stopping trainer.gpus=1 model.wait_steps=10000 model.linear_steps=1000 model.z_dim_invariant=64

# resnet
python run_training.py trainer.accelerator='cpu' ckpt_path=null model.optimizer.lr=0.0001 datamodule=md_balls datamodule.dataset.n_balls_invariant=1 datamodule.dataset.n_balls_spurious=1 model=balls model.z_dim=64 model/autoencoder=resnet18_ae_balls model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["mila","balls","test"] ~callbacks.early_stopping


deactivate
module purge
