#!/bin/bash

#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=2:59:00
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
# export WANDB_MODE=offline


# python run_training.py ckpt_path=null trainer.accelerator='cpu' model.optimizer.lr=0.001,0.0001 datamodule=balls_encoded model=balls_md_encoded_autoencoder model.z_dim=64,128 model.z_dim_invariant_fraction=0.2,0.5,0.8 model.hinge_loss_weight=0.0,0.01,0.1 model.penalty_criterion="minmax" model.penalty_weight=1.0,0.1 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["narval","balls-encoded-sweep"] ~callbacks.early_stopping ~callbacks.visualization_callback run_path="/home/aminm/scratch/logs/training/runs/autoencoder_md_balls_128_iv_1_sp_1/2023-09-19_14-03-08" ckpt_path=null --multirun

# gold standard
python run_training.py ckpt_path=null trainer.accelerator='cpu' model.optimizer.lr=0.001,0.0001 datamodule=balls_encoded model=balls_md_encoded_autoencoder model.z_dim=64,128 model.z_dim_invariant_fraction=0.2,0.5,0.8 model.hinge_loss_weight=0.0,0.01,0.1 model.penalty_criterion="minmax" model.penalty_weight=1.0,0.1 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["narval","balls-encoded-sweep","resnet-bn"] ~callbacks.early_stopping ~callbacks.visualization_callback run_path="/home/aminm/scratch/logs/training/runs/autoencoder_md_balls_128_iv_1_sp_1/2023-09-19_16-57-50" ckpt_path=null --multirun

# ------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------- #
# --------------------------------- Synthethic Mixing --------------------------------- #
# ------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------- #
# ---------------------------- Linear Mixing, Linear Model ---------------------------- #
# ------------------------------------------------------------------------------------- #

# num_domains = 2
python run_training.py ckpt_path=null model.optimizer.lr=0.1,0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1,0.01,0.001 datamodule.dataset.z_dim=4,6,8,10,16,20 ~callbacks.early_stopping --multirun

# num_domains = 4

# z_dim = 4
# gpu
# python run_training.py ckpt_path=null model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=4 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["narval","4-domain","fixed"] ~callbacks.early_stopping --multirun
# cpu
# python run_training.py ckpt_path=null hydra/launcher=submitit_slurm_narval_cpu trainer.gpus=0 model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=4 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["narval","4-domain","fixed"] ~callbacks.early_stopping --multirun
# python run_training.py ckpt_path=null model/optimizer=adam model.optimizer.lr=0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0 datamodule.dataset.z_dim=4 model.wait_steps=0 model.linear_steps=1 logger.wandb.tags=["narval","4-domain","fixed"] ~callbacks.early_stopping --multirun

# z_dim = 8
# gpu
# python run_training.py ckpt_path=null model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=8 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["narval","4-domain","fixed"] ~callbacks.early_stopping --multirun
# cpu
# python run_training.py ckpt_path=null hydra/launcher=submitit_slurm_narval_cpu trainer.gpus=0 model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=8 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["narval","4-domain","fixed"] ~callbacks.early_stopping --multirun

# z_dim = 12
# gpu
# python run_training.py ckpt_path=null model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=12 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["narval","4-domain","fixed"] ~callbacks.early_stopping --multirun
# cpu
# python run_training.py ckpt_path=null hydra/launcher=submitit_slurm_narval_cpu trainer.gpus=0 model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=12 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["narval","4-domain","fixed"] ~callbacks.early_stopping --multirun

# z_dim = 16
# gpu
# python run_training.py ckpt_path=null model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=16 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["narval","4-domain","fixed"] ~callbacks.early_stopping --multirun
# cpu
# python run_training.py ckpt_path=null hydra/launcher=submitit_slurm_narval_cpu trainer.gpus=0 model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=16 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["narval","4-domain","fixed"] ~callbacks.early_stopping --multirun

# z_dim = 20
# gpu
# python run_training.py ckpt_path=null model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=20 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["narval","4-domain","fixed"] ~callbacks.early_stopping --multirun
# cpu
# python run_training.py ckpt_path=null hydra/launcher=submitit_slurm_narval_cpu trainer.gpus=0 model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=20 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["narval","4-domain","fixed"] ~callbacks.early_stopping --multirun

# ------------------------------------------------------------------------------------- #
# ------------ Sweep No dimension mismatch, min-max penalty, no hinge loss ------------ #

# cpu, each sweep contains 2x3x5x3x2x2=360 runs
# num_domains=2
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam,adamw model.optimizer.lr=0.05,0.01,0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=0.0 datamodule=mixing datamodule.dataset.num_domains=2 datamodule.dataset.z_dim=2,8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1,0.01 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["narval","sweep-linGlinM-match-minmax-nohinge"] --multirun
# num_domains=4
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam,adamw model.optimizer.lr=0.05,0.01,0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=0.0 datamodule=mixing datamodule.dataset.num_domains=4 datamodule.dataset.z_dim=2,8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1,0.01 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["narval","sweep-linGlinM-match-minmax-nohinge"] --multirun
# num_domains=8
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam,adamw model.optimizer.lr=0.05,0.01,0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=0.0 datamodule=mixing datamodule.dataset.num_domains=8 datamodule.dataset.z_dim=2,8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1,0.01 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["narval","sweep-linGlinM-match-minmax-nohinge"] --multirun
# num_domains=16
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam,adamw model.optimizer.lr=0.05,0.01,0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=0.0 datamodule=mixing datamodule.dataset.num_domains=16 datamodule.dataset.z_dim=2,8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1,0.01 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["narval","sweep-linGlinM-match-minmax-nohinge"] --multirun

# ------------------------------------------------------------------------------------- #
# ---------- Sweep No dimension mismatch, min-max penalty, with hinge loss ------------ #

# cpu, each sweep contains 4x2x3=24 runs
# num_domains=2
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=10.0,1.0,0.1 datamodule=mixing datamodule.dataset.num_domains=2 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 logger.wandb.tags=["narval","sweep-linGlinM-match-minmax-hinge"] --multirun
# num_domains=4
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=10.0,1.0,0.1 datamodule=mixing datamodule.dataset.num_domains=4 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 logger.wandb.tags=["narval","sweep-linGlinM-match-minmax-hinge"] --multirun
# num_domains=8
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=10.0,1.0,0.1 datamodule=mixing datamodule.dataset.num_domains=8 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 logger.wandb.tags=["narval","sweep-linGlinM-match-minmax-hinge"] --multirun
# num_domains=16
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=10.0,1.0,0.1 datamodule=mixing datamodule.dataset.num_domains=16 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 logger.wandb.tags=["narval","sweep-linGlinM-match-minmax-hinge"] --multirun


# ------------------------------------------------------------------------------------- #
# -------------- Sweep No dimension mismatch, std penalty, no hinge loss -------------- #

# cpu, each sweep contains 2x4=8 runs
# num_domains=2
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.penalty_criterion="stddev" model.hinge_loss_weight=0.0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic datamodule=mixing datamodule.dataset.num_domains=2 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 model.linear_steps=1 logger.wandb.tags=["narval","sweep-linGlinM-match-stddev-nohinge"] ~callbacks.early_stopping --multirun
# num_domains=4
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.penalty_criterion="stddev" model.hinge_loss_weight=0.0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic datamodule=mixing datamodule.dataset.num_domains=4 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 model.linear_steps=1 logger.wandb.tags=["narval","sweep-linGlinM-match-stddev-nohinge"] ~callbacks.early_stopping --multirun
# num_domains=8
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.penalty_criterion="stddev" model.hinge_loss_weight=0.0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic datamodule=mixing datamodule.dataset.num_domains=8 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 model.linear_steps=1 logger.wandb.tags=["narval","sweep-linGlinM-match-stddev-nohinge"] ~callbacks.early_stopping --multirun
# num_domains=16
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.penalty_criterion="stddev" model.hinge_loss_weight=0.0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic datamodule=mixing datamodule.dataset.num_domains=16 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 model.linear_steps=1 logger.wandb.tags=["narval","sweep-linGlinM-match-stddev-nohinge"] ~callbacks.early_stopping --multirun

# ------------------------------------------------------------------------------------- #
# ------------ Sweep No dimension mismatch, std penalty, with hinge loss -------------- #

# cpu, each sweep contains 2x4x3=24 runs
# num_domains=2
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.penalty_criterion="stddev" model.hinge_loss_weight=10.0,1.0,0.1 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic datamodule=mixing datamodule.dataset.num_domains=2 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 model.linear_steps=1 logger.wandb.tags=["narval","sweep-linGlinM-match-stddev-hinge"] ~callbacks.early_stopping --multirun
# num_domains=4
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.penalty_criterion="stddev" model.hinge_loss_weight=10.0,1.0,0.1 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic datamodule=mixing datamodule.dataset.num_domains=4 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 model.linear_steps=1 logger.wandb.tags=["narval","sweep-linGlinM-match-stddev-hinge"] ~callbacks.early_stopping --multirun
# num_domains=8
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.penalty_criterion="stddev" model.hinge_loss_weight=10.0,1.0,0.1 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic datamodule=mixing datamodule.dataset.num_domains=8 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 model.linear_steps=1 logger.wandb.tags=["narval","sweep-linGlinM-match-stddev-hinge"] ~callbacks.early_stopping --multirun
# num_domains=16
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.penalty_criterion="stddev" model.hinge_loss_weight=10.0,1.0,0.1 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic datamodule=mixing datamodule.dataset.num_domains=16 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 model.linear_steps=1 logger.wandb.tags=["narval","sweep-linGlinM-match-stddev-hinge"] ~callbacks.early_stopping --multirun


# ------------------------------------------------------------------------------------- #
# ------------ Sweep + dimension mismatch, min-max penalty, no hinge loss ------------- #

# cpu, each sweep contains 4x4x2=32 runs
# num_domains=2
# python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=0.0 datamodule=mixing datamodule.dataset.num_domains=2 model.mismatch_dims=2,4,6,8 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback ~callbacks.early_stopping model.penalty_weight=1.0,0.1 logger.wandb.tags=["narval","sweep-linGlinM-mismatch-minmax-nohinge"] --multirun
# num_domains=4
# python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=0.0 datamodule=mixing datamodule.dataset.num_domains=4 model.mismatch_dims=2,4,6,8 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback ~callbacks.early_stopping model.penalty_weight=1.0,0.1 logger.wandb.tags=["narval","sweep-linGlinM-mismatch-minmax-nohinge"] --multirun
# num_domains=8
# python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=0.0 datamodule=mixing datamodule.dataset.num_domains=8 model.mismatch_dims=2,4,6,8 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback ~callbacks.early_stopping model.penalty_weight=1.0,0.1 logger.wandb.tags=["narval","sweep-linGlinM-mismatch-minmax-nohinge"] --multirun
# num_domains=16
# python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=0.0 datamodule=mixing datamodule.dataset.num_domains=16 model.mismatch_dims=2,4,6,8 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback ~callbacks.early_stopping model.penalty_weight=1.0,0.1 logger.wandb.tags=["narval","sweep-linGlinM-mismatch-minmax-nohinge"] --multirun

# ------------------------------------------------------------------------------------- #
# ----------- Sweep + dimension mismatch, min-max penalty, with hinge loss ------------ #

# cpu, each sweep contains 4x4x2x2=64 runs
# num_domains=2
# python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=10.0,1.0 datamodule=mixing datamodule.dataset.num_domains=2 model.mismatch_dims=2,4,6,8 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback ~callbacks.early_stopping model.penalty_weight=1.0,0.1 logger.wandb.tags=["narval","sweep-linGlinM-mismatch-minmax-hinge"] --multirun
# num_domains=4
# python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=10.0,1.0 datamodule=mixing datamodule.dataset.num_domains=4 model.mismatch_dims=2,4,6,8 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback ~callbacks.early_stopping model.penalty_weight=1.0,0.1 logger.wandb.tags=["narval","sweep-linGlinM-mismatch-minmax-hinge"] --multirun
# num_domains=8
# python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=10.0,1.0 datamodule=mixing datamodule.dataset.num_domains=8 model.mismatch_dims=2,4,6,8 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback ~callbacks.early_stopping model.penalty_weight=1.0,0.1 logger.wandb.tags=["narval","sweep-linGlinM-mismatch-minmax-hinge"] --multirun
# num_domains=16
# python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=10.0,1.0 datamodule=mixing datamodule.dataset.num_domains=16 model.mismatch_dims=2,4,6,8 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback ~callbacks.early_stopping model.penalty_weight=1.0,0.1 logger.wandb.tags=["narval","sweep-linGlinM-mismatch-minmax-hinge"] --multirun

# ------------------------------------------------------------------------------------- #
# -------------- Sweep + dimension mismatch, std penalty, no hinge loss --------------- #

# cpu, each sweep contains 4x4x2=32 runs
# num_domains=2
# python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="stddev" model.hinge_loss_weight=0.0 datamodule=mixing datamodule.dataset.num_domains=2 model.mismatch_dims=2,4,6,8 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback ~callbacks.early_stopping model.penalty_weight=1.0,0.1 logger.wandb.tags=["narval","sweep-linGlinM-mismatch-stddev-nohinge"] --multirun
# num_domains=4
# python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="stddev" model.hinge_loss_weight=0.0 datamodule=mixing datamodule.dataset.num_domains=4 model.mismatch_dims=2,4,6,8 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback ~callbacks.early_stopping model.penalty_weight=1.0,0.1 logger.wandb.tags=["narval","sweep-linGlinM-mismatch-stddev-nohinge"] --multirun
# num_domains=8
# python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="stddev" model.hinge_loss_weight=0.0 datamodule=mixing datamodule.dataset.num_domains=8 model.mismatch_dims=2,4,6,8 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback ~callbacks.early_stopping model.penalty_weight=1.0,0.1 logger.wandb.tags=["narval","sweep-linGlinM-mismatch-stddev-nohinge"] --multirun
# num_domains=16
# python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="stddev" model.hinge_loss_weight=0.0 datamodule=mixing datamodule.dataset.num_domains=16 model.mismatch_dims=2,4,6,8 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback ~callbacks.early_stopping model.penalty_weight=1.0,0.1 logger.wandb.tags=["narval","sweep-linGlinM-mismatch-stddev-nohinge"] --multirun

# ------------------------------------------------------------------------------------- #
# ------------- Sweep + dimension mismatch, std penalty, with hinge loss -------------- #

# cpu, each sweep contains 4x4x2x2=64 runs
# num_domains=2
# python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="stddev" model.hinge_loss_weight=10.0,1.0 datamodule=mixing datamodule.dataset.num_domains=2 model.mismatch_dims=2,4,6,8 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback ~callbacks.early_stopping model.penalty_weight=1.0,0.1 logger.wandb.tags=["narval","sweep-linGlinM-mismatch-stddev-hinge"] --multirun
# num_domains=4
# python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="stddev" model.hinge_loss_weight=10.0,1.0 datamodule=mixing datamodule.dataset.num_domains=4 model.mismatch_dims=2,4,6,8 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback ~callbacks.early_stopping model.penalty_weight=1.0,0.1 logger.wandb.tags=["narval","sweep-linGlinM-mismatch-stddev-hinge"] --multirun
# num_domains=8
# python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="stddev" model.hinge_loss_weight=10.0,1.0 datamodule=mixing datamodule.dataset.num_domains=8 model.mismatch_dims=2,4,6,8 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback ~callbacks.early_stopping model.penalty_weight=1.0,0.1 logger.wandb.tags=["narval","sweep-linGlinM-mismatch-stddev-hinge"] --multirun
# num_domains=16
# python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="stddev" model.hinge_loss_weight=10.0,1.0 datamodule=mixing datamodule.dataset.num_domains=16 model.mismatch_dims=2,4,6,8 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback ~callbacks.early_stopping model.penalty_weight=1.0,0.1 logger.wandb.tags=["narval","sweep-linGlinM-mismatch-stddev-hinge"] --multirun

# ------------------------------------------------------------------------------------- #
# -------------------------- Linear Mixing, Non-Linear Model -------------------------- #
# ------------------------------------------------------------------------------------- #

# python run_training.py ckpt_path=null model/autoencoder=fc_mlp_ae_synthetic model/optimizer=adam model.optimizer.lr=0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=4 model.wait_steps=0 model.linear_steps=1 logger.wandb.tags=["narval","non-linear-model"] ~callbacks.early_stopping --multirun
# python run_training.py ckpt_path=null model/optimizer=adam model.optimizer.lr=0.01 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0 datamodule.dataset.z_dim=4 model.wait_steps=0 model.linear_steps=1 logger.wandb.tags=["narval","stddev"] ~callbacks.early_stopping

# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.optimizer.lr=0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0 datamodule.dataset.z_dim=4 model.wait_steps=0 model.linear_steps=1 logger.wandb.tags=["narval","4-domain","hinge_loss"] ~callbacks.early_stopping

deactivate
module purge
