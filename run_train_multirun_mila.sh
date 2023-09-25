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
# python run_training.py ckpt_path=null model.optimizer.lr=0.1,0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1,0.01,0.001 datamodule.dataset.z_dim=4,6,8,10,16,20 ~callbacks.early_stopping --multirun

# num_domains = 4

# z_dim = 4
# gpu
# python run_training.py ckpt_path=null model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=4 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["mila","4-domain","fixed"] ~callbacks.early_stopping --multirun
# cpu
# python run_training.py ckpt_path=null hydra/launcher=submitit_slurm_narval_cpu trainer.gpus=0 model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=4 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["mila","4-domain","fixed"] ~callbacks.early_stopping --multirun
# python run_training.py ckpt_path=null model/optimizer=adam model.optimizer.lr=0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0 datamodule.dataset.z_dim=4 model.wait_steps=0 model.linear_steps=1 logger.wandb.tags=["mila","4-domain","fixed"] ~callbacks.early_stopping --multirun

# z_dim = 8
# gpu
# python run_training.py ckpt_path=null model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=8 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["mila","4-domain","fixed"] ~callbacks.early_stopping --multirun
# cpu
# python run_training.py ckpt_path=null hydra/launcher=submitit_slurm_narval_cpu trainer.gpus=0 model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=8 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["mila","4-domain","fixed"] ~callbacks.early_stopping --multirun

# z_dim = 12
# gpu
# python run_training.py ckpt_path=null model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=12 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["mila","4-domain","fixed"] ~callbacks.early_stopping --multirun
# cpu
# python run_training.py ckpt_path=null hydra/launcher=submitit_slurm_narval_cpu trainer.gpus=0 model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=12 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["mila","4-domain","fixed"] ~callbacks.early_stopping --multirun

# z_dim = 16
# gpu
# python run_training.py ckpt_path=null model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=16 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["mila","4-domain","fixed"] ~callbacks.early_stopping --multirun
# cpu
# python run_training.py ckpt_path=null hydra/launcher=submitit_slurm_narval_cpu trainer.gpus=0 model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=16 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["mila","4-domain","fixed"] ~callbacks.early_stopping --multirun

# z_dim = 20
# gpu
# python run_training.py ckpt_path=null model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=20 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["mila","4-domain","fixed"] ~callbacks.early_stopping --multirun
# cpu
# python run_training.py ckpt_path=null hydra/launcher=submitit_slurm_narval_cpu trainer.gpus=0 model/optimizer=adam model.optimizer.lr=0.01,0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=20 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["mila","4-domain","fixed"] ~callbacks.early_stopping --multirun

# ------------------------------------------------------------------------------------- #
# ------------ Sweep No dimension mismatch, min-max penalty, no hinge loss ------------ #

# cpu, each sweep contains 2x3x5x3x2x2=360 runs
# num_domains=2
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam,adamw model.optimizer.lr=0.05,0.01,0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=0.0 datamodule=mixing datamodule.dataset.num_domains=2 datamodule.dataset.z_dim=2,8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1,0.01 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["mila","sweep-linGlinM-match-minmax-nohinge"] --multirun
# num_domains=4
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam,adamw model.optimizer.lr=0.05,0.01,0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=0.0 datamodule=mixing datamodule.dataset.num_domains=4 datamodule.dataset.z_dim=2,8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1,0.01 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["mila","sweep-linGlinM-match-minmax-nohinge"] --multirun
# num_domains=8
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam,adamw model.optimizer.lr=0.05,0.01,0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=0.0 datamodule=mixing datamodule.dataset.num_domains=8 datamodule.dataset.z_dim=2,8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1,0.01 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["mila","sweep-linGlinM-match-minmax-nohinge"] --multirun
# num_domains=16
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam,adamw model.optimizer.lr=0.05,0.01,0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=0.0 datamodule=mixing datamodule.dataset.num_domains=16 datamodule.dataset.z_dim=2,8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1,0.01 model.wait_steps=0,1000 model.linear_steps=1,2000 logger.wandb.tags=["mila","sweep-linGlinM-match-minmax-nohinge"] --multirun

# ------------------------------------------------------------------------------------- #
# ---------- Sweep No dimension mismatch, min-max penalty, with hinge loss ------------ #

# cpu, each sweep contains 4x2x3=24 runs
# num_domains=2
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=10.0,1.0,0.1 datamodule=mixing datamodule.dataset.num_domains=2 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 logger.wandb.tags=["mila","sweep-linGlinM-match-minmax-hinge"] --multirun
# num_domains=4
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=10.0,1.0,0.1 datamodule=mixing datamodule.dataset.num_domains=4 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 logger.wandb.tags=["mila","sweep-linGlinM-match-minmax-hinge"] --multirun
# num_domains=8
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=10.0,1.0,0.1 datamodule=mixing datamodule.dataset.num_domains=8 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 logger.wandb.tags=["mila","sweep-linGlinM-match-minmax-hinge"] --multirun
# num_domains=16
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=10.0,1.0,0.1 datamodule=mixing datamodule.dataset.num_domains=16 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 logger.wandb.tags=["mila","sweep-linGlinM-match-minmax-hinge"] --multirun

# ------------------------------------------------------------------------------------- #
# -------------- Sweep No dimension mismatch, std penalty, no hinge loss -------------- #

# cpu, each sweep contains 2x4=8 runs
# num_domains=2
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.penalty_criterion="stddev" model.hinge_loss_weight=0.0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic datamodule=mixing datamodule.dataset.num_domains=2 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 model.linear_steps=1 logger.wandb.tags=["mila","sweep-linGlinM-match-stddev-nohinge"] ~callbacks.early_stopping --multirun
# num_domains=4
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.penalty_criterion="stddev" model.hinge_loss_weight=0.0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic datamodule=mixing datamodule.dataset.num_domains=4 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 model.linear_steps=1 logger.wandb.tags=["mila","sweep-linGlinM-match-stddev-nohinge"] ~callbacks.early_stopping --multirun
# num_domains=8
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.penalty_criterion="stddev" model.hinge_loss_weight=0.0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic datamodule=mixing datamodule.dataset.num_domains=8 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 model.linear_steps=1 logger.wandb.tags=["mila","sweep-linGlinM-match-stddev-nohinge"] ~callbacks.early_stopping --multirun
# num_domains=16
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.penalty_criterion="stddev" model.hinge_loss_weight=0.0 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic datamodule=mixing datamodule.dataset.num_domains=16 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 model.linear_steps=1 logger.wandb.tags=["mila","sweep-linGlinM-match-stddev-nohinge"] ~callbacks.early_stopping --multirun

# ------------------------------------------------------------------------------------- #
# ------------ Sweep No dimension mismatch, std penalty, with hinge loss -------------- #

# cpu, each sweep contains 2x4x3=24 runs
# num_domains=2
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.penalty_criterion="stddev" model.hinge_loss_weight=10.0,1.0,0.1 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic datamodule=mixing datamodule.dataset.num_domains=2 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 model.linear_steps=1 logger.wandb.tags=["mila","sweep-linGlinM-match-stddev-hinge"] ~callbacks.early_stopping --multirun
# num_domains=4
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.penalty_criterion="stddev" model.hinge_loss_weight=10.0,1.0,0.1 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic datamodule=mixing datamodule.dataset.num_domains=4 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 model.linear_steps=1 logger.wandb.tags=["mila","sweep-linGlinM-match-stddev-hinge"] ~callbacks.early_stopping --multirun
# num_domains=8
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.penalty_criterion="stddev" model.hinge_loss_weight=10.0,1.0,0.1 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic datamodule=mixing datamodule.dataset.num_domains=8 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 model.linear_steps=1 logger.wandb.tags=["mila","sweep-linGlinM-match-stddev-hinge"] ~callbacks.early_stopping --multirun
# num_domains=16
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.penalty_criterion="stddev" model.hinge_loss_weight=10.0,1.0,0.1 model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic datamodule=mixing datamodule.dataset.num_domains=16 datamodule.dataset.z_dim=8,16,32,64 ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 model.linear_steps=1 logger.wandb.tags=["mila","sweep-linGlinM-match-stddev-hinge"] ~callbacks.early_stopping --multirun


# ------------------------------------------------------------------------------------- #
# ---------------------------- Linear Mixing, Linear Model ---------------------------- #
# ----------- z_{m, \perp} > z_{\perp}, z_{m, \not\perp} > z_{\not\perp} -------------- #
# ------------------------------------------------------------------------------------- #

# cpu
# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.optimizer.lr=0.05 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0 datamodule.dataset.z_dim=8 model.autoencoder.latent_dim=12 model.z_dim_invariant=6 logger.wandb.tags=["mila","4-domain","diff_z_zm"] ~callbacks.early_stopping

# ------------------------------------------------------------------------------------- #
# -------------------------- Linear Mixing, Non-Linear Model -------------------------- #
# ------------------------------------------------------------------------------------- #

# python run_training.py ckpt_path=null model/autoencoder=fc_mlp_ae_synthetic model/optimizer=adam model.optimizer.lr=0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=4 model.wait_steps=0 model.linear_steps=1 logger.wandb.tags=["mila","non-linear-model"] --multirun

# cpu
# python run_training.py ckpt_path=null trainer.gpus=0 model/autoencoder=fc_mlp_ae_synthetic model/optimizer=adam model.optimizer.lr=0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0 datamodule.dataset.z_dim=4 logger.wandb.tags=["mila","4-domain","non-linear-model"] ~callbacks.early_stopping

# ------------------------------------------------------------------------------------- #
# -------------------------- Non-Linear Mixing, Non-Linear Model -------------------------- #
# ------------------------------------------------------------------------------------- #

# python run_training.py ckpt_path=null model/autoencoder=fc_mlp_ae_synthetic model/optimizer=adam model.optimizer.lr=0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0,0.1 datamodule.dataset.z_dim=4 model.wait_steps=0 model.linear_steps=1 logger.wandb.tags=["mila","non-linear-model"] --multirun

# cpu
# python run_training.py ckpt_path=null trainer.gpus=0 model/autoencoder=fc_mlp_ae_synthetic datamodule.dataset.linear=False model/optimizer=adam model.optimizer.lr=0.01 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0 datamodule.dataset.z_dim=4 logger.wandb.tags=["mila","4-domain","non-linear-model-data"] ~callbacks.early_stopping

# python run_training.py ckpt_path=null trainer.gpus=0 model/optimizer=adam model.optimizer.lr=0.001 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=1.0 datamodule.dataset.z_dim=4 model.wait_steps=0 model.linear_steps=1 logger.wandb.tags=["mila","4-domain","hinge_loss"]

# ------------------------------------------------------------------------------------- #
# ----------------------- Polynomial Mixing, Non-Linear Model ------------------------- #
# ------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------- #
# ----------------------------------- Just reconstruction ----------------------------- #

python run_training.py ckpt_path=null model=mixing_synthetic model/autoencoder=poly_ae model.optimizer.lr=0.001 datamodule=mixing datamodule.dataset.linear=False datamodule.dataset.non_linearity=polynomial datamodule.dataset.polynomial_degree=2 datamodule.batch_size=512 datamodule.dataset.z_dim=6 model.z_dim=6 datamodule.dataset.num_domains=8 datamodule.dataset.x_dim=200 ~callbacks.visualization_callback logger.wandb.tags=["mila","poly-mixing"]
# ------------------------------------------------------------------------------------- #
# ------------------------------------- Disentanglement ------------------------------- #

python run_training.py ckpt_path=null model=mixing_md_encoded_autoencoder model.optimizer.lr=0.001 datamodule=mixing_encoded datamodule.batch_size=512 ~callbacks.visualization_callback logger.wandb.tags=["mila","poly-mixing-disentanglement"] run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_synthetic_mixing_6/2023-09-24_13-57-37"
# ------------------------------------------------------------------------------------- #
# --------------------------------------- MNIST --------------------------------------- #
# ------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------- #
# -------------------------------- Sweep just reconstruction -------------------------- #

# python run_training.py ckpt_path=null model.optimizer.lr=0.001,0.01 datamodule/dataset=multi_domain_mnist datamodule.dataset.num_domains=8 model=default model.z_dim=8,16,32,64 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" model.autoencoder.num_channels=3 logger.wandb.tags=["mila","mnist","reconstruction"] ~callbacks.early_stopping --multirun
# python run_training.py ckpt_path=null model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist datamodule.dataset.num_domains=8 model=default model.z_dim=256 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" model.autoencoder.num_channels=3 logger.wandb.tags=["mila","mnist","reconstruction"] ~callbacks.early_stopping
# python run_training.py ckpt_path=null model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist datamodule.dataset.num_domains=8 model=mnist_md_autoencoder model.autoencoder.num_channels=3 model.z_dim=8 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["mila","mnist","8-domain","autoencoder","test"] ~callbacks.early_stopping

# ------------------------------------------------------------------------------------- #
# ---------------------- Sweep min-max penalty, no hinge loss ------------------------- #

# cpu, each sweep contains 4x2x2x2=32 runs
# num_domains=2
python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=mnist_md_autoencoder model.z_dim=256 model.z_dim_invariant_fraction=0.2,0.4,0.6,0.8 model.penalty_criterion="minmax" model.penalty_weight=1.0,0.1 model.hinge_loss_weight=0.0 model.autoencoder.num_channels=3 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" datamodule.dataset.num_domains=2 model.wait_steps=0,5000 model.linear_steps=1,5000 logger.wandb.tags=["mila","sweep-mnist-minmax-nohinge"] --multirun
python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=mnist_md_autoencoder model.z_dim=256 model.z_dim_invariant_fraction=0.2 model.penalty_criterion="minmax" model.penalty_weight=1.0 model.hinge_loss_weight=0.0 model.autoencoder.num_channels=3 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" datamodule.dataset.num_domains=2 model.wait_steps=0 model.linear_steps=1 logger.wandb.tags=["mila","test"]
# num_domains=4
python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=mnist_md_autoencoder model.z_dim=256 model.z_dim_invariant_fraction=0.2,0.4,0.6,0.8 model.penalty_criterion="minmax" model.penalty_weight=1.0,0.1 model.hinge_loss_weight=0.0 model.autoencoder.num_channels=3 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" datamodule.dataset.num_domains=4 model.wait_steps=0,5000 model.linear_steps=1,5000 logger.wandb.tags=["mila","sweep-mnist-minmax-nohinge"] --multirun
# num_domains=8
python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=mnist_md_autoencoder model.z_dim=256 model.z_dim_invariant_fraction=0.2,0.4,0.6,0.8 model.penalty_criterion="minmax" model.penalty_weight=1.0,0.1 model.hinge_loss_weight=0.0 model.autoencoder.num_channels=3 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" datamodule.dataset.num_domains=8 model.wait_steps=0,5000 model.linear_steps=1,5000 logger.wandb.tags=["mila","sweep-mnist-minmax-nohinge"] --multirun
# num_domains=16
python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=mnist_md_autoencoder model.z_dim=256 model.z_dim_invariant_fraction=0.2,0.4,0.6,0.8 model.penalty_criterion="minmax" model.penalty_weight=1.0,0.1 model.hinge_loss_weight=0.0 model.autoencoder.num_channels=3 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" datamodule.dataset.num_domains=16 model.wait_steps=0,5000 model.linear_steps=1,5000 logger.wandb.tags=["mila","sweep-mnist-minmax-nohinge"] --multirun

# ------------------------------------------------------------------------------------- #
# ---------------------- Sweep min-max penalty, with hinge loss ----------------------- #

# cpu, each sweep contains 2x2x3x2x2=48 runs
# num_domains=2
python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=mnist_md_autoencoder model.z_dim=256 model.z_dim_invariant_fraction=0.2,0.8 model.penalty_criterion="minmax" model.penalty_weight=1.0,0.1 model.hinge_loss_weight=10.0,1.0,0.1 model.autoencoder.num_channels=3 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" datamodule.dataset.num_domains=2 model.wait_steps=0,5000 model.linear_steps=1,5000 logger.wandb.tags=["mila","sweep-mnist-minmax-hinge"] --multirun
python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=mnist_md_autoencoder model.z_dim=256 model.z_dim_invariant_fraction=0.2 model.penalty_criterion="minmax" model.penalty_weight=1.0 model.hinge_loss_weight=0.1 model.autoencoder.num_channels=3 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" datamodule.dataset.num_domains=2 model.wait_steps=0 model.linear_steps=1 logger.wandb.tags=["mila","test"]
# num_domains=4
python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=mnist_md_autoencoder model.z_dim=256 model.z_dim_invariant_fraction=0.2,0.8 model.penalty_criterion="minmax" model.penalty_weight=1.0,0.1 model.hinge_loss_weight=10.0,1.0,0.1 model.autoencoder.num_channels=3 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" datamodule.dataset.num_domains=4 model.wait_steps=0,5000 model.linear_steps=1,5000 logger.wandb.tags=["mila","sweep-mnist-minmax-hinge"] --multirun
# num_domains=8
python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=mnist_md_autoencoder model.z_dim=256 model.z_dim_invariant_fraction=0.2,0.8 model.penalty_criterion="minmax" model.penalty_weight=1.0,0.1 model.hinge_loss_weight=10.0,1.0,0.1 model.autoencoder.num_channels=3 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" datamodule.dataset.num_domains=8 model.wait_steps=0,5000 model.linear_steps=1,5000 logger.wandb.tags=["mila","sweep-mnist-minmax-hinge"] --multirun
# num_domains=16
python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=mnist_md_autoencoder model.z_dim=256 model.z_dim_invariant_fraction=0.2,0.8 model.penalty_criterion="minmax" model.penalty_weight=1.0,0.1 model.hinge_loss_weight=10.0,1.0,0.1 model.autoencoder.num_channels=3 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" datamodule.dataset.num_domains=16 model.wait_steps=0,5000 model.linear_steps=1,5000 logger.wandb.tags=["mila","sweep-mnist-minmax-hinge"] --multirun

# python run_training.py ckpt_path=null model/optimizer=adam model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=mnist_md_autoencoder model.penalty_criterion="minmax" model.hinge_loss_weight=1.0 model.autoencoder.num_channels=3 model.z_dim=64 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" datamodule.dataset.num_domains=16 model.penalty_weight=1.0 logger.wandb.tags=["mila","test"]

# ------------------------------------------------------------------------------------- #
# ------------------------- Sweep std penalty, no hinge loss -------------------------- #

# cpu, each sweep contains 4x2x2x2=32 runs
# num_domains=2
python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=mnist_md_autoencoder model.z_dim=256 model.z_dim_invariant_fraction=0.2,0.4,0.6,0.8 model.penalty_criterion="stddev" model.penalty_weight=1.0,0.1 model.hinge_loss_weight=0.0 model.autoencoder.num_channels=3 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" datamodule.dataset.num_domains=2 model.wait_steps=0,5000 model.linear_steps=1,5000 logger.wandb.tags=["mila","sweep-mnist-stddev-nohinge"] --multirun
python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=mnist_md_autoencoder model.z_dim=256 model.z_dim_invariant_fraction=0.2 model.penalty_criterion="stddev" model.penalty_weight=1.0 model.hinge_loss_weight=0.0 model.autoencoder.num_channels=3 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" datamodule.dataset.num_domains=2 model.wait_steps=0 model.linear_steps=1 logger.wandb.tags=["mila","test"]
# num_domains=4
python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=mnist_md_autoencoder model.z_dim=256 model.z_dim_invariant_fraction=0.2,0.4,0.6,0.8 model.penalty_criterion="stddev" model.penalty_weight=1.0,0.1 model.hinge_loss_weight=0.0 model.autoencoder.num_channels=3 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" datamodule.dataset.num_domains=4 model.wait_steps=0,5000 model.linear_steps=1,5000 logger.wandb.tags=["mila","sweep-mnist-stddev-nohinge"] --multirun
# num_domains=8
python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=mnist_md_autoencoder model.z_dim=256 model.z_dim_invariant_fraction=0.2,0.4,0.6,0.8 model.penalty_criterion="stddev" model.penalty_weight=1.0,0.1 model.hinge_loss_weight=0.0 model.autoencoder.num_channels=3 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" datamodule.dataset.num_domains=8 model.wait_steps=0,5000 model.linear_steps=1,5000 logger.wandb.tags=["mila","sweep-mnist-stddev-nohinge"] --multirun
# num_domains=16
python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=mnist_md_autoencoder model.z_dim=256 model.z_dim_invariant_fraction=0.2,0.4,0.6,0.8 model.penalty_criterion="stddev" model.penalty_weight=1.0,0.1 model.hinge_loss_weight=0.0 model.autoencoder.num_channels=3 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" datamodule.dataset.num_domains=16 model.wait_steps=0,5000 model.linear_steps=1,5000 logger.wandb.tags=["mila","sweep-mnist-stddev-nohinge"] --multirun

# python run_training.py ckpt_path=null model/optimizer=adam model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=mnist_md_autoencoder model.penalty_criterion="stddev" model.hinge_loss_weight=0.0 model.autoencoder.num_channels=3 model.z_dim=64 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" datamodule.dataset.num_domains=16 model.penalty_weight=1.0 logger.wandb.tags=["mila","test"]

# ------------------------------------------------------------------------------------- #
# ------------------------- Sweep std penalty, with hinge loss ------------------------ #

# cpu, each sweep contains 2x2x3x2x2=48 runs
# num_domains=2
python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=mnist_md_autoencoder model.z_dim=256 model.z_dim_invariant_fraction=0.2,0.8 model.penalty_criterion="stddev" model.penalty_weight=1.0,0.1 model.hinge_loss_weight=10.0,1.0,0.1 model.autoencoder.num_channels=3 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" datamodule.dataset.num_domains=2 model.wait_steps=0,5000 model.linear_steps=1,5000 logger.wandb.tags=["mila","sweep-mnist-stddev-hinge"] --multirun
python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=mnist_md_autoencoder model.z_dim=256 model.z_dim_invariant_fraction=0.2 model.penalty_criterion="stddev" model.penalty_weight=1.0 model.hinge_loss_weight=10.0 model.autoencoder.num_channels=3 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" datamodule.dataset.num_domains=2 model.wait_steps=0 model.linear_steps=1 logger.wandb.tags=["mila","test"]
# num_domains=4
python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=mnist_md_autoencoder model.z_dim=256 model.z_dim_invariant_fraction=0.2,0.8 model.penalty_criterion="stddev" model.penalty_weight=1.0,0.1 model.hinge_loss_weight=10.0,1.0,0.1 model.autoencoder.num_channels=3 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" datamodule.dataset.num_domains=4 model.wait_steps=0,5000 model.linear_steps=1,5000 logger.wandb.tags=["mila","sweep-mnist-stddev-hinge"] --multirun
# num_domains=8
python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=mnist_md_autoencoder model.z_dim=256 model.z_dim_invariant_fraction=0.2,0.8 model.penalty_criterion="stddev" model.penalty_weight=1.0,0.1 model.hinge_loss_weight=10.0,1.0,0.1 model.autoencoder.num_channels=3 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" datamodule.dataset.num_domains=8 model.wait_steps=0,5000 model.linear_steps=1,5000 logger.wandb.tags=["mila","sweep-mnist-stddev-hinge"] --multirun
# num_domains=16
python run_training.py ckpt_path=null trainer.gpus=0 model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=mnist_md_autoencoder model.z_dim=256 model.z_dim_invariant_fraction=0.2,0.8 model.penalty_criterion="stddev" model.penalty_weight=1.0,0.1 model.hinge_loss_weight=10.0,1.0,0.1 model.autoencoder.num_channels=3 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" datamodule.dataset.num_domains=16 model.wait_steps=0,5000 model.linear_steps=1,5000 logger.wandb.tags=["mila","sweep-mnist-stddev-hinge"] --multirun

# python run_training.py ckpt_path=null model/optimizer=adam model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=mnist_md_autoencoder model.penalty_criterion="stddev" model.hinge_loss_weight=10.0 model.autoencoder.num_channels=3 model.z_dim=64 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" datamodule.dataset.num_domains=16 model.penalty_weight=1.0 logger.wandb.tags=["mila","test"]

python run_training.py ckpt_path=null model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist datamodule.dataset.num_domains=8 model=mnist_md_autoencoder model.z_dim=512 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" model.autoencoder.num_channels=3 logger.wandb.tags=["mila","mnist","reconstruction"] ~callbacks.early_stopping trainer.gpus=1 model.wait_steps=1000 model.linear_steps=1000 model.z_dim_invariant=64


# ------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------- #
# -------------------------- Disentanglement with encoded images ---------------------- #

# ------------------------------------------------------------------------------------- #
# -------------------------- min-max penalty, no hinge loss ---------------------- #
# cpu
# python run_training.py trainer.gpus=0 ckpt_path=null model.optimizer.lr=0.001 datamodule=mnist_encoded model=mnist_md_encoded_autoencoder model.z_dim=16,32,64,128 model.z_dim_invariant_fraction=0.2,0.4,0.6,0.8 model.hinge_loss_weight=0.0 model.penalty_criterion="minmax" model.penalty_weight=1.0,0.1 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["mila","sweep-encoded-mnist-minmax-nohinge"] ~callbacks.early_stopping ~callbacks.visualization_callback run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_multi_domain_mnist_8_256/2023-09-09_07-59-59" --multirun
# python run_training.py trainer.gpus=0 ckpt_path=null model.optimizer.lr=0.001 datamodule=mnist_encoded model=mnist_md_encoded_autoencoder model.z_dim=16 model.z_dim_invariant_fraction=0.8 model.hinge_loss_weight=0.0 model.penalty_criterion="minmax" model.penalty_weight=1.0 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["mila","test"] ~callbacks.early_stopping ~callbacks.visualization_callback run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_multi_domain_mnist_8_256/2023-09-09_07-59-59"

# ------------------------------------------------------------------------------------- #
# -------------------------- min-max penalty, + hinge loss ---------------------- #
# cpu
# python run_training.py trainer.gpus=0 ckpt_path=null model.optimizer.lr=0.001 datamodule=mnist_encoded model=mnist_md_encoded_autoencoder model.z_dim=16,32,64,128 model.z_dim_invariant_fraction=0.2,0.4,0.6,0.8 model.hinge_loss_weight=10.0,1.0,0.1 model.penalty_criterion="minmax" model.penalty_weight=1.0,0.1 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["mila","sweep-encoded-mnist-minmax-hinge"] ~callbacks.early_stopping ~callbacks.visualization_callback run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_multi_domain_mnist_8_256/2023-09-09_07-59-59" --multirun
# python run_training.py trainer.gpus=0 ckpt_path=null model.optimizer.lr=0.001 datamodule=mnist_encoded model=mnist_md_encoded_autoencoder model.z_dim=16 model.z_dim_invariant_fraction=0.8 model.hinge_loss_weight=1.0 model.penalty_criterion="minmax" model.penalty_weight=1.0 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["mila","test"] ~callbacks.early_stopping ~callbacks.visualization_callback run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_multi_domain_mnist_8_256/2023-09-09_07-59-59"

# ------------------------------------------------------------------------------------- #
# -------------------------- stddev penalty, no hinge loss ---------------------- #
# cpu
# python run_training.py trainer.gpus=0 ckpt_path=null model.optimizer.lr=0.001 datamodule=mnist_encoded model=mnist_md_encoded_autoencoder model.z_dim=16,32,64,128 model.z_dim_invariant_fraction=0.2,0.4,0.6,0.8 model.hinge_loss_weight=0.0 model.penalty_criterion="stddev" model.penalty_weight=1.0,0.1 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["mila","sweep-encoded-mnist-stddev-nohinge"] ~callbacks.early_stopping ~callbacks.visualization_callback run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_multi_domain_mnist_8_256/2023-09-09_07-59-59" --multirun
# python run_training.py trainer.gpus=0 ckpt_path=null model.optimizer.lr=0.001 datamodule=mnist_encoded model=mnist_md_encoded_autoencoder model.z_dim=16 model.z_dim_invariant_fraction=0.8 model.hinge_loss_weight=0.0 model.penalty_criterion="stddev" model.penalty_weight=1.0 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["mila","test"] ~callbacks.early_stopping ~callbacks.visualization_callback run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_multi_domain_mnist_8_256/2023-09-09_07-59-59"

# ------------------------------------------------------------------------------------- #
# -------------------------- stddev penalty, + hinge loss ---------------------- #
# cpu
# python run_training.py trainer.gpus=0 ckpt_path=null model.optimizer.lr=0.001 datamodule=mnist_encoded model=mnist_md_encoded_autoencoder model.z_dim=16,32,64,128 model.z_dim_invariant_fraction=0.2,0.4,0.6,0.8 model.hinge_loss_weight=10.0,1.0,0.1 model.penalty_criterion="stddev" model.penalty_weight=1.0,0.1 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["mila","sweep-encoded-mnist-stddev-hinge"] ~callbacks.early_stopping ~callbacks.visualization_callback run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_multi_domain_mnist_8_256/2023-09-09_07-59-59" --multirun
# python run_training.py trainer.gpus=0 ckpt_path=null model.optimizer.lr=0.001 datamodule=mnist_encoded model=mnist_md_encoded_autoencoder model.z_dim=16 model.z_dim_invariant_fraction=0.8 model.hinge_loss_weight=1.0 model.penalty_criterion="stddev" model.penalty_weight=1.0 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["mila","test"] ~callbacks.early_stopping ~callbacks.visualization_callback run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_multi_domain_mnist_8_256/2023-09-09_07-59-59"


conda deactivate
module purge
