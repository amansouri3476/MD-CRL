#!/bin/bash

#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=./slurm_out/slot-attention-%j.out
#SBATCH --error=./slurm_err/slot-attention-%j.err

module load StdEnv/2020
module load coinmp
module load python/3.8
module load scipy-stack
module load httpproxy
source /home/aminm/mbd/bin/activate

export WANDB_API_KEY=1406ef3255ef2806f2ecc925a5e845e7164b5eef
wandb login

export LD_PRELOAD=/home/aminm/mechanism-based-disentanglement/disentanglement_by_mechanisms/hack.so
export WANDB_MODE=offline

# python3 run.py mode=train model=inertia_balls_saae_contrastive_recons model.w_latent_loss=0.1,1.0,10.0,100. model.w_recons_loss=0.01,0.1,1.0,10.0 model.w_similarity_loss=0.0,0.01,0.1,1.0,10.0 datamodule=inertia_balls datamodule.n_balls=2 model/optimizer=adamw model.optimizer.lr=0.0001 model/scheduler_config=reduce_on_plateau model.encoder.slot_size=128 model.encoder.resolution_dim=64 model.encoder.hid_dim=128 ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","sweep-loss-w"] ckpt_path=null --multirun

# sweep over n_balls
# python3 run_training.py model=inertia_balls_saae_contrastive_recons datamodule=inertia_balls datamodule.n_balls=3,4,5,6,7,8,9,10 model.w_latent_loss=100. model.w_recons_loss=10.0 model/optimizer=adamw model.optimizer.lr=0.0001 model/scheduler_config=reduce_on_plateau model.encoder.slot_size=128 model.encoder.resolution_dim=64 model.encoder.hid_dim=128 ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","sweep-nball"] ckpt_path=null --multirun

# python3 run.py mode=train model=inertia_balls_saae_contrastive_enc_only datamodule=inertia_balls datamodule.n_balls=1,2,3,4 model/optimizer=adamw model.optimizer.lr=0.0001 model/scheduler_config=reduce_on_plateau model.encoder.slot_size=128,256 model.encoder.resolution_dim=64 model.encoder.hid_dim=128,256 ~callbacks.visualization_callback ckpt_path=null ~callbacks.early_stopping logger.wandb.tags=["enc","narval","sweep-2"] --multirun
# python3 run_training.py model=inertia_balls_saae_contrastive_recons datamodule=inertia_balls datamodule.n_balls=3 model.w_latent_loss=100. model.w_recons_loss=10.0 model/optimizer=adamw model.optimizer.lr=0.0001 model/scheduler_config=reduce_on_plateau model.encoder.slot_size=128 model.encoder.resolution_dim=64 model.encoder.hid_dim=128 ckpt_path=null

# loading best ckpts for different datamodule.n_balls, no lr reduction on plateau, no ckpt_path=null
# python3 run_training.py model=inertia_balls_saae_contrastive_recons_ckpt datamodule=inertia_balls datamodule.n_balls=4,5,6,7,8,9,10 model/optimizer=adamw model.optimizer.lr=0.0001 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","sweep-nball","continue"] trainer.max_epochs=400 --multirun
# python3 run_training.py model=inertia_balls_saae_contrastive_recons_ckpt datamodule=inertia_balls datamodule.n_balls=3 model/optimizer=adamw model.optimizer.lr=0.0001 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval"] trainer.max_epochs=400

# ablation of latent loss
# multirun
# python3 run_training.py model=inertia_balls_saae_contrastive_recons_ablation_latent_loss model.encoder.slot_size=2 model.encoder.hid_dim=8,16,32,64 datamodule=inertia_balls datamodule.n_balls=2,3 model/optimizer=adamw model.optimizer.lr=0.0001 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","ablation-latent-loss","sweep-hid-dim"] ckpt_path=null --multirun
# single run
# python3 run_training.py model=inertia_balls_saae_contrastive_recons_ablation_latent_loss model.encoder.slot_size=2 model.encoder.hid_dim=8 datamodule=inertia_balls datamodule.n_balls=2 model/optimizer=adamw model.optimizer.lr=0.0001 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","ablation-latent-loss"] ckpt_path=null

# ------------------------------------------
# lin_sum_assignment as matching to speed up training enc-dec
# python3 run_training.py model=inertia_balls_saae_contrastive_recons datamodule=inertia_balls datamodule.n_balls=2,3,4,5,6,7,8 datamodule.color_selection="cyclic_fixed" model/optimizer=adamw model.optimizer.lr=0.0002 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","lin_sum","fixed_color"] ckpt_path=null trainer.max_epochs=400 --multirun
# python3 run_training.py model=inertia_balls_saae_contrastive_recons datamodule=inertia_balls datamodule.n_balls=2,3,4,5,6,7,8 datamodule.color_selection="same" model/optimizer=adamw model.optimizer.lr=0.0002 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","lin_sum","same","high_dim"] ckpt_path=null trainer.max_epochs=1600 --multirun


# ------------------------------------------
# running baseline (vanilla slot attention)
# fixed_color
# python3 run_training.py model=inertia_balls_slot_attention_ae datamodule.color_selection="cyclic_fixed" datamodule=inertia_balls datamodule.n_balls=2,3,4,5,6 model.optimizer.lr=0.0002 callbacks=vanilla_slot_attention ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","vanilla_SA","fixed_color"] ckpt_path=null trainer.max_epochs=800 --multirun
# same color
# python3 run_training.py model=inertia_balls_slot_attention_ae datamodule=inertia_balls datamodule/dataset=position_offset_only datamodule.color_selection="same" datamodule.n_balls=2,3,4,5,6 model.optimizer.lr=0.0002 callbacks=vanilla_slot_attention ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","vanilla_SA","same"] ckpt_path=null trainer.max_epochs=800 --multirun
# python3 run_training.py model=inertia_balls_slot_attention_ae datamodule=inertia_balls datamodule/dataset=position_offset_only datamodule.color_selection="same" datamodule.n_balls=3,4 model.optimizer.lr=0.0002,0.00002 callbacks=vanilla_slot_attention ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","vanilla_SA","same"] ckpt_path=null trainer.max_epochs=800 --multirun
# ------------------------------------------
# sweep August 29. For 3 balls trained for 24h, sweep over the following
# learning_rate=2e-5,2e-4, wait_steps=0,5000, linear_steps=1,5000, slot_size=hid_dim=64,128, w_latent_loss=10.0,100.0, w_recons_loss=1.0,10.0
# model=inertia_balls_saae_contrastive_recons, datamodule=inertia_balls, datamodule/dataset=position_offset_only, datamodule.color_selection="same", model.z_dim=2, datamodule.n_balls=3
# python3 run_training.py model=inertia_balls_saae_contrastive_recons model.z_dim=2 datamodule=inertia_balls datamodule/dataset=position_offset_only datamodule.color_selection="same" datamodule.n_balls=3 model/optimizer=adamw model.optimizer.lr=0.0002,0.00002 model.wait_steps=0,5000 model.linear_steps=1,5000 model.encoder.slot_size=64,128 model.w_latent_loss=10.0,100.0 model.w_recons_loss=1.0,10.0 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","hp_search_same","same"] ckpt_path=null trainer.max_epochs=700 --multirun

# ------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------- #
# --------------------------------- SPARSE CLEVR RUNS --------------------------------- #
# ---------------------- Vanilla Slot Attention ------------------------- #

python3 run_training.py datamodule=sparse_clevr datamodule.n_balls=2 datamodule.batch_size=256 model=inertia_balls_slot_attention_ae model.encoder.slot_size=64 model.encoder.n_channels=4 model.optimizer.lr=0.0000006,0.000001,,0.000002,0.00001,0.00006,0.0002 model.additional_logger.logging_interval=400 callbacks=vanilla_slot_attention ~callbacks.visualization_callback ~callbacks.early_stopping callbacks.model_checkpoint.monitor="train_loss" logger.wandb.tags=["narval","vanilla_SA","clevr"] trainer.max_epochs=15000 ckpt_path="/home/aminm/scratch/logs/training/runs/slot_attention_xycsl_autoencoder_2_cyclic_fixed/2023-02-16_20-31-10/checkpoints/slot_attention_xycsl_autoencoder_2_cyclic_fixed-epoch\=9208-train_loss\=0.000051-Linear_Disentanglement_regression\=0.42-Permutation_Disentanglement_regression\=0.53.ckpt" --multirun
# -------------------------- Disentanglement ---------------------------- #
# n_balls = 2

# zdim = 5 x,y,c,s,l
# with slot attention ckpt
# /home/aminm/scratch/logs/training/runs/slot_attention_xycsl_autoencoder_2_cyclic_fixed/2023-02-16_20-31-10/checkpoints/slot_attention_xycsl_autoencoder_2_cyclic_fixed-epoch\=6533-train_loss\=0.000058-Linear_Disentanglement_regression\=0.40-Permutation_Disentanglement_regression\=0.50.ckpt
# python3 run_training.py model=inertia_balls_saae_contrastive_recons datamodule=sparse_clevr datamodule.dataset.z_dim=5 datamodule.dataset.properties_list=["x","y","c","s","l"] datamodule.n_balls=2 datamodule.batch_size=256 datamodule.start_idx.train=0 datamodule.num_samples.train=8000 datamodule.start_idx.valid=8000 datamodule.num_samples.valid=1000 datamodule.start_idx.test=9000 datamodule.num_samples.test=1000 model.z_dim=5 model.disentangle_z_dim=5 model.encoder.slot_size=64 model.encoder.n_channels=4 model.optimizer.lr=0.00006,0.0002,0.0004 model.w_recons_loss=100.0 model.w_latent_loss=1.0,10.0,100.0 model.wait_steps=0 model.linear_steps=1 model.latent_matching="argmin" model.ball_matching=True model.double_matching=True model.known_mechanism=True model.known_action=True model.use_all_balls_mcc=True,False model.rm_background_in_matching=True model.pair_recons=True model.encoder_freeze=False model.additional_logger.logging_interval=400 callbacks=default ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping callbacks.model_checkpoint.monitor="train_loss" logger.wandb.tags=["narval","disentanglement","clevr"] trainer.max_epochs=2000 ckpt_path=null model.pl_model_ckpt_path="/home/aminm/scratch/logs/training/runs/slot_attention_xycsl_autoencoder_2_cyclic_fixed/2023-02-16_20-31-10/checkpoints/slot_attention_xycsl_autoencoder_2_cyclic_fixed-epoch\=6533-train_loss\=0.000058-Linear_Disentanglement_regression\=0.40-Permutation_Disentanglement_regression\=0.50.ckpt" --multirun
# python3 run_training.py model=inertia_balls_saae_contrastive_recons datamodule=sparse_clevr datamodule.dataset.z_dim=5 datamodule.dataset.properties_list=["x","y","c","s","l"] datamodule.n_balls=2 datamodule.batch_size=256 datamodule.start_idx.train=0 datamodule.num_samples.train=8000 datamodule.start_idx.valid=8000 datamodule.num_samples.valid=1000 datamodule.start_idx.test=9000 datamodule.num_samples.test=1000 model.z_dim=5 model.disentangle_z_dim=5 model.encoder.slot_size=64 model.encoder.n_channels=4 model.optimizer.lr=0.0002 model.w_recons_loss=100.0 model.w_latent_loss=10.0 model.wait_steps=0 model.linear_steps=1 model.latent_matching="argmin" model.ball_matching=True model.double_matching=True model.known_mechanism=True model.known_action=True model.use_all_balls_mcc=True model.rm_background_in_matching=True model.pair_recons=True model.encoder_freeze=True,False model.additional_logger.logging_interval=400 callbacks=default ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping callbacks.model_checkpoint.monitor="train_loss" logger.wandb.tags=["narval","disentanglement","clevr"] trainer.max_epochs=2000 ckpt_path=null model.pl_model_ckpt_path="/home/aminm/scratch/logs/training/runs/slot_attention_xycsl_autoencoder_2_cyclic_fixed/2023-02-16_20-31-10/checkpoints/slot_attention_xycsl_autoencoder_2_cyclic_fixed-epoch\=6533-train_loss\=0.000058-Linear_Disentanglement_regression\=0.40-Permutation_Disentanglement_regression\=0.50.ckpt" --multirun
# with complete model ckpt
# python3 run_training.py model=inertia_balls_saae_contrastive_recons datamodule=sparse_clevr datamodule.dataset.z_dim=5 datamodule.dataset.properties_list=["x","y","c","s","l"] datamodule.n_balls=2 datamodule.batch_size=256 datamodule.start_idx.train=0 datamodule.num_samples.train=8000 datamodule.start_idx.valid=8000 datamodule.num_samples.valid=1000 datamodule.start_idx.test=9000 datamodule.num_samples.test=1000 model.z_dim=5 model.disentangle_z_dim=5 model.encoder.slot_size=64 model.encoder.n_channels=4 model.optimizer.lr=0.0002 model.w_recons_loss=100.0 model.w_latent_loss=10.0 model.wait_steps=0 model.linear_steps=1 model.latent_matching="argmin" model.ball_matching=False model.double_matching=False model.known_mechanism=True model.known_action=True model.use_all_balls_mcc=False model.rm_background_in_matching=True model.pair_recons=True model.encoder_freeze=False model.additional_logger.logging_interval=400 callbacks=default ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping callbacks.model_checkpoint.monitor="train_loss" logger.wandb.tags=["narval","disentanglement","no_matching","clevr"] trainer.max_epochs=2000 ckpt_path="/home/aminm/scratch/logs/training/runs/slot_attention_xycsl_autoencoder_2_cyclic_fixed/2023-02-16_20-31-10/checkpoints/slot_attention_xycsl_autoencoder_2_cyclic_fixed-epoch\=6108-train_loss\=0.000061-Linear_Disentanglement_regression\=0.42-Permutation_Disentanglement_regression\=0.51.ckpt" # model.target_property_indices=[0,1,2,3,4]
# ------------------------------------------------------ #

deactivate
module purge
