#!/bin/bash

#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=./slurm_out/slot-attention-%j.out
#SBATCH --error=./slurm_err/slot-attention-%j.err

module load miniconda/3
conda activate bb

export WANDB_API_KEY=1406ef3255ef2806f2ecc925a5e845e7164b5eef
wandb login

export LD_PRELOAD=/home/mila/s/sayed.mansouri-tehrani/mechanism-based-disentanglement/disentanglement_by_mechanisms/hack.so
# export WANDB_MODE=offline

# python3 run_training.py model=inertia_balls_saae_contrastive_recons model.w_latent_loss=1.0,10.0,100. model.w_recons_loss=10.0 datamodule=inertia_balls datamodule.n_balls=3,4,5,6,7,8,9,10 model/optimizer=adamw model.optimizer.lr=0.0001 model/scheduler_config=reduce_on_plateau model.encoder.slot_size=128 model.encoder.resolution_dim=64 model.encoder.hid_dim=128 ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["mila","sweep-nballs"] ckpt_path=null --multirun
# python3 run.py mode=train model=inertia_balls_saae_contrastive_recons model.w_latent_loss=0.1,1.0,10.0,100. model.w_recons_loss=0.01,0.1,1.0,10.0 model.w_similarity_loss=0.01,0.1,1.0,10.0 datamodule=inertia_balls datamodule.n_balls=2 model/optimizer=adamw model.optimizer.lr=0.0001 model/scheduler_config=reduce_on_plateau model.encoder.slot_size=128 model.encoder.resolution_dim=64 model.encoder.hid_dim=128 ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["mila","sweep-loss-w"] ckpt_path=null --multirun
# python3 run_evaluation.py datamodule=inertia_balls datamodule.n_balls=4 ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["mila","eval"]
# python3 run.py mode=train model=inertia_balls_saae_contrastive_enc_only datamodule=inertia_balls model/optimizer=adamw model.optimizer.lr=0.001 model.encoder.slot_size=32 model.encoder.resolution_dim=64 model.encoder.hid_dim=64 ~callbacks.visualization_callback ckpt_path=null ~callbacks.early_stopping model/scheduler_config=reduce_on_plateau,polynomial,null logger.wandb.tags=["lr-opt","enc","mila"] --multirun
# python3 ../run.py mode=train model=inertia_balls_slot_attention_ae datamodule=inertia_balls model/optimizer=adam,adamw model.optimizer.lr=0.00001,0.00001,0.0001,0.001  model.encoder.slot_size=32 model.encoder.resolution_dim=64 model.encoder.decoder_init_res_dim=4 model.encoder.hid_dim=64 model/scheduler_config=polynomial ckpt_path=null logger.wandb.tags=["lr-opt"] --multirun

# ablation of latent loss
# multirun
python3 run_training.py model=inertia_balls_saae_contrastive_recons_ablation_latent_loss model.encoder.slot_size=2 model.encoder.hid_dim=8,16,32,64 datamodule=inertia_balls datamodule.n_balls=2,3 model/optimizer=adamw model.optimizer.lr=0.0001 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["mila","ablation-latent-loss","sweep-hid-dim","40-core"] ckpt_path=null --multirun
# single run
# python3 run_training.py model=inertia_balls_saae_contrastive_recons_ablation_latent_loss model.encoder.slot_size=2 model.encoder.hid_dim=8 datamodule=inertia_balls datamodule.n_balls=2 model/optimizer=adamw model.optimizer.lr=0.0001 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["mila","ablation-latent-loss"] ckpt_path=null

conda deactivate
module purge
