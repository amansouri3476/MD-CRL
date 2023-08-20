#!/bin/bash

#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24G
#SBATCH --time=06:00:00
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

# resume from a mila cluster ckpt using a decoder, same slot init, etc., to train the decoder with much more compute. ALSO uses truncate bp
# python3 ../run.py mode=train model=inertia_balls_saae_contrastive_recons datamodule=inertia_balls datamodule.n_balls=2 model/optimizer=adamw model.optimizer.lr=0.0001 model/scheduler_config=reduce_on_plateau model.encoder.slot_size=64 model.encoder.resolution_dim=64 model.encoder.hid_dim=64 ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","same-init","dec","svd"] ckpt_path=null
# python3 ../run_training.py model=inertia_balls_saae_contrastive_recons_ckpt datamodule=inertia_balls datamodule.n_balls=3 model/optimizer=adamw model.optimizer.lr=0.0001 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval"] trainer.max_epochs=400
# python3 ../run_training.py model=inertia_balls_saae_contrastive_recons datamodule=inertia_balls datamodule.n_balls=3 model/optimizer=adamw model.optimizer.lr=0.0001 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","cvxpy"] trainer.max_epochs=600 ckpt_path=null

# python3 ../run.py mode=train model=inertia_balls_saae_contrastive_enc_only datamodule=inertia_balls model/optimizer=adamw model.optimizer.lr=0.0001 model/scheduler_config=polynomial model.encoder.slot_size=32 model.encoder.resolution_dim=64 model.encoder.hid_dim=64 ~callbacks.visualization_callback ckpt_path=null ~callbacks.early_stopping

# python3 ../run.py mode=train model=inertia_balls_slot_attention_ae datamodule=inertia_balls model/optimizer=adam model.optimizer.lr=0.00001 model.encoder.slot_size=32 model.encoder.resolution_dim=64 model.encoder.decoder_init_res_dim=4 model.encoder.hid_dim=64 model/scheduler_config=polynomial ckpt_path=null
# python3 run.py mode=train model=clevr_slot_attention_ae datamodule=clevr.yaml model/optimizer=adam model.optimizer.lr=0.001  model.encoder.resolution_dim=128 model.encoder.decoder_init_res_dim=8 model.encoder.hid_dim=128 # model/scheduler_config=polynomial datamodule.num_workers=0

# ablation same color
# python3 ../run_training.py model=inertia_balls_saae_contrastive_recons model.w_latent_loss=100. model.w_recons_loss=10.0 datamodule=inertia_balls datamodule.color_selection=same datamodule.n_balls=2 model/optimizer=adamw model.optimizer.lr=0.0001 ~model/scheduler_config model.encoder.slot_size=128 model.encoder.resolution_dim=64 model.encoder.hid_dim=128 ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","same_color","ablation"] ckpt_path=null

# ablation random color
# python3 ../run_training.py model=inertia_balls_saae_contrastive_recons model.w_latent_loss=100. model.w_recons_loss=10.0 datamodule=inertia_balls datamodule.color_selection=random datamodule.n_balls=2 model/optimizer=adamw model.optimizer.lr=0.0001 ~model/scheduler_config model.encoder.slot_size=128 model.encoder.resolution_dim=64 model.encoder.hid_dim=128 ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","random_color","ablation"] ckpt_path=null

# ----------------------------------------------------------------------------------------------------------------
# lin_sum_assignment as matching to speed up training enc-dec
# python3 run_training.py model=inertia_balls_saae_contrastive_recons datamodule.color_selection="cyclic_fixed" datamodule=inertia_balls datamodule.n_balls=2 model/optimizer=adamw model.optimizer.lr=0.0001 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","lin_sum","fixed_color"] ckpt_path=null trainer.max_epochs=800
# same color
# python3 run_training.py model=inertia_balls_saae_contrastive_recons datamodule.color_selection="same" datamodule=inertia_balls datamodule.n_balls=8 model/optimizer=adamw model.optimizer.lr=0.0001 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","lin_sum","same_color"] ckpt_path=null trainer.max_epochs=800

# evaluating with a ckpt
# ---------------------------------------------------- models trained on 2 balls  ----------------------------------------------------
# evaluate the model trained on 2 balls ckpt with 2 balls, vary the number of slots
python3 run_evaluation.py model=inertia_balls_saae_contrastive_recons_ckpt datamodule=inertia_balls datamodule.color_selection="cyclic_fixed" datamodule.n_balls=2 model.num_slots=2,3,4 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","constrained_lp","eval"] --multirun
python3 run_evaluation.py model=inertia_balls_saae_contrastive_recons_ckpt datamodule=inertia_balls datamodule.color_selection="cyclic_fixed" datamodule.n_balls=2 model.num_slots=2 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","constrained_lp","eval"]
# evaluate the model trained on 2 balls ckpt with 3 balls, vary the number of slots
python3 run_evaluation.py model=inertia_balls_saae_contrastive_recons_ckpt datamodule=inertia_balls datamodule.color_selection="cyclic_fixed" datamodule.n_balls=3 model.num_slots=3,4,5 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","constrained_lp","eval"] --multirun
# evaluate the model trained on 2 balls ckpt with 4 balls, vary the number of slots
python3 run_evaluation.py model=inertia_balls_saae_contrastive_recons_ckpt datamodule=inertia_balls datamodule.color_selection="cyclic_fixed" datamodule.n_balls=4 model.num_slots=4,5,6 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","constrained_lp","eval"] --multirun
# evaluate the model trained on 2 balls ckpt with 5 balls, vary the number of slots
python3 run_evaluation.py model=inertia_balls_saae_contrastive_recons_ckpt datamodule=inertia_balls datamodule.color_selection="cyclic_fixed" datamodule.n_balls=5 model.num_slots=5,6,7 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","constrained_lp","eval"] --multirun

# ---------------------------------------------------- models trained on 3 balls  ----------------------------------------------------
# evaluate the model trained on 3 balls ckpt with 3 balls, vary the number of slots
python3 run_evaluation.py model=inertia_balls_saae_contrastive_recons_ckpt datamodule=inertia_balls datamodule.color_selection="cyclic_fixed" datamodule.n_balls=3 model.num_slots=3,4,5 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","constrained_lp","eval"] --multirun
# evaluate the model trained on 3 balls ckpt with 4 balls, vary the number of slots
python3 run_evaluation.py model=inertia_balls_saae_contrastive_recons_ckpt datamodule=inertia_balls datamodule.color_selection="cyclic_fixed" datamodule.n_balls=4 model.num_slots=4,5,6 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","constrained_lp","eval"] --multirun
# evaluate the model trained on 3 balls ckpt with 5 balls, vary the number of slots
python3 run_evaluation.py model=inertia_balls_saae_contrastive_recons_ckpt datamodule=inertia_balls datamodule.color_selection="cyclic_fixed" datamodule.n_balls=5 model.num_slots=5,6,7 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","constrained_lp","eval"] --multirun
# evaluate the model trained on 3 balls ckpt with 6 balls, vary the number of slots
python3 run_evaluation.py model=inertia_balls_saae_contrastive_recons_ckpt datamodule=inertia_balls datamodule.color_selection="cyclic_fixed" datamodule.n_balls=6 model.num_slots=6,7,8 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","constrained_lp","eval"] --multirun

# ---------------------------------------------------- models trained on 4 balls  ----------------------------------------------------
# evaluate the model trained on 4 balls ckpt with 4 balls, vary the number of slots
python3 run_evaluation.py model=inertia_balls_saae_contrastive_recons_ckpt datamodule=inertia_balls datamodule.color_selection="cyclic_fixed" datamodule.n_balls=4 model.num_slots=4,5,6 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","constrained_lp","eval"] --multirun
# evaluate the model trained on 4 balls ckpt with 5 balls, vary the number of slots
python3 run_evaluation.py model=inertia_balls_saae_contrastive_recons_ckpt datamodule=inertia_balls datamodule.color_selection="cyclic_fixed" datamodule.n_balls=5 model.num_slots=5,6,7 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","constrained_lp","eval"] --multirun
# evaluate the model trained on 4 balls ckpt with 6 balls, vary the number of slots
python3 run_evaluation.py model=inertia_balls_saae_contrastive_recons_ckpt datamodule=inertia_balls datamodule.color_selection="cyclic_fixed" datamodule.n_balls=6 model.num_slots=6,7,8 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","constrained_lp","eval"] --multirun
# evaluate the model trained on 4 balls ckpt with 7 balls, vary the number of slots
python3 run_evaluation.py model=inertia_balls_saae_contrastive_recons_ckpt datamodule=inertia_balls datamodule.color_selection="cyclic_fixed" datamodule.n_balls=7 model.num_slots=7,8,9 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","constrained_lp","eval"] --multirun

# ---------------------------------------------------- models trained on 5 balls  ----------------------------------------------------
# evaluate the model trained on 5 balls ckpt with 5 balls, vary the number of slots
python3 run_evaluation.py model=inertia_balls_saae_contrastive_recons_ckpt datamodule=inertia_balls datamodule.color_selection="cyclic_fixed" datamodule.n_balls=5 model.num_slots=5,6,7 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","constrained_lp","eval"] --multirun
# evaluate the model trained on 5 balls ckpt with 6 balls, vary the number of slots
python3 run_evaluation.py model=inertia_balls_saae_contrastive_recons_ckpt datamodule=inertia_balls datamodule.color_selection="cyclic_fixed" datamodule.n_balls=6 model.num_slots=6,7,8 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","constrained_lp","eval"] --multirun
# evaluate the model trained on 5 balls ckpt with 7 balls, vary the number of slots
python3 run_evaluation.py model=inertia_balls_saae_contrastive_recons_ckpt datamodule=inertia_balls datamodule.color_selection="cyclic_fixed" datamodule.n_balls=7 model.num_slots=7,8,9 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","constrained_lp","eval"] --multirun
# evaluate the model trained on 5 balls ckpt with 8 balls, vary the number of slots
python3 run_evaluation.py model=inertia_balls_saae_contrastive_recons_ckpt datamodule=inertia_balls datamodule.color_selection="cyclic_fixed" datamodule.n_balls=8 model.num_slots=8,9,10 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","constrained_lp","eval"] --multirun

# resume from 3 balls ckpt
# python3 run_training.py model=inertia_balls_saae_contrastive_recons datamodule=inertia_balls datamodule.n_balls=3 model/optimizer=adamw model.optimizer.lr=0.00001 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","lin_sum","resumed"] trainer.max_epochs=800 ckpt_path="/home/aminm/mechanism-based-disentanglement/disentanglement_by_mechanisms/logs/training/multiruns/SA_inertia_balls_contrastive_recons/2022-08-06_13-04-15/datamodule.n_balls\=3\,model.optimizer.lr\=0.0001/checkpoints/SA_inertia_balls_contrastive_recons-epoch\=192-train_loss\=1.09-Linear_Disentanglement\=0.53.ckpt"
# resume from 4 balls ckpt
# python3 run_training.py model=inertia_balls_saae_contrastive_recons datamodule=inertia_balls datamodule.n_balls=4 model/optimizer=adamw model.optimizer.lr=0.00001 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","lin_sum","resumed"] trainer.max_epochs=800 ckpt_path="/home/aminm/mechanism-based-disentanglement/disentanglement_by_mechanisms/logs/training/multiruns/SA_inertia_balls_contrastive_recons/2022-08-06_13-04-15/datamodule.n_balls\=4\,model.optimizer.lr\=0.0001/checkpoints/SA_inertia_balls_contrastive_recons-epoch\=98-train_loss\=1.96-Linear_Disentanglement\=0.42.ckpt"
# resume from 5 balls ckpt
# python3 run_training.py model=inertia_balls_saae_contrastive_recons datamodule=inertia_balls datamodule.n_balls=5 model/optimizer=adamw model.optimizer.lr=0.00001 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","lin_sum","resumed"] trainer.max_epochs=800 ckpt_path="/home/aminm/mechanism-based-disentanglement/disentanglement_by_mechanisms/logs/training/multiruns/SA_inertia_balls_contrastive_recons/2022-08-06_13-04-15/datamodule.n_balls\=5\,model.optimizer.lr\=0.0001/checkpoints/SA_inertia_balls_contrastive_recons-epoch\=99-train_loss\=2.37-Linear_Disentanglement\=0.35.ckpt"
# resume from 6 balls ckpt
# python3 run_training.py model=inertia_balls_saae_contrastive_recons datamodule=inertia_balls datamodule.n_balls=6 model/optimizer=adamw model.optimizer.lr=0.00001 ~model/scheduler_config ~callbacks.visualization_callback ~callbacks.early_stopping logger.wandb.tags=["narval","lin_sum","resumed"] trainer.max_epochs=800 ckpt_path="/home/aminm/mechanism-based-disentanglement/disentanglement_by_mechanisms/logs/training/multiruns/SA_inertia_balls_contrastive_recons/2022-08-06_13-04-15/datamodule.n_balls\=6\,model.optimizer.lr\=0.0001/checkpoints/SA_inertia_balls_contrastive_recons-epoch\=199-train_loss\=3.04-Linear_Disentanglement\=0.19.ckpt"

deactivate
module purge
