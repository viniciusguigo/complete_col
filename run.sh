#!/bin/bash
# Run Cycle-of-Learning demo using the landing environment built on AirSim.

# # Run AirSim
# airsim &
# sleep 8  # let AirSim start before agent connects to it

# define parameters and start CoL
TRAIN_STEPS=10000
SCHEDULE_STEPS=1
PRETRAIN_STEPS=2000
EXP_ID='airsim_demo'
DI_LOSS=1.0
Q_LOSS=1.0
N_STEP_LOSS=0.
NORM_REWARD=1.
N_EXPERT_TRAJS=0

python col_loss.py \
    --env HRI_AirSim_Landing-v0 \
    --data_addr data/$EXP_ID \
    --lambda_ac_di_loss $DI_LOSS \
    --lambda_ac_qloss $Q_LOSS \
    --lambda_n_step $N_STEP_LOSS \
    --train_steps $TRAIN_STEPS \
    --schedule_steps $SCHEDULE_STEPS \
    --pretraining_steps $PRETRAIN_STEPS \
    --action_noise_sigma 0. \
    --norm_reward $NORM_REWARD \
    --n_expert_trajs $N_EXPERT_TRAJS \
    --dataset_addr data/20human_trajs.npz