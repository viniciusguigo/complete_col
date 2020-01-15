#!/bin/bash
# Run Cycle-of-Learning demo using the landing environment built on AirSim.

# Run AirSim
airsim &  # replace with your AirSim binary address
          # or place the file called airsim at your home/bin folder
sleep 8  # let AirSim start before agent connects to it

# define parameters and start CoL
TRAIN_STEPS=2000
PRETRAIN_STEPS=2000
EXP_ID='airsim_demo'
DI_LOSS=1.0
AC_Q_LOSS=1.0
Q_LOSS=1.0
N_EXPERT_TRAJS=0

python col_loss.py \
    --env HRI_AirSim_Landing-v0 \
    --data_addr data/$EXP_ID \
    --lambda_ac_di_loss $DI_LOSS \
    --lambda_ac_qloss $AC_Q_LOSS \
    --lambda_qloss $Q_LOSS \
    --train_steps $TRAIN_STEPS \
    --pretraining_steps $PRETRAIN_STEPS \
    --action_noise_sigma 0. \
    --n_expert_trajs $N_EXPERT_TRAJS \
    --dataset_addr data/20human_trajs.npz