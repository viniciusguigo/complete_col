#!/usr/bin/env python
"""
Minimum script to load expert data and evaluate CoL loss.
"""

__author__ = "Vinicius Guimaraes Goecks"
__date__ = "July 19, 2019"

# import
import argparse
import sys, os
import gym
import pybulletgym
import numpy as np
sys.path.append('../')
import hri_airsim

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", font_scale=1.25)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# DDPG DEPENDENCIES
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines import DDPG
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines.ddpg.noise import NormalActionNoise

# PPO DEPENDENCIES
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy as MlpPolicyPPO2

# COMMON DEPENDENCIES
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.gail import generate_expert_traj, ExpertDataset
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds

# CUSTOM AGENTS
from ddpg_col import DDPG_CoL2

# GLOBAL
n_steps_eval = 0
n_steps_save = 0

def main(**kwargs):
    # callback to evaluate models during training
    def eval_callback(_locals, _globals):
        """
        Evaluate model for a given number of episodes and save mean total
        reward and its standard deviation.
        """
        global n_steps_eval
        # save model after a given number of steps
        if (n_steps_eval + 1) % 20000 == 0:
            # run model for a given number of episodes
            mean_rew, std_rew = _locals['self']._eval_model(n_epi_eval=100)

            # write results to file
            _locals['self']._write_log(log_mode='eval', step=0,
                                       data=[mean_rew, std_rew])

        n_steps_eval += 1

        return True

    # callback to save models during training
    def save_model_callback(_locals, _globals):
        """
        Save training models after a given amount of training steps.
        """
        global n_steps_save
        # save model after a given number of steps
        if (n_steps_save + 1) % 5000 == 0:
            print("Saving RL model at episode {} ({})".format(
                n_steps_save+1, data_addr))
            _locals['self'].save('{}/model_step{}.pkl'.format(
                model_name, n_steps_save+1))
        n_steps_save += 1
        return True

    # ------------------------------------------------------------------
    # LOG DATA AND ENVIRONMENT
    # ------------------------------------------------------------------
    env_gym = gym.make(kwargs['env'])
    env = DummyVecEnv([lambda: env_gym])
    # # normalizes environment
    # env = VecNormalize(env, norm_obs=False, norm_reward=True)

    data_addr = kwargs['data_addr']
    # use same data_addr to save models
    model_name = kwargs['data_addr']  # kwargs['model_name']
    bc_model_name = kwargs['bc_model_name']
    dataset_addr = kwargs['dataset_addr']

    # ------------------------------------------------------------------
    # CoL LEARNING AGENT
    # ------------------------------------------------------------------
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=float(kwargs['action_noise_sigma']) * np.ones(n_actions))
    policy_kwargs = dict(
        act_fun=tf.nn.elu,
        layers=kwargs['n_layers']*[kwargs['n_neurons']])

    # ## USING DDPG_CoL2
    model = DDPG_CoL2(
            MlpPolicy, env, policy_kwargs=policy_kwargs, verbose=0,
            param_noise=None, action_noise=action_noise,
            tensorboard_log=None,
            batch_size=kwargs['batch_size'],
            dataset_addr=dataset_addr,
            lambda_ac_di_loss=kwargs['lambda_ac_di_loss'],
            lambda_ac_qloss=kwargs['lambda_ac_qloss'],
            lambda_qloss=kwargs['lambda_qloss'],
            lambda_n_step=kwargs['lambda_n_step'],
            critic_l2_reg=kwargs['critic_l2_reg'],
            actor_l2_reg=kwargs['actor_l2_reg'],
            act_prob_expert_schedule=kwargs['act_prob_expert_schedule'],
            train_steps=kwargs['train_steps'],
            bc_model_name=kwargs['bc_model_name'],
            schedule_steps=kwargs['schedule_steps'],
            normalize_returns=False,
            enable_popart=False,
            actor_lr=kwargs['actor_lr'],
            critic_lr=kwargs['critic_lr'],
            dynamic_sampling_ratio=kwargs['dynamic_sampling_ratio'],
            dynamic_loss=kwargs['dynamic_loss'],
            schedule_expert_actions=False,
            log_addr=data_addr,
            buffer_size=kwargs['memory_limit'],
            norm_reward=kwargs['norm_reward'],
            n_expert_trajs=kwargs['n_expert_trajs'],
            prioritized_replay=kwargs['prioritized_replay'],
            max_n=kwargs['max_n'])
    # initial conditions for actor and critic
    model.freeze_actor = False
    model.freeze_critic = False

    # ------------------------------------------------------------------
    # TRAIN WITH CoL LOSS
    # ------------------------------------------------------------------
    model.learn(
        total_timesteps=model.train_steps,
        callback=save_model_callback,
        dataset_addr=dataset_addr,
        pretrain_steps=kwargs['pretraining_steps'],
        pretrain_model_name=model_name + '/model_pretrained')
    model.save(model_name+'/model')

    # ------------------------------------------------------------------
    # CLOSE EXPERIMENT
    # ------------------------------------------------------------------
    model.close_logs()
    env.close()
    print('[*] Training done ({}).'.format(data_addr))
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Testing different loss function for the CoL.'))

    parser.add_argument(
        '--env', type=str, help='Environment name.', default='LunarLanderContinuous-v2')
    parser.add_argument(
        '--data_addr', type=str, help='Address to save data.', default='data/test')
    parser.add_argument(
        '--model_name', type=str, help='Address to save model.', default='data/test')
    parser.add_argument(
        '--dataset_addr', type=str, help='Address to expert dataset.', default='data/llc_sac_expert.npz')
    parser.add_argument(
        '--bc_model_name', type=str, help='Address of behavior cloning (BC) model.', default=None)
    parser.add_argument(
        '--lambda_ac_di_loss', type=float, help='Scale actor supervised loss', default=1.0)
    parser.add_argument(
        '--lambda_ac_qloss', type=float, help='Scale actor Q loss.', default=1.0)
    parser.add_argument(
        '--lambda_qloss', type=float, help='Scale critic Q loss.', default=1.0)
    parser.add_argument(
        '--lambda_n_step', type=float, help='Scale n-step loss.', default=1.0)
    parser.add_argument(
        '--critic_l2_reg', type=float, help='Critic L2 regularization.', default=0.00001)
    parser.add_argument(
        '--actor_l2_reg', type=float, help='Actor L2 regularization.', default=0.00001)
    parser.add_argument(
        '--critic_lr', type=float, help='Critic learning rate.', default=0.0001)
    parser.add_argument(
        '--actor_lr', type=float, help='Actor learning rate.', default=0.001)
    parser.add_argument(
        '--batch_size', type=int, help='RL batch size.', default=512)
    parser.add_argument(
        '--action_noise_sigma', type=float, help='Action noise sigma.', default=0.25)
    parser.add_argument(
        '--norm_reward', type=float, help='Normalize reward by a scalar.', default=1.)
    parser.add_argument(
        '--act_prob_expert_schedule', type=str, help='Scheme to schedule expert actions.', default='linear')
    parser.add_argument(
        '--train_steps', type=int, help='RL training steps.', default=2000000)
    parser.add_argument(
        '--schedule_steps', type=int, help='Action scheduling steps.', default=1)
    parser.add_argument(
        '--pretraining_steps', type=int, help='Pretraining steps.', default=100000)
    parser.add_argument(
        '--memory_limit', type=int, help='Max samples in memory.', default=500000)
    parser.add_argument(
        '--n_layers', type=int, help='Number of hidden layers.', default=3)
    parser.add_argument(
        '--n_neurons', type=int, help='Number of neurons per layer.', default=128)
    parser.add_argument(
        '--n_expert_trajs', type=int, help='Max number of expert trajectories.', default=-1)
    parser.add_argument(
        '--max_n', type=int, help='N value for n-step loss.', default=10)

    parser.add_argument('--dynamic_sampling_ratio', dest='dynamic_sampling_ratio', action='store_true')
    parser.set_defaults(dynamic_sampling_ratio=False)

    parser.add_argument('--dynamic_loss', dest='dynamic_loss', action='store_true')
    parser.set_defaults(dynamic_loss=False)
    
    parser.add_argument('--prioritized_replay', dest='prioritized_replay', action='store_true')
    parser.set_defaults(prioritized_replay=False)

    parser.add_argument('--complete_episodes', dest='complete_episodes', action='store_true')
    parser.set_defaults(complete_episodes=False)

    args = parser.parse_args()
    main(**vars(args))
