#!/usr/bin/env python
"""
Mostly modifies RL agents from Stable Baselines to fit our needs for the
Cycle-of-Learning.

Original code:
https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/ddpg/ddpg.py
"""

__author__ = "Vinicius Guimaraes Goecks"
__date__ = "June 12, 2019"

from functools import reduce
import os
import time
from collections import deque
import pickle
import warnings
import copy
import threading

import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from mpi4py import MPI
import pickle
import pygame

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

from stable_baselines import logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.mpi_running_mean_std import RunningMeanStd
from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from replay_buffer_col import PrioritizedReplayBuffer

# SAC DEPENDENCIES (used as expert model)
from stable_baselines import SAC
from stable_baselines import DDPG
# training epis: 5e6
# policy_kwargs = dict(act_fun=tf.nn.elu, layers=[32,32])

class JoystickAgent(object):
    """ Reads an Xbox joystick to control OpenAI Gym envs.

    Nov 25, 2018: 2 continuous actions using left and right stick.
    """
    def __init__(self, env):
        self.name = 'JoystickAgent'
        # env information
        self.env = env

        # setup xbox joystick using pygame
        pygame.init()

        # Initialize the connected joysticks
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        print('Joysticks connected: {}'.format(joystick_count))

        # only use first connected
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print('Using joystick: {}'.format(self.joystick.get_name()))
       

    def act(self, ob=None, reward=None, done=None):
        """ Reads and return left and right sticks."""
        pygame.event.pump()

        # check if human controlling (top left button)
        self.human_control = self.joystick.get_button(4)

        # case for 2 actions
        if self.env.action_space.shape[0] == 2:
            # read left stick (vertical)
            left_stick_vert = -self.joystick.get_axis(1)

            # read right stick (horizontal)
            right_stick_horz = self.joystick.get_axis(3)

            # concatenate joystick values
            action = np.array([left_stick_vert, right_stick_horz])

        # case for 4 actions
        elif self.env.action_space.shape[0] == 4:
            # read left stick (vertical)
            left_stick_vert = self.joystick.get_axis(1)
            left_stick_horz = self.joystick.get_axis(0)

            # read right stick (horizontal)
            right_stick_horz = self.joystick.get_axis(3)
            right_stick_vert = -self.joystick.get_axis(4)

            # concatenate joystick values
            action = np.array([
                right_stick_horz, right_stick_vert,
                left_stick_vert, left_stick_horz])

        return [action]

    def close(self):
        """ Stop any thread (if any) or save additional data (if any)"""
        pygame.quit()
        print('Joystick connection closed.')


def tag_episode_reward_logger(rew_acc, rewards, masks, writer, steps,
                              tag_name='episode_reward'):
    """
    calculates the cumulated episode reward, and prints to tensorflow log the output
    :param rew_acc: (np.array float) the total running reward
    :param rewards: (np.array float) the rewards
    :param masks: (np.array bool) the end of episodes
    :param writer: (TensorFlow Session.writer) the writer to log to
    :param steps: (int) the current timestep
    :return: (np.array float) the updated total running reward
    :return: (np.array float) the updated total running reward
    """
    with tf.variable_scope("environment_info", reuse=True):
        for env_idx in range(rewards.shape[0]):
            dones_idx = np.sort(np.argwhere(masks[env_idx]))

            if len(dones_idx) == 0:
                rew_acc[env_idx] += sum(rewards[env_idx])
            else:
                rew_acc[env_idx] += sum(rewards[env_idx, :dones_idx[0, 0]])
                summary = tf.Summary(value=[tf.Summary.Value(tag=tag_name,
                                     simple_value=rew_acc[env_idx])])
                writer.add_summary(summary, steps + dones_idx[0, 0])
                for k in range(1, len(dones_idx[:, 0])):
                    rew_acc[env_idx] = sum(rewards[env_idx,
                                           dones_idx[k-1, 0]:dones_idx[k, 0]])
                    summary = tf.Summary(value=[tf.Summary.Value(tag=tag_name,
                                         simple_value=rew_acc[env_idx])])
                    writer.add_summary(summary, steps + dones_idx[k, 0])
                rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[-1, 0]:])

    return rew_acc


def normalize(tensor, stats):
    """
    normalize a tensor using a running mean and std

    :param tensor: (TensorFlow Tensor) the input tensor
    :param stats: (RunningMeanStd) the running mean and std of the input to normalize
    :return: (TensorFlow Tensor) the normalized tensor
    """
    if stats is None:
        return tensor
    return (tensor - stats.mean) / stats.std


def denormalize(tensor, stats):
    """
    denormalize a tensor using a running mean and std

    :param tensor: (TensorFlow Tensor) the normalized tensor
    :param stats: (RunningMeanStd) the running mean and std of the input to normalize
    :return: (TensorFlow Tensor) the restored tensor
    """
    if stats is None:
        return tensor
    return tensor * stats.std + stats.mean


def reduce_std(tensor, axis=None, keepdims=False):
    """
    get the standard deviation of a Tensor

    :param tensor: (TensorFlow Tensor) the input tensor
    :param axis: (int or [int]) the axis to itterate the std over
    :param keepdims: (bool) keep the other dimensions the same
    :return: (TensorFlow Tensor) the std of the tensor
    """
    return tf.sqrt(reduce_var(tensor, axis=axis, keepdims=keepdims))


def reduce_var(tensor, axis=None, keepdims=False):
    """
    get the variance of a Tensor

    :param tensor: (TensorFlow Tensor) the input tensor
    :param axis: (int or [int]) the axis to itterate the variance over
    :param keepdims: (bool) keep the other dimensions the same
    :return: (TensorFlow Tensor) the variance of the tensor
    """
    tensor_mean = tf.reduce_mean(tensor, axis=axis, keepdims=True)
    devs_squared = tf.square(tensor - tensor_mean)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def get_target_updates(_vars, target_vars, tau, verbose=0):
    """
    get target update operations

    :param _vars: ([TensorFlow Tensor]) the initial variables
    :param target_vars: ([TensorFlow Tensor]) the target variables
    :param tau: (float) the soft update coefficient (keep old values, between 0 and 1)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :return: (TensorFlow Operation, TensorFlow Operation) initial update, soft update
    """
    if verbose >= 2:
        logger.info('setting up target updates ...')
    soft_updates = []
    init_updates = []
    assert len(_vars) == len(target_vars)
    for var, target_var in zip(_vars, target_vars):
        if verbose >= 2:
            logger.info('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.assign(target_var, var))
        soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
    assert len(init_updates) == len(_vars)
    assert len(soft_updates) == len(_vars)
    return tf.group(*init_updates), tf.group(*soft_updates)


def get_perturbable_vars(scope):
    """
    Get the trainable variables that can be perturbed when using
    parameter noise.

    :param scope: (str) tensorflow scope of the variables
    :return: ([tf.Variables])
    """
    return [var for var in tf_util.get_trainable_vars(scope) if 'LayerNorm' not in var.name]


def get_perturbed_actor_updates(actor, perturbed_actor, param_noise_stddev, verbose=0):
    """
    Get the actor update, with noise.

    :param actor: (str) the actor
    :param perturbed_actor: (str) the pertubed actor
    :param param_noise_stddev: (float) the std of the parameter noise
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :return: (TensorFlow Operation) the update function
    """
    assert len(tf_util.get_globals_vars(actor)) == len(tf_util.get_globals_vars(perturbed_actor))
    assert len(get_perturbable_vars(actor)) == len(get_perturbable_vars(perturbed_actor))

    updates = []
    for var, perturbed_var in zip(tf_util.get_globals_vars(actor), tf_util.get_globals_vars(perturbed_actor)):
        if var in get_perturbable_vars(actor):
            if verbose >= 2:
                logger.info('  {} <- {} + noise'.format(perturbed_var.name, var.name))
            # Add gaussian noise to the parameter
            updates.append(tf.assign(perturbed_var,
                                     var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
        else:
            if verbose >= 2:
                logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
            updates.append(tf.assign(perturbed_var, var))
    assert len(updates) == len(tf_util.get_globals_vars(actor))
    return tf.group(*updates)


class DDPG_CoL(OffPolicyRLModel):
    """
    Deep Deterministic Policy Gradient (DDPG) model modified for the Cycle-of-Learning.

    Original DDPG implementation from stable-baselines: https://github.com/hill-a/stable-baselines

    DDPG: https://arxiv.org/pdf/1509.02971.pdf

    :param policy: (DDPGPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param memory_policy: (ReplayBuffer) the replay buffer
        (if None, default to baselines.deepq.replay_buffer.ReplayBuffer)

        .. deprecated:: 2.6.0
            This parameter will be removed in a future version
    :param eval_env: (Gym Environment) the evaluation environment (can be None)
    :param nb_train_steps: (int) the number of training steps
    :param nb_rollout_steps: (int) the number of rollout steps
    :param nb_eval_steps: (int) the number of evalutation steps
    :param param_noise: (AdaptiveParamNoiseSpec) the parameter noise type (can be None)
    :param action_noise: (ActionNoise) the action noise type (can be None)
    :param param_noise_adaption_interval: (int) apply param noise every N steps
    :param tau: (float) the soft update coefficient (keep old values, between 0 and 1)
    :param normalize_returns: (bool) should the critic output be normalized
    :param enable_popart: (bool) enable pop-art normalization of the critic output
        (https://arxiv.org/pdf/1602.07714.pdf), normalize_returns must be set to True.
    :param normalize_observations: (bool) should the observation be normalized
    :param batch_size: (int) the size of the batch for learning the policy
    :param observation_range: (tuple) the bounding values for the observation
    :param return_range: (tuple) the bounding values for the critic output
    :param critic_l2_reg: (float) l2 regularizer coefficient
    :param actor_lr: (float) the actor learning rate
    :param critic_lr: (float) the critic learning rate
    :param clip_norm: (float) clip the gradients (disabled if None)
    :param reward_scale: (float) the value the reward should be scaled by
    :param render: (bool) enable rendering of the environment
    :param render_eval: (bool) enable rendering of the evalution environment
    :param memory_limit: (int) the max number of transitions to store, size of the replay buffer

        .. deprecated:: 2.6.0
            Use `buffer_size` instead.

    :param buffer_size: (int) the max number of transitions to store, size of the replay buffer
    :param prioritized_replay: (bool) if True prioritized replay buffer will be used.
    :param prioritized_replay_alpha: (float)alpha parameter for prioritized replay buffer.
        It determines how much prioritization is used, with alpha=0 corresponding to the uniform case.
    :param prioritized_replay_beta0: (float) initial value of beta for prioritized replay buffer
    :param prioritized_replay_beta_iters: (int) number of iterations over which beta will be annealed from initial
            value to 1.0. If set to None equals to max_timesteps.
    :param prioritized_replay_eps: (float) epsilon to add to the TD errors when updating priorities.
    :param random_exploration: (float) Probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for DDPG normally but can help exploring when using HER + DDPG.
        This hack was present in the original OpenAI Baselines repo (DDPG + HER)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    """

    def __init__(self, policy, env, gamma=0.99, memory_policy=None, eval_env=None, nb_train_steps=50,
                 nb_rollout_steps=100, nb_eval_steps=100, param_noise=None, action_noise=None,
                 normalize_observations=False, tau=0.001, batch_size=128, param_noise_adaption_interval=50,
                 normalize_returns=False, enable_popart=False, observation_range=(-5., 5.), critic_l2_reg=1e-9, actor_l2_reg=1e-9,
                 return_range=(-np.inf, np.inf), actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.,
                 render=False, render_eval=False, memory_limit=None, buffer_size=50000, random_exploration=0.0, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,dataset_addr=None,
                 lambda_ac_di_loss=1.0, lambda_ac_qloss=1.0, lambda_qloss=1.0, lambda_n_step=1.0, act_prob_expert_schedule=None,
                 train_steps=0, schedule_steps=0, bc_model_name=None, dynamic_sampling_ratio=False,
                 log_addr='data/test', schedule_expert_actions=False, dynamic_loss=False, csv_log_interval=10,
                 norm_reward=1., n_expert_trajs=-1, prioritized_replay=False,
                 prioritized_replay_alpha=0.3, prioritized_replay_beta0=1.0, prioritized_replay_beta_iters=None,
                 prioritized_replay_eps=1e-6,max_n=10, live_plot=False):

        super(DDPG_CoL, self).__init__(policy=policy, env=env, replay_buffer=None,
                                verbose=verbose, policy_base=DDPGPolicy,
                                requires_vec_env=False, policy_kwargs=policy_kwargs)

        # Parameters.
        self.gamma = gamma
        self.tau = tau
        if memory_policy is not None:
            warnings.warn("memory_policy will be removed in a future version (v3.x.x) "
                          "it is now ignored and replaced with ReplayBuffer", DeprecationWarning)

        if memory_limit is not None:
            warnings.warn("memory_limit will be removed in a future version (v3.x.x) "
                          "use buffer_size instead", DeprecationWarning)
            buffer_size = memory_limit
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.return_range = return_range
        self.observation_range = observation_range
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.critic_l2_reg = critic_l2_reg
        self.actor_l2_reg = actor_l2_reg
        self.eval_env = eval_env
        self.render = render
        self.render_eval = render_eval
        self.nb_eval_steps = nb_eval_steps
        self.param_noise_adaption_interval = param_noise_adaption_interval
        self.nb_train_steps = nb_train_steps
        self.nb_rollout_steps = nb_rollout_steps
        self.memory_limit = memory_limit
        self.buffer_size = buffer_size
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.random_exploration = random_exploration
        self.dataset_addr = dataset_addr
        self.lambda_ac_di_loss = lambda_ac_di_loss
        self.initial_lambda_ac_di_loss = lambda_ac_di_loss
        self.lambda_ac_qloss = lambda_ac_qloss
        self.initial_lambda_ac_qloss = lambda_ac_qloss
        self.initial_lambda_qloss = lambda_qloss
        self.lambda_qloss = lambda_qloss
        self.lambda_n_step = lambda_n_step
        self.act_prob_expert_schedule = act_prob_expert_schedule
        self.train_steps = train_steps
        self.bc_model_name = bc_model_name
        self.schedule_steps = schedule_steps
        self.max_samples_expert = memory_limit
        self.dynamic_sampling_ratio = dynamic_sampling_ratio
        self.dynamic_loss = dynamic_loss
        self.log_addr = log_addr
        self.schedule_expert_actions = schedule_expert_actions
        self.csv_log_interval = csv_log_interval
        self.norm_reward = norm_reward
        self.n_expert_trajs = n_expert_trajs
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_iters = prioritized_replay_beta_iters
        self.max_n = max_n
        self.live_plot = live_plot
        self.joystick = JoystickAgent(env)

        # init
        self.graph = None
        self.stats_sample = None
        self.replay_buffer = None
        self.policy_tf = None
        self.target_init_updates = None
        self.target_soft_updates = None
        self.critic_loss = None
        self.critic_n_step_loss = None
        self.critic_1_step_loss = None
        self.critic_n_step_loss_val = None
        self.critic_1_step_loss_val = None
        self.critic_grads = None
        self.critic_optimizer = None
        self.sess = None
        self.stats_ops = None
        self.stats_names = None
        self.perturbed_actor_tf = None
        self.perturb_policy_ops = None
        self.perturb_adaptive_policy_ops = None
        self.adaptive_policy_distance = None
        self.actor_loss = None
        self.actor_grads = None
        self.actor_optimizer = None
        self.old_std = None
        self.old_mean = None
        self.renormalize_q_outputs_op = None
        self.obs_rms = None
        self.ret_rms = None
        self.target_policy = None
        self.actor_tf = None
        self.normalized_critic_tf = None
        self.critic_tf = None
        self.normalized_critic_with_actor_tf = None
        self.critic_with_actor_tf = None
        self.target_q = None
        self.obs_train = None
        self.action_train_ph = None
        self.obs_target = None
        self.action_target = None
        self.obs_noise = None
        self.action_noise_ph = None
        self.obs_adapt_noise = None
        self.action_adapt_noise = None
        self.terminals_ph = None
        self.rewards = None
        self.actions = None
        self.critic_target = None
        self.critic_target_n = None
        self.param_noise_stddev = None
        self.param_noise_actor = None
        self.adaptive_param_noise_actor = None
        self.params = None
        self.summary = None
        self.episode_reward = None
        self.tb_seen_steps = None
        self.target_params = None
        self.obs_rms_params = None
        self.ret_rms_params = None
        self.actor_qloss_val = None
        self.actor_loss_di_val = None
        self.actor_reg_val = None
        self.critic_reg_val = None
        self.target_q_val = None
        self.n_trajs_complete = None

        if _init_setup_model:
            self.setup_model()

    def _parallel_training(self):
        """ Starts parallel training on a separate thread when human 
        intervention/demonstrations is triggered by joystick button.
        """
        # define training limits
        min_expert_samples = 1  # 80 samples per episode, on average
        actor_loss_limit = 0.005
        n_parallel_steps = 200

        while True:
            # period for checking if human is pressing the trigger
            time.sleep(10)

            # inform user about number of collected samples in case it is less than the minimum
            if self.replay_buffer_expert.__len__() < min_expert_samples:
                print('[*] Expert samples: {}. Minimum required: {}.'.format(
                    self.replay_buffer_expert.__len__(), min_expert_samples
                ))

            # human pressing trigger, trains with human data for a fixed number of steps
            # (also check for a minimun number of samples on the buffer)
            if self.prev_samples_buffer < self.replay_buffer_expert.__len__() and self.replay_buffer_expert.__len__() > min_expert_samples:
                # update actor and critic with expert data
                self.prev_samples_buffer = self.replay_buffer_expert.__len__()
                with self.sess.as_default():
                    for i in range(n_parallel_steps):
                        # trains actor and critic
                        # note: pretrain_mode=True ensures we only use expert data
                        critic_loss, actor_loss = self._train_step(
                            step=i, writer=None, pretrain_mode=True)
                        self._update_target_net()

                        if i % 50 == 0:
                            print('** Parallel training step {}+/{} | Actor loss: {:.4f} | Critic loss: {:.4f} | Actor expert loss: {:.4f} | Expert samples: {} **'.format(
                                i+1, n_parallel_steps, actor_loss, critic_loss, self.actor_loss_di_val, self.replay_buffer_expert.__len__()))

                        # early stop based on actor and critic loss
                        if self.actor_loss_di_val < actor_loss_limit:
                            break

                print('** Completed parallel training. **')

    def _open_logs(self):
        """
        Creates CSV log files to store evaluation, losses, and reward data.
        """
        # create log files
        os.makedirs(self.log_addr, exist_ok=True)
        # self.eval_log = open('{}/eval_log.csv'.format(
        #     self.log_addr), 'w')
        self.loss_log = open('{}/loss_log.csv'.format(
            self.log_addr), 'w')
        self.reward_log = open('{}/reward_log.csv'.format(
            self.log_addr), 'w')

        # write labels
        # self.eval_log.write('{},{},{}\n'.format(
        #                 'step',
        #                 'mean_rew',
        #                 'std_rew'))

        self.loss_log.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                            'step',
                            'actor_loss',
                            'actor_loss_di_val',
                            'actor_loss_di_val_scaled',
                            'actor_qloss_val',
                            'actor_qloss_val_scaled',
                            'actor_reg_val',
                            'critic_loss',
                            'critic_reg_val',
                            'act_prob_expert',
                            'critic_1_step_loss_val'))

        self.reward_log.write('{},{},{}\n'.format(
                            'step',
                            'reward_val',
                            'unsc_reward_val'))

    def close_logs(self):
        """
        Closes CSV log files to store evaluation, losses, and reward data.
        """
        # self.eval_log.close()
        self.loss_log.close()
        self.reward_log.close()

    def _write_log(self, log_mode, step, data):
        """
        Writes data to specific log files based on selected log_mode.
        
        Inputs:
            log_modes: 'eval', 'loss', 'reward'.
        """
        if log_mode == 'eval':
            mean_rew = data[0]
            std_rew = data[1]
            self.eval_log.write('{},{},{}\n'.format(
                        step,
                        mean_rew,
                        std_rew))
        
        elif log_mode == 'loss':
            actor_loss = data[0]
            critic_loss = data[1]
            self.loss_log.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                            step,
                            actor_loss,
                            self.actor_loss_di_val,
                            self.actor_loss_di_val*self.lambda_ac_di_loss,
                            self.actor_qloss_val,
                            self.actor_qloss_val*self.lambda_ac_qloss,
                            self.actor_reg_val,
                            critic_loss*self.lambda_qloss,
                            self.critic_reg_val,
                            self.act_prob_expert,
                            self.critic_1_step_loss_val))

        elif log_mode == 'reward':
            reward_val = data[0]
            unsc_reward_val = data[1]
            self.reward_log.write('{},{},{}\n'.format(
                            step,
                            reward_val,
                            unsc_reward_val))


    def pretrain(self, dataset_addr, pretrain_steps, max_samples_expert):
        """
        Pretraining step for the Cycle-of-Learning: loads demonstrations and
        perform actor and critic updates.
        """
        # parse demonstrations from expert dataset
        print('Loading ', dataset_addr)
        dataset = dict(np.load(dataset_addr))
        n_samples = dataset['obs'].shape[0]
        print('Found {} expert samples.'.format(n_samples))

        # limit the number of expert trajectories
        # finds where each trajectory starts and get number of samples
        if self.n_expert_trajs == 0:  # no expert trajectories
            print('[INFO] Using {} expert trajectories'.format(self.n_expert_trajs))
            epi_starts = np.where(dataset['episode_starts'] == True)[0]
            max_samples_expert = 1

        elif self.n_expert_trajs > 0:  # using just a few trajectories
            print('[INFO] Using {} expert trajectories'.format(self.n_expert_trajs))
            epi_starts = np.where(dataset['episode_starts'] == True)[0]
            max_samples_expert = epi_starts[self.n_expert_trajs]

        else: # -1 means all trajs
            print('[INFO] Using all expert trajectories')
            epi_starts = np.where(dataset['episode_starts'] == True)[0]
            max_samples_expert = n_samples

        # if no experts samples will be used, just copy current buffer and end
        # the pretraining phase
        if max_samples_expert == 1:
            # create a copy of buffer to keep expert data forever
            self.replay_buffer_expert = copy.deepcopy(self.replay_buffer)
            self.prev_samples_buffer = 0
            print('[*] No expert samples.')
            return
        else:
            # define inital samples on buffer, which triggers parallel updates
            self.prev_samples_buffer = max_samples_expert
        
        # on PER case, make sure expert demos are not overwritten
        if self.prioritized_replay:
            self.replay_buffer.expert_idx = max_samples_expert

        # reduce expert dataset size so it matches the max number of samples
        # or the maximum number of desired trajectories
        dataset_obs = dataset['obs'][0:max_samples_expert, :]
        dataset_obs1 = dataset['obs1'][0:max_samples_expert, :]
        dataset_actions = dataset['actions'][0:max_samples_expert, :]
        dataset_rewards = dataset['rewards'][0:max_samples_expert, :]
        dataset_dones = dataset['done'][0:max_samples_expert]

        # scaled rewards following the dqfd paper
        dataset_rewards = self._scale_reward(dataset_rewards)

        # add to memory_expert buffer
        # do not use last samples because we will have no next obs
        print('Adding expert samples to memory...')
        if max_samples_expert > n_samples:
            max_samples_expert = n_samples
        print('Using {} samples'.format(max_samples_expert))

        # populate memory buffer (agent, anget n_step buffer, and expert)
        for i in range(max_samples_expert):
            # agent buffer (1-step)
            obs0 = dataset_obs[i, :]
            obs1 = dataset_obs1[i, :]
            action = dataset_actions[i, :]
            reward = dataset_rewards[i, :]
            terminal1 = dataset_dones[i]
            self.replay_buffer.add(obs0, action, reward, obs1, terminal1)


        # create a copy of buffer to keep expert data forever
        self.replay_buffer_expert = copy.deepcopy(self.replay_buffer)

        # update actor and critic with expert data
        with self.sess.as_default():
            for i in range(pretrain_steps):
                # trains actor and critic
                critic_loss, actor_loss = self._train_step(
                    step=i-pretrain_steps, writer=None, pretrain_mode=True)
                self._update_target_net()

                if i % 50 == 0:
                    print('** Pretraining step {}+/{} | Actor loss: {:.4f} | Critic loss: {:.4f} | Actor expert loss: {:.4f} | Expert samples: {} **'.format(
                        i, pretrain_steps, actor_loss, critic_loss, self.actor_loss_di_val, self.replay_buffer_expert.__len__()))

                # log losses values during pretraining at a fixed rate
                if i % self.csv_log_interval == 0:
                    self._write_log(log_mode='loss', step=i-pretrain_steps,
                                    data=[actor_loss, critic_loss])

            print('Pretraining completed! | Final actor loss: {} | Final critic loss: {}'.format(
                        actor_loss, critic_loss))
            
            print('Completed pretraining.')

    def _scale_reward(self, r):
        '''
        Scales rewards to improve learning.
        scale_rew = np.sign(r)*np.log(1+np.abs(r))
        '''
        return r/self.norm_reward

    def _eval_model(self, n_epi_eval=10):
        '''
        Evaluates current model over a given number of episodes
        '''
        # evaluate agent's performance after pretraining by computing
        # the average reward over a specific number of episodes
        total_rew_eval = np.zeros(n_epi_eval)
        for i in range(n_epi_eval):
            obs = self.env.reset()
            total_rew = 0
            while True:
                action, _ = self.predict(obs)
                obs, reward, done, _ = self.env.step(action)
                total_rew += reward
                if done:
                    total_rew_eval[i] = total_rew
                    break

        # comparison between performance of generated trajectories and
        # pretrained model
        mean_rew = np.mean(total_rew_eval)
        std_rew = np.std(total_rew_eval)
        # print('* Mean reward = {:.2f} +- {:.2f}'.format(mean_rew, std_rew))

        return mean_rew, std_rew

    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        # Rescale
        deterministic_action = self.actor_tf * np.abs(self.action_space.low)
        return policy.obs_ph, self.actions, deterministic_action

    def setup_model(self, loading_model=False):
        with SetVerbosity(self.verbose):

            assert isinstance(self.action_space, gym.spaces.Box), \
                "Error: DDPG cannot output a {} action space, only spaces.Box is supported.".format(self.action_space)
            assert issubclass(self.policy, DDPGPolicy), "Error: the input policy for the DDPG model must be " \
                                                        "an instance of DDPGPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.single_threaded_session(graph=self.graph)

                # initialize CSV log files
                # (stores evaluation, losses, and reward data during training)
                if not loading_model:
                    self._open_logs()

                with tf.variable_scope("bc_model", reuse=False):
                    # Defines behavior cloning (expert) model to schedule actions
                    if self.bc_model_name is not None:
                        self.bc_model = SAC.load(self.bc_model_name)

                with tf.variable_scope("schedule", reuse=False):
                    # Set schedule for expert actions
                    self.current_t_ph = tf.placeholder(tf.float32, shape=(), name="current_t_ph")
                    if self.act_prob_expert_schedule is not None:
                        self._setup_schedule()
                        tf.summary.scalar('act_prob_expert', self.act_prob_expert_op)
                    else:
                        self.act_prob_expert_op = self.current_t_ph*0.

                with tf.variable_scope("input", reuse=False):
                    # Observation normalization.
                    if self.normalize_observations:
                        with tf.variable_scope('obs_rms'):
                            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
                    else:
                        self.obs_rms = None

                    # Return normalization.
                    if self.normalize_returns:
                        with tf.variable_scope('ret_rms'):
                            self.ret_rms = RunningMeanStd()
                    else:
                        self.ret_rms = None

                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space, 1, 1, None,
                                                 **self.policy_kwargs)

                    # Create target networks.
                    self.target_policy = self.policy(self.sess, self.observation_space, self.action_space, 1, 1, None,
                                                     **self.policy_kwargs)
                    self.obs_target = self.target_policy.obs_ph
                    self.action_target = self.target_policy.action_ph

                    normalized_obs = tf.clip_by_value(normalize(self.policy_tf.processed_obs, self.obs_rms),
                                                       self.observation_range[0], self.observation_range[1])
                    normalized_next_obs = tf.clip_by_value(normalize(self.target_policy.processed_obs, self.obs_rms),
                                                       self.observation_range[0], self.observation_range[1])

                    if self.param_noise is not None:
                        # Configure perturbed actor.
                        self.param_noise_actor = self.policy(self.sess, self.observation_space, self.action_space, 1, 1,
                                                             None, **self.policy_kwargs)
                        self.obs_noise = self.param_noise_actor.obs_ph
                        self.action_noise_ph = self.param_noise_actor.action_ph

                        # Configure separate copy for stddev adoption.
                        self.adaptive_param_noise_actor = self.policy(self.sess, self.observation_space,
                                                                      self.action_space, 1, 1, None,
                                                                      **self.policy_kwargs)
                        self.obs_adapt_noise = self.adaptive_param_noise_actor.obs_ph
                        self.action_adapt_noise = self.adaptive_param_noise_actor.action_ph

                    # Inputs.
                    self.obs_train = self.policy_tf.obs_ph
                    self.action_train_ph = self.policy_tf.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.actions = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape, name='actions')
                    self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
                    self.critic_target_n = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target_n')
                    self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')

                # Create networks and core TF parts that are shared across setup parts.
                with tf.variable_scope("model", reuse=False):
                    self.actor_tf = self.policy_tf.make_actor(normalized_obs)
                    self.normalized_critic_tf = self.policy_tf.make_critic(normalized_obs, self.actions)
                    self.normalized_critic_with_actor_tf = self.policy_tf.make_critic(normalized_obs,
                                                                                      self.actor_tf,
                                                                                      reuse=True)
                # Noise setup
                if self.param_noise is not None:
                    self._setup_param_noise(normalized_obs)

                with tf.variable_scope("target", reuse=False):
                    critic_target = self.target_policy.make_critic(normalized_next_obs,
                                                                   self.target_policy.make_actor(normalized_next_obs))

                with tf.variable_scope("loss", reuse=False):
                    self.critic_tf = denormalize(
                        tf.clip_by_value(self.normalized_critic_tf, self.return_range[0], self.return_range[1]),
                        self.ret_rms)

                    self.critic_with_actor_tf = denormalize(
                        tf.clip_by_value(self.normalized_critic_with_actor_tf,
                                         self.return_range[0], self.return_range[1]),
                        self.ret_rms)

                    q_next_obs = denormalize(critic_target, self.ret_rms)
                    self.target_q = self.rewards + (1. - self.terminals_ph) * self.gamma * q_next_obs

                    tf.summary.scalar('critic_target', tf.reduce_mean(self.critic_target))
                    if self.full_tensorboard_log:
                        tf.summary.histogram('critic_target', self.critic_target)

                    # Set up parts.
                    if self.normalize_returns and self.enable_popart:
                        self._setup_popart()
                    self._setup_stats()
                    self._setup_target_network_updates()

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('rewards', tf.reduce_mean(self.rewards))
                    tf.summary.scalar('param_noise_stddev', tf.reduce_mean(self.param_noise_stddev))

                    if self.full_tensorboard_log:
                        tf.summary.histogram('rewards', self.rewards)
                        tf.summary.histogram('param_noise_stddev', self.param_noise_stddev)
                        if len(self.observation_space.shape) == 3 and self.observation_space.shape[0] in [1, 3, 4]:
                            tf.summary.image('observation', self.obs_train)
                        else:
                            tf.summary.histogram('observation', self.obs_train)

                with tf.variable_scope("Adam_mpi", reuse=False):
                    self._setup_actor_optimizer()
                    self._setup_critic_optimizer()
                    tf.summary.scalar('actor_qloss', self.actor_qloss)
                    tf.summary.scalar('actor_loss_di', self.actor_loss_di)
                    tf.summary.scalar('actor_qloss_scaled', self.actor_qloss*self.lambda_ac_qloss)
                    tf.summary.scalar('actor_loss_di_scaled', self.actor_loss_di*self.lambda_ac_di_loss)
                    tf.summary.scalar('actor_loss', self.actor_loss)
                    # tf.summary.scalar('actor_reg', self.actor_reg)
                    tf.summary.scalar('critic_loss', self.critic_loss*self.lambda_qloss)
                    tf.summary.scalar('critic_reg', self.critic_reg)

                self.params = tf_util.get_trainable_vars("model") \
                    + tf_util.get_trainable_vars('noise/') + tf_util.get_trainable_vars('noise_adapt/')

                self.target_params = tf_util.get_trainable_vars("target")
                self.obs_rms_params = [var for var in tf.global_variables()
                                        if "obs_rms" in var.name]
                self.ret_rms_params = [var for var in tf.global_variables()
                                        if "ret_rms" in var.name]

                with self.sess.as_default():
                    self._initialize(self.sess)

                self.summary = tf.summary.merge_all()

    def _setup_schedule(self):
        """
        schedule expert actions and lambda decay
        """
        self.act_prob_expert = 1.
        if self.act_prob_expert_schedule == 'linear':
            self.act_prob_expert_rate = 1/self.schedule_steps
            self.act_prob_expert_op = tf.clip_by_value(1.-self.act_prob_expert_rate*self.current_t_ph,0,1)

        else:
            self.act_prob_expert_rate = 0.
            self.act_prob_expert_op = 1.

    def _setup_target_network_updates(self):
        """
        set the target update operations
        """
        init_updates, soft_updates = get_target_updates(tf_util.get_trainable_vars('model/'),
                                                        tf_util.get_trainable_vars('target/'), self.tau,
                                                        self.verbose)
        self.target_init_updates = init_updates
        self.target_soft_updates = soft_updates

    def _setup_param_noise(self, normalized_obs):
        """
        Setup the parameter noise operations

        :param normalized_obs: (TensorFlow Tensor) the normalized observation
        """
        assert self.param_noise is not None

        with tf.variable_scope("noise", reuse=False):
            self.perturbed_actor_tf = self.param_noise_actor.make_actor(normalized_obs)

        with tf.variable_scope("noise_adapt", reuse=False):
            adaptive_actor_tf = self.adaptive_param_noise_actor.make_actor(normalized_obs)

        with tf.variable_scope("noise_update_func", reuse=False):
            if self.verbose >= 2:
                logger.info('setting up param noise')
            self.perturb_policy_ops = get_perturbed_actor_updates('model/pi/', 'noise/pi/', self.param_noise_stddev,
                                                                  verbose=self.verbose)

            self.perturb_adaptive_policy_ops = get_perturbed_actor_updates('model/pi/', 'noise_adapt/pi/',
                                                                           self.param_noise_stddev,
                                                                           verbose=self.verbose)
            self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.actor_tf - adaptive_actor_tf)))

    def _setup_actor_optimizer(self):
        """
        setup the optimizer for the actor
        """
        if self.verbose >= 2:
            logger.info('setting up actor optimizer')

        # setup supervised learning loss (demonstrations and interventions)
        self.obs_ph, self.actions_ph, self.deterministic_actions_ph = self._get_pretrain_placeholders()
        self.actor_loss_di = tf.reduce_mean(tf.square(self.actions_ph - self.deterministic_actions_ph))

        # standard actor based on q-values
        self.actor_qloss = -tf.reduce_mean(self.critic_with_actor_tf)

        # add different loss components to actor
        self.actor_loss = self.actor_qloss*self.lambda_ac_qloss + self.actor_loss_di*self.lambda_ac_di_loss

        # actor regularization
        if self.actor_l2_reg > 0.:
            actor_reg_vars = [var for var in tf_util.get_trainable_vars('model/pi/')
                               if 'bias' not in var.name and 'b' not in var.name]
            if self.verbose >= 2:
                for var in actor_reg_vars:
                    logger.info('  regularizing actor: {}'.format(var.name))
                logger.info('  applying actor l2 regularization with {}'.format(self.actor_l2_reg))
            self.actor_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.actor_l2_reg),
                weights_list=actor_reg_vars
            )
            self.actor_loss += self.actor_reg

        actor_shapes = [var.get_shape().as_list() for var in tf_util.get_trainable_vars('model/pi/')]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        if self.verbose >= 2:
            logger.info('  actor shapes: {}'.format(actor_shapes))
            logger.info('  actor params: {}'.format(actor_nb_params))
        self.actor_grads = tf_util.flatgrad(self.actor_loss, tf_util.get_trainable_vars('model/pi/'),
                                            clip_norm=self.clip_norm)
        self.actor_optimizer = MpiAdam(var_list=tf_util.get_trainable_vars('model/pi/'), beta1=0.9, beta2=0.999,
                                       epsilon=1e-08)

    def _setup_critic_optimizer(self):
        """
        setup the optimizer for the critic
        """
        if self.verbose >= 2:
            logger.info('setting up critic optimizer')

        # setup 1_step loss
        normalized_critic_target_tf = tf.clip_by_value(normalize(self.critic_target, self.ret_rms),
                                                       self.return_range[0], self.return_range[1])
        self.critic_1_step_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - normalized_critic_target_tf))

        # combine losses
        self.critic_loss = self.critic_1_step_loss*self.lambda_qloss
            


        # critic regularization
        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in tf_util.get_trainable_vars('model/qf/')
                               if 'bias' not in var.name and 'qf_output' not in var.name and 'b' not in var.name]
            if self.verbose >= 2:
                for var in critic_reg_vars:
                    logger.info('  regularizing critic: {}'.format(var.name))
                logger.info('  applying critic l2 regularization with {}'.format(self.critic_l2_reg))
            self.critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += self.critic_reg

        critic_shapes = [var.get_shape().as_list() for var in tf_util.get_trainable_vars('model/qf/')]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        if self.verbose >= 2:
            logger.info('  critic shapes: {}'.format(critic_shapes))
            logger.info('  critic params: {}'.format(critic_nb_params))
        self.critic_grads = tf_util.flatgrad(self.critic_loss, tf_util.get_trainable_vars('model/qf/'),
                                             clip_norm=self.clip_norm)
        self.critic_optimizer = MpiAdam(var_list=tf_util.get_trainable_vars('model/qf/'), beta1=0.9, beta2=0.999,
                                        epsilon=1e-08)

    def _setup_popart(self):
        """
        setup pop-art normalization of the critic output

        See https://arxiv.org/pdf/1602.07714.pdf for details.
        Preserving Outputs Precisely, while Adaptively Rescaling Targets”.
        """
        self.old_std = tf.placeholder(tf.float32, shape=[1], name='old_std')
        new_std = self.ret_rms.std
        self.old_mean = tf.placeholder(tf.float32, shape=[1], name='old_mean')
        new_mean = self.ret_rms.mean

        self.renormalize_q_outputs_op = []
        for out_vars in [[var for var in tf_util.get_trainable_vars('model/qf/') if 'qf_output' in var.name],
                         [var for var in tf_util.get_trainable_vars('target/qf/') if 'qf_output' in var.name]]:
            assert len(out_vars) == 2
            # wieght and bias of the last layer
            weight, bias = out_vars
            assert 'kernel' in weight.name
            assert 'bias' in bias.name
            assert weight.get_shape()[-1] == 1
            assert bias.get_shape()[-1] == 1
            self.renormalize_q_outputs_op += [weight.assign(weight * self.old_std / new_std)]
            self.renormalize_q_outputs_op += [bias.assign((bias * self.old_std + self.old_mean - new_mean) / new_std)]

    def _setup_stats(self):
        """
        Setup the stat logger for DDPG.
        """
        ops = [
            tf.reduce_mean(self.critic_tf),
            reduce_std(self.critic_tf),
            tf.reduce_mean(self.critic_with_actor_tf),
            reduce_std(self.critic_with_actor_tf),
            tf.reduce_mean(self.actor_tf),
            reduce_std(self.actor_tf)
        ]
        names = [
            'reference_Q_mean',
            'reference_Q_std',
            'reference_actor_Q_mean',
            'reference_actor_Q_std',
            'reference_action_mean',
            'reference_action_std'
        ]

        if self.normalize_returns:
            ops += [self.ret_rms.mean, self.ret_rms.std]
            names += ['ret_rms_mean', 'ret_rms_std']

        if self.normalize_observations:
            ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
            names += ['obs_rms_mean', 'obs_rms_std']

        ops += [tf.reduce_mean(self.critic_tf)]
        names += ['reference_Q_mean']
        ops += [reduce_std(self.critic_tf)]
        names += ['reference_Q_std']

        ops += [tf.reduce_mean(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_mean']
        ops += [reduce_std(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_std']

        ops += [tf.reduce_mean(self.actor_tf)]
        names += ['reference_action_mean']
        ops += [reduce_std(self.actor_tf)]
        names += ['reference_action_std']

        if self.param_noise:
            ops += [tf.reduce_mean(self.perturbed_actor_tf), reduce_std(self.perturbed_actor_tf)]
            names += ['reference_perturbed_action_mean', 'reference_perturbed_action_std']

        self.stats_ops = ops
        self.stats_names = names

    def _policy(self, obs, apply_noise=True, compute_q=True, prob_dropout=0.0):
        """
        Get the actions and critic output, from a given observation

        :param obs: ([float] or [int]) the observation
        :param apply_noise: (bool) enable the noise
        :param compute_q: (bool) compute the critic output
        :return: ([float], float) the action and critic value
        """
        obs = np.array(obs).reshape((-1,) + self.observation_space.shape)
        feed_dict = {self.obs_train: obs, self.policy_tf.prob_dropout_ph: prob_dropout}
        if self.param_noise is not None and apply_noise:
            actor_tf = self.perturbed_actor_tf
            feed_dict[self.obs_noise] = obs
        else:
            actor_tf = self.actor_tf

        if compute_q:
            action, q_value = self.sess.run([actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
        else:
            action = self.sess.run(actor_tf, feed_dict=feed_dict)
            q_value = None

        action = action.flatten()
        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == action.shape
            action += noise
        action = np.clip(action, -1, 1)
        return action, q_value

    def _store_transition(self, obs, action, reward, next_obs, done):
        """
        Store a transition in the replay buffer

        :param obs: ([float] or [int]) the last observation
        :param action: ([float]) the action
        :param reward: (float] the reward
        :param next_obs: ([float] or [int]) the current observation
        :param done: (bool) Whether the episode is over
        """
        reward *= self.reward_scale
        self.replay_buffer.add(obs, action, reward, next_obs, float(done))
        if self.normalize_observations:
            self.obs_rms.update(np.array([obs]))


    def _combine_batches(self, batch1, batch2):
        """
        Combines data from different batches in a single one, key by key.

        Keys:
            'obs0'
            'obs1'
            'rewards'
            'actions'
            'terminals1'        
        """
        # create empty batch
        combined_batch = dict()

        # combine each key
        combined_batch['obs0'] = np.vstack((batch1['obs0'], batch2['obs0']))
        combined_batch['obs1'] = np.vstack((batch1['obs1'], batch2['obs1']))
        combined_batch['rewards'] = np.vstack((batch1['rewards'], batch2['rewards']))
        combined_batch['actions'] = np.vstack((batch1['actions'], batch2['actions']))
        combined_batch['terminals1'] = np.vstack((batch1['terminals1'], batch2['terminals1']))

        return combined_batch

    def _train_step(self, step, writer, log=False, pretrain_mode=False):
        """
        run a step of training from batch

        :param step: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :param log: (bool) whether or not to log to metadata
        :return: (float, float) critic loss, actor loss
        """
        # fix step for scheduling
        if step is None:
            step = 0.

        # update lambda values to scale loss terms
        # log makes sure value is only update once, even after doing multiple
        # training steps within the same episode
        if self.dynamic_loss and not pretrain_mode and log:
            self.lambda_ac_di_loss = self.act_prob_expert*self.initial_lambda_ac_di_loss

        # computer batch_size for expert and agent buffer based on scheduling
        # mechanic create for action scheduling
        if self.dynamic_sampling_ratio and log:
            batch_size_expert = int(self.act_prob_expert*self.batch_size)
            batch_size_agent = self.batch_size - batch_size_expert
            self.last_batch_size_expert = batch_size_expert
            self.last_batch_size_agent = batch_size_agent
            
        elif not self.dynamic_sampling_ratio and not pretrain_mode:
            # using fixed sampling ratio
            batch_size_expert = int(0.25*self.batch_size)
            batch_size_agent = int(0.75*self.batch_size)

        elif pretrain_mode:
            # only sample from expert during pretraining
            batch_size_expert = self.batch_size
            batch_size_agent = 0

        else:
            # do not update batch sizes, only repeat last one
            batch_size_expert = self.last_batch_size_expert
            batch_size_agent = self.last_batch_size_agent

        if self.prioritized_replay:
            # used on DDPGfD implementation
            experience = self.replay_buffer.sample(
                self.batch_size, beta=self.beta_schedule.value(self.num_timesteps))
            (obs, actions, rewards, next_obs, terminals, weights, batch_idxes) = experience

            # Reshape to match previous behavior and placeholder shape
            rewards = rewards.reshape(-1, 1)
            terminals = terminals.reshape(-1, 1)

            # still samples from episode but the loss computed is multiplied by zero when
            # using PER (for the DDPGfD implementation)
            # implementated that way to guarantee compatibility with CoL code
            # sampling expert data
            expert_experience = self.replay_buffer_expert.sample(
                batch_size=batch_size_expert, beta=self.beta_schedule.value(self.num_timesteps))
            (expert_obs0, expert_actions, expert_rewards, expert_obs1, expert_terminals1, expert_weights, expert_batch_size) = expert_experience

        else:
            # get a batch of data from buffer
            obs0, actions, rewards, obs1, terminals1 = self.replay_buffer.sample(batch_size=batch_size_agent)
            weights, batch_idxes = np.ones_like(rewards), None
            
            # Reshape to match previous behavior and placeholder shape
            rewards = rewards.reshape(-1, 1)
            terminals1 = terminals1.reshape(-1, 1)

            # sampling expert data (only if there's data in the buffer)
            if self.replay_buffer_expert.__len__() > 0:
                expert_obs0, expert_actions, expert_rewards, expert_obs1, expert_terminals1 = self.replay_buffer_expert.sample(batch_size=batch_size_expert)
                # Reshape to match previous behavior and placeholder shape
                expert_rewards = expert_rewards.reshape(-1, 1)
                expert_terminals1 = expert_terminals1.reshape(-1, 1)

                # stack data from expert and agent data
                # (skip when one of the batches has 0 samples)
                if batch_size_agent == 0:
                    # only have expert samples (pre-training phase)
                    obs = expert_obs0
                    actions = expert_actions
                    rewards = expert_rewards
                    next_obs = expert_obs1
                    terminals = expert_terminals1
                else:
                    # have data from expert and agent
                    obs = np.vstack((obs0, expert_obs0))
                    actions = np.vstack((actions, expert_actions))
                    rewards = np.vstack((rewards, expert_rewards))
                    next_obs = np.vstack((obs1, expert_obs1))
                    terminals = np.vstack((terminals1, expert_terminals1))
            else:
                # only agent date
                obs = obs0
                actions = actions
                rewards = rewards
                next_obs = obs1
                terminals = terminals1
        

        if self.normalize_returns and self.enable_popart:
            old_mean, old_std, target_q = self.sess.run([self.ret_rms.mean, self.ret_rms.std, self.target_q],
                                                        feed_dict={
                                                            self.obs_target: next_obs,
                                                            self.rewards: rewards,
                                                            self.terminals_ph: terminals
                                                        })
            self.ret_rms.update(target_q.flatten())
            self.sess.run(self.renormalize_q_outputs_op, feed_dict={
                self.old_std: np.array([old_std]),
                self.old_mean: np.array([old_mean]),
            })

        else:
            target_q = self.sess.run(self.target_q, feed_dict={
                self.obs_target: next_obs,
                self.rewards: rewards,
                self.terminals_ph: terminals
            })

        # Get all gradients and perform a synced update.
        self.target_q_val = target_q
        ops = [self.actor_grads, self.actor_loss, self.critic_grads,
               self.critic_loss, self.act_prob_expert_op,
               self.actor_qloss, self.actor_loss_di,
               self.actor_reg, self.critic_reg,
               self.critic_1_step_loss]
        
        # if does not have expert data, do not use supervised loss
        if self.replay_buffer_expert.__len__() > 0:
            td_map = {
                self.current_t_ph: float(step),
                self.obs_ph: expert_obs0,         # supervised loss states
                self.actions_ph: expert_actions, # supervised loss actions
                self.obs_train: obs,
                self.actions: actions,
                self.action_train_ph: actions,
                self.rewards: rewards,
                self.critic_target: target_q,
                self.param_noise_stddev: 0 if self.param_noise is None else self.param_noise.current_stddev
            }
        else:
            td_map = {
                self.current_t_ph: float(step),
                self.obs_train: obs,
                self.actions: actions,
                self.action_train_ph: actions,
                self.rewards: rewards,
                self.critic_target: target_q,
                self.param_noise_stddev: 0 if self.param_noise is None else self.param_noise.current_stddev
            }
        
        if writer is not None:
            # run loss backprop with summary if the step_id was not already logged (can happen with the right
            # parameters as the step value is only an estimate)
            if self.full_tensorboard_log and log and step not in self.tb_seen_steps:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, actor_grads, actor_loss, critic_grads, critic_loss, self.act_prob_expert = \
                    self.sess.run([self.summary] + ops, td_map, options=run_options, run_metadata=run_metadata)

                writer.add_run_metadata(run_metadata, 'step%d' % step)
                self.tb_seen_steps.append(step)
            else:
                summary, actor_grads, actor_loss, critic_grads, critic_loss, \
                    self.act_prob_expert, self.actor_qloss_val, \
                    self.actor_loss_di_val, self.actor_reg_val, \
                    self.critic_reg_val, \
                    self.critic_1_step_loss_val = self.sess.run([self.summary] + ops, td_map)
            writer.add_summary(summary, step)
        else:
            actor_grads, actor_loss, critic_grads, critic_loss, \
                self.act_prob_expert, self.actor_qloss_val, \
                self.actor_loss_di_val, self.actor_reg_val, \
                self.critic_reg_val, \
                self.critic_1_step_loss_val = self.sess.run(ops, td_map)

        # # perform multiple gradient updates
        # for i in range(10):

        # double check if training actor or not
        # only calls actor_optimizer if we are training actor
        # print('[*] Freeze actor? {}'.format(self.freeze_actor))
        # self.display_actor_vars()
        if not self.freeze_actor:
            self.actor_optimizer.update(actor_grads, learning_rate=self.actor_lr)
        # print('[*] Freeze critic? {}'.format(self.freeze_critic))
        if not self.freeze_critic:
            self.critic_optimizer.update(critic_grads, learning_rate=self.critic_lr)

        # update priorities
        if self.prioritized_replay:
            td_errors = self.target_q_val
            new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
            self.replay_buffer.update_priorities(batch_idxes, new_priorities)

        return critic_loss, actor_loss

    def _initialize(self, sess):
        """
        initialize the model parameters and optimizers

        :param sess: (TensorFlow Session) the current TensorFlow session
        """
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.actor_optimizer.sync()
        self.critic_optimizer.sync()
        self.sess.run(self.target_init_updates)

    def _update_target_net(self):
        """
        run target soft update operation
        """
        self.sess.run(self.target_soft_updates)

    def _get_stats(self):
        """
        Get the mean and standard deviation of the model's inputs and outputs

        :return: (dict) the means and stds
        """
        if self.stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            if self.prioritized_replay:
                experience = self.replay_buffer.sample(
                    self.batch_size, beta=self.beta_schedule.value(self.num_timesteps))
                (obs, actions, rewards, next_obs, terminals, weights, batch_idxes) = experience
            else:
                obs, actions, rewards, next_obs, terminals = self.replay_buffer.sample(batch_size=self.batch_size)
            self.stats_sample = {
                'obs': obs,
                'actions': actions,
                'rewards': rewards,
                'next_obs': next_obs,
                'terminals': terminals
            }

        feed_dict = {
            self.actions: self.stats_sample['actions']
        }

        for placeholder in [self.action_train_ph, self.action_target, self.action_adapt_noise, self.action_noise_ph]:
            if placeholder is not None:
                feed_dict[placeholder] = self.stats_sample['actions']

        for placeholder in [self.obs_train, self.obs_target, self.obs_adapt_noise, self.obs_noise]:
            if placeholder is not None:
                feed_dict[placeholder] = self.stats_sample['obs']

        values = self.sess.run(self.stats_ops, feed_dict=feed_dict)

        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))

        if self.param_noise is not None:
            stats = {**stats, **self.param_noise.get_stats()}

        return stats

    def _adapt_param_noise(self):
        """
        calculate the adaptation for the parameter noise

        :return: (float) the mean distance for the parameter noise
        """
        if self.param_noise is None:
            return 0.

        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        obs, *_ = self.replay_buffer.sample(batch_size=self.batch_size)
        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        distance = self.sess.run(self.adaptive_policy_distance, feed_dict={
            self.obs_adapt_noise: obs, self.obs_train: obs,
            self.param_noise_stddev: self.param_noise.current_stddev,
        })

        mean_distance = MPI.COMM_WORLD.allreduce(distance, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        self.param_noise.adapt(mean_distance)
        return mean_distance

    def _reset(self):
        """
        Reset internal state after an episode is complete.
        """
        if self.action_noise is not None:
            self.action_noise.reset()
        if self.param_noise is not None:
            self.sess.run(self.perturb_policy_ops, feed_dict={
                self.param_noise_stddev: self.param_noise.current_stddev,
            })

    def _actor_mc_dropout(self, obs, n_samples=20):
        """
        Performs multiple forward passes in the actor network to compute mean
        and stddev of action and predicted q-values.
        Use that to compute uncertainty: in this case, the highest stddev
        representes the total uncertainty.

        Since actions are scaled [-1,1], stddev of 1 means 100% uncertainty.
        """
        # arrays to store sampled actions and qvalues
        action_uncerts = np.zeros((n_samples, self.env.action_space.shape[0]))
        q_value_uncerts = np.zeros((n_samples, 1))

        # compute multiple actions and q-values
        for i in range(n_samples):
            action_uncerts[i], q_value_uncerts[i] = self._policy(
                                obs, apply_noise=True, compute_q=True, prob_dropout=0.1)

        # compute action uncertainty (stddev scaled from 0 to 100%)
        action_uncert = np.std(action_uncerts, axis=0)*100
        action_uncert = np.amax(np.clip(action_uncert, 0,100))

        # compute qval uncertainty (just stddev)
        q_value_uncert = np.std(q_value_uncerts, axis=0)

        return action_uncert, q_value_uncert

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100,
              tb_log_name="DDPG", reset_num_timesteps=True, dataset_addr=None,
              pretrain_steps=None, max_samples_expert=None, pretrain_model_name=None,
              replay_wrapper=None):

        print('Training {} for {} timesteps...'.format(
            self.log_addr, total_timesteps))

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)


        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn(seed)

            # initialize memory buffer
            if self.prioritized_replay:
                self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
                if self.prioritized_replay_beta_iters is None:
                    prioritized_replay_beta_iters = total_timesteps
                else:
                    prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
                self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                    initial_p=self.prioritized_replay_beta0,
                                                    final_p=1.0)
            else:
                self.replay_buffer = ReplayBuffer(self.buffer_size)
                self.beta_schedule = None

            if replay_wrapper is not None:
                assert not self.prioritized_replay, "Prioritized replay buffer is not supported by HER"
                self.replay_buffer = replay_wrapper(self.replay_buffer)

            # a list for tensorboard logging, to prevent logging with the same step number, if it already occured
            self.tb_seen_steps = []

            rank = MPI.COMM_WORLD.Get_rank()
            # we assume symmetric actions.
            assert np.all(np.abs(self.env.action_space.low) == self.env.action_space.high)
            if self.verbose >= 2:
                logger.log('Using agent with the following configuration:')
                logger.log(str(self.__dict__.items()))

            eval_episode_rewards_history = deque(maxlen=100)
            episode_rewards_history = deque(maxlen=100)
            self.episode_reward = np.zeros((1,))
            self.unscaled_episode_reward = np.zeros((1,))
            episode_successes = []
            with self.sess.as_default(), self.graph.as_default():
                # pretrain agent
                if dataset_addr is not None:
                    self.pretrain(
                        dataset_addr, pretrain_steps=pretrain_steps,
                        max_samples_expert=max_samples_expert)
                    self.save(pretrain_model_name)

                # setup live plot
                if self.live_plot:
                    # create vectors to store plot values
                    n_avg = 5  # number of points to compute the average
                    x_vec = np.zeros(n_avg)
                    y_vec = np.zeros(n_avg)
                    x_cnt = 0

                    # setup figure
                    plt.figure()
                    plt.title('Cycle-of-Learning')
                    plt.xlabel('Time Step')
                    plt.ylabel('Reward')
                    plt.ylim([0,1.05])
                    plt.xlim([0,total_timesteps])
                    plt.pause(0.01)

                # setup parallel training
                self.train_thread = threading.Thread(target=self._parallel_training)
                self.allow_thread = False
                self.train_thread.start()

                # Prepare everything.
                print('Training {} experiment...'.format(self.log_addr))
                self._reset()
                obs = self.env.reset()
                eval_obs = None
                if self.eval_env is not None:
                    eval_obs = self.eval_env.reset()
                episode_reward = 0.
                unscaled_episode_reward = 0.
                episode_step = 0
                episodes = 0
                step = 0
                total_steps = 0

                start_time = time.time()

                self.obs0_n, self.action_n, self.reward_n, self.obs1_n, \
                    self.terminal1_n = [], [], [], [], []
                epoch_episode_rewards = []
                epoch_episode_steps = []
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                eval_episode_rewards = []
                eval_qs = []
                epoch_actions = []
                epoch_qs = []
                epoch_episodes = 0
                epoch = 0
                while True:
                    for _ in range(log_interval):
                        # Perform rollouts.
                        for _ in range(self.nb_rollout_steps):
                            if total_steps >= total_timesteps:
                                return self

                            # Prediction action and qval uncertainty using MC Dropout
                            # (compute multiple samples with dropout rate = 0.1)
                            action_uncert, q_value_uncert = self._actor_mc_dropout(obs, n_samples=20)
                            self.env.envs[0].confidence = [action_uncert]  # update GUI confidence value

                            # Predict next action or use expert action if scheduling
                            action, q_value = self._policy(
                                obs, apply_noise=True, compute_q=True, prob_dropout=0.0)
                            self.env.envs[0].qval_t = np.squeeze(q_value)  # update GUI q-value

                            # check if human is controlling and read its actions
                            human_controlling = self.env.envs[0].human_control

                            # store human actions flipping throttle and yaw so it matches
                            # the data collection script
                            human_actions = self.joystick.act()[0]

                            assert action.shape == self.env.action_space.shape

                            # Execute next action.
                            if rank == 0 and self.render:
                                self.env.render()

                            # Randomly sample actions from a uniform distribution
                            # with a probabilty self.random_exploration (used in HER + DDPG)
                            if np.random.rand() < self.random_exploration:
                                rescaled_action = action = self.action_space.sample()
                            else:
                                rescaled_action = action * np.abs(self.action_space.low)

                            new_obs, reward, done, info = self.env.step(rescaled_action)

                            # scaled rewards following the dqfd paper
                            unscaled_reward = np.copy(reward)
                            reward = self._scale_reward(reward)

                            # if human controlling, add data to expert buffer and trigger parallel updates
                            if human_controlling:
                                
                                # add data to expert buffer
                                obs0 = obs
                                obs1 = new_obs
                                action = human_actions
                                terminal1 = done

                                # add to both buffers
                                self.replay_buffer.add(obs0, action, unscaled_reward, obs1, terminal1)
                                self.replay_buffer_expert.add(obs0, action, unscaled_reward, obs1, terminal1)

                                # trigger parallel training
                                self.allow_thread = True
                            else:
                                # no parallel training (only when human demonstrates/intervenes)
                                self.allow_thread = False

                            if writer is not None:
                                # uses tensorboard
                                unscaled_ep_rew = np.array([unscaled_reward]).reshape((1, -1))
                                ep_rew = np.array([reward]).reshape((1, -1))
                                ep_done = np.array([done]).reshape((1, -1))
                                self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_rew, ep_done,
                                                                                  writer, self.num_timesteps)
                                self.unscaled_episode_reward = tag_episode_reward_logger(self.unscaled_episode_reward, unscaled_ep_rew, ep_done,
                                                          writer, self.num_timesteps, tag_name='unscaled_episode_reward')                                

                            step += 1
                            total_steps += 1
                            self.num_timesteps += 1
                            if rank == 0 and self.render:
                                self.env.render()
                            episode_reward += reward
                            unscaled_episode_reward += unscaled_reward
                            episode_step += 1

                            # Book-keeping.
                            epoch_actions.append(action)
                            epoch_qs.append(q_value)
                            self._store_transition(obs, action, reward, new_obs, done)
                            obs = new_obs
                            if callback is not None:
                                # Only stop training if return value is False, not when it is None.
                                # This is for backwards compatibility with callbacks that have no return statement.
                                if callback(locals(), globals()) is False:
                                    return self

                            if done:
                                # Log episode rewards using CSV files
                                self._write_log(log_mode='reward', step=step,
                                                data=[episode_reward,
                                                      unscaled_episode_reward])

                                # Episode done.
                                print('[{}] Time step {} reward (unscaled): {}'.format(
                                    self.log_addr, total_steps, unscaled_episode_reward))
                                epoch_episode_rewards.append(episode_reward)
                                episode_rewards_history.append(episode_reward)
                                epoch_episode_steps.append(episode_step)

                                # update live plot
                                if self.live_plot:
                                    # append to the end of array to be plotted
                                    y_vec[-1] = episode_reward
                                    x_vec[-1] = total_steps

                                    # plot marker and line showing the average value
                                    if x_cnt == n_avg-1:
                                        # compute the first average value
                                        prev_mean = np.mean(y_vec)
                                    plt.scatter(x_vec[-1], y_vec[-1], color='tab:blue', marker='o', alpha=0.35)
                                    if np.count_nonzero(y_vec) >= n_avg:
                                        # only plot line when have enough samples to compute average
                                        curr_mean = np.mean(y_vec)
                                        plt.plot([x_vec[-2], x_vec[-1]], [prev_mean, curr_mean], color='tab:red', linestyle='--')
                                        prev_mean = curr_mean

                                    # append new data to previous to make a continuous plot
                                    # (move old samples one spot to the left)
                                    y_vec = np.append(y_vec[1:],0.0)
                                    x_vec = np.append(x_vec[1:],0.0)

                                    # draw lines and update counters
                                    plt.savefig(self.log_addr+"/reward.png", dpi=300)
                                    plt.pause(0.001)
                                    x_cnt += 1

                                # reset variables
                                episode_reward = 0.
                                unscaled_episode_reward = 0.
                                episode_step = 0
                                epoch_episodes += 1
                                episodes += 1

                                maybe_is_success = info.get('is_success')
                                if maybe_is_success is not None:
                                    episode_successes.append(float(maybe_is_success))

                                self._reset()
                                if not isinstance(self.env, VecEnv):
                                    obs = self.env.reset()


                        # Train.
                        epoch_actor_losses = []
                        epoch_critic_losses = []
                        epoch_adaptive_distances = []
                        for t_train in range(self.nb_train_steps):
                            # Adapt param noise, if necessary.
                            if self.replay_buffer.__len__() >= self.batch_size and \
                                    t_train % self.param_noise_adaption_interval == 0:
                                distance = self._adapt_param_noise()
                                epoch_adaptive_distances.append(distance)

                            # train step
                            step = total_steps
                            critic_loss, actor_loss = self._train_step(step, writer, log=t_train == 0)

                            # log data using CSV files
                            if (t_train == 0) and (step % self.csv_log_interval == 0):
                                self._write_log(log_mode='loss', step=step,
                                                data=[actor_loss, critic_loss])
                            epoch_critic_losses.append(critic_loss)
                            epoch_actor_losses.append(actor_loss)
                            self._update_target_net()

                        # Evaluate.
                        eval_episode_rewards = []
                        eval_qs = []
                        if self.eval_env is not None:
                            eval_episode_reward = 0.
                            for _ in range(self.nb_eval_steps):
                                if total_steps >= total_timesteps:
                                    return self

                                eval_action, eval_q = self._policy(eval_obs, apply_noise=False, compute_q=True)
                                eval_obs, eval_r, eval_done, _ = self.eval_env.step(eval_action *
                                                                                    np.abs(self.action_space.low))
                                if self.render_eval:
                                    self.eval_env.render()
                                eval_episode_reward += eval_r

                                eval_qs.append(eval_q)
                                if eval_done:
                                    if not isinstance(self.env, VecEnv):
                                        eval_obs = self.eval_env.reset()
                                    eval_episode_rewards.append(eval_episode_reward)
                                    eval_episode_rewards_history.append(eval_episode_reward)
                                    eval_episode_reward = 0.

                    mpi_size = MPI.COMM_WORLD.Get_size()
                    # Log stats.
                    # XXX shouldn't call np.mean on variable length lists
                    duration = time.time() - start_time
                    stats = self._get_stats()
                    combined_stats = stats.copy()
                    combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
                    combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
                    combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
                    combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
                    combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
                    combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
                    combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
                    if len(epoch_adaptive_distances) != 0:
                        combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
                    combined_stats['total/duration'] = duration
                    combined_stats['total/steps_per_second'] = float(step) / float(duration)
                    combined_stats['total/episodes'] = episodes
                    combined_stats['rollout/episodes'] = epoch_episodes
                    combined_stats['rollout/actions_std'] = np.std(epoch_actions)
                    # Evaluation statistics.
                    if self.eval_env is not None:
                        combined_stats['eval/return'] = np.mean(eval_episode_rewards)
                        combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                        combined_stats['eval/Q'] = np.mean(eval_qs)
                        combined_stats['eval/episodes'] = len(eval_episode_rewards)

                    def as_scalar(scalar):
                        """
                        check and return the input if it is a scalar, otherwise raise ValueError

                        :param scalar: (Any) the object to check
                        :return: (Number) the scalar if x is a scalar
                        """
                        if isinstance(scalar, np.ndarray):
                            assert scalar.size == 1
                            return scalar[0]
                        elif np.isscalar(scalar):
                            return scalar
                        else:
                            raise ValueError('expected scalar, got %s' % scalar)

                    combined_stats_sums = MPI.COMM_WORLD.allreduce(
                        np.array([as_scalar(x) for x in combined_stats.values()]))
                    combined_stats = {k: v / mpi_size for (k, v) in zip(combined_stats.keys(), combined_stats_sums)}

                    # Total statistics.
                    combined_stats['total/epochs'] = epoch + 1
                    combined_stats['total/steps'] = step

                    for key in sorted(combined_stats.keys()):
                        logger.record_tabular(key, combined_stats[key])
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    logger.dump_tabular()
                    logger.info('')
                    logdir = logger.get_dir()
                    if rank == 0 and logdir:
                        if hasattr(self.env, 'get_state'):
                            with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as file_handler:
                                pickle.dump(self.env.get_state(), file_handler)
                        if self.eval_env and hasattr(self.eval_env, 'get_state'):
                            with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as file_handler:
                                pickle.dump(self.eval_env.get_state(), file_handler)

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions, _, = self._policy(observation, apply_noise=not deterministic, compute_q=False)
        actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        actions = actions * np.abs(self.action_space.low)  # scale the output for the prediction

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        _ = np.array(observation)

        if actions is not None:
            raise ValueError("Error: DDPG does not have action probabilities.")

        # here there are no action probabilities, as DDPG does not use a probability distribution
        warnings.warn("Warning: action probability is meaningless for DDPG. Returning None")
        return None

    def get_parameter_list(self):
        return (self.params +
                self.target_params +
                self.obs_rms_params +
                self.ret_rms_params)

    def save(self, save_path):
        data = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "nb_eval_steps": self.nb_eval_steps,
            "param_noise_adaption_interval": self.param_noise_adaption_interval,
            "nb_train_steps": self.nb_train_steps,
            "nb_rollout_steps": self.nb_rollout_steps,
            "verbose": self.verbose,
            "param_noise": self.param_noise,
            "action_noise": self.action_noise,
            "gamma": self.gamma,
            "tau": self.tau,
            "normalize_returns": self.normalize_returns,
            "enable_popart": self.enable_popart,
            "normalize_observations": self.normalize_observations,
            "batch_size": self.batch_size,
            "observation_range": self.observation_range,
            "return_range": self.return_range,
            "critic_l2_reg": self.critic_l2_reg,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "clip_norm": self.clip_norm,
            "reward_scale": self.reward_scale,
            "memory_limit": self.memory_limit,
            "buffer_size": self.buffer_size,
            "random_exploration": self.random_exploration,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
            "prioritized_replay": self.prioritized_replay,
            "prioritized_replay_eps": self.prioritized_replay_eps,
            "prioritized_replay_alpha": self.prioritized_replay_alpha,
            "prioritized_replay_beta0": self.prioritized_replay_beta0,
            "prioritized_replay_beta_iters": self.prioritized_replay_beta_iters
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path,
                            data=data,
                            params=params_to_save)

        # also save memory buffer (expert and agent)
        outfile_memory_expert = open(self.log_addr + '/memory_expert.pkl', 'wb')
        outfile_memory = open(self.log_addr + '/memory.pkl', 'wb')
        pickle.dump(self.replay_buffer_expert, outfile_memory_expert)
        pickle.dump(self.replay_buffer, outfile_memory)
        outfile_memory_expert.close()
        outfile_memory.close()


    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        data, params = cls._load_from_file(load_path)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                                "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                                kwargs['policy_kwargs']))

        model = cls(None, env, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model(loading_model=True)
        # Patch for version < v2.6.0, duplicated keys where saved
        if len(params) > len(model.get_parameter_list()):
            n_params = len(model.params)
            n_target_params = len(model.target_params)
            n_normalisation_params = len(model.obs_rms_params) + len(model.ret_rms_params)
            # Check that the issue is the one from
            # https://github.com/hill-a/stable-baselines/issues/363
            assert len(params) == 2 * (n_params + n_target_params) + n_normalisation_params,\
                "The number of parameter saved differs from the number of parameters"\
                " that should be loaded: {}!={}".format(len(params), len(model.get_parameter_list()))
            # Remove duplicates
            params_ = params[:n_params + n_target_params]
            if n_normalisation_params > 0:
                params_ += params[-n_normalisation_params:]
            params = params_
        model.load_parameters(params)

        # # also load memory buffer (expert and agent)
        # infile_memory_expert = open(model.log_addr + '/memory_expert.pkl', 'rb')
        # infile_memory = open(model.log_addr + '/memory.pkl', 'rb')
        # pickle.load(model.memory_expert, infile_memory_expert)
        # pickle.load(model.memory, infile_memory)
        # infile_memory_expert.close()
        # infile_memory.close()

        return model
            
