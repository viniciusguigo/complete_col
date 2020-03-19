#!/usr/bin/env python
""" hri_airsim_env.py:
Defines AirSim environment following OpenAI template.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "March 3, 2018"

# import
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import tensorflow as tf
from pathlib import Path
import time, math, datetime
import os
import configparser
import sys
import numpy as np
import cv2
import signal
from contextlib import contextmanager
import imutils

# handles timeouts
class TimeoutException(Exception): pass

# @contextmanager
# def time_limit(seconds):
#     def signal_handler(signum, frame):
#         raise TimeoutException("Timed out!")
#     signal.signal(signal.SIGALRM, signal_handler)
#     signal.alarm(seconds)
#     try:
#         yield
#     finally:
#         signal.alarm(0)

import airsim

DEBUG_STATEMENTS = False


def debug(*args, **kwargs):
    """Basic debug print function."""
    force = kwargs['force'] if 'force' in kwargs else False
    if DEBUG_STATEMENTS or force:
        print('DEBUG: {:s}'.format(*args))


def print_both(msg, client):
    """ Print both on terminal and AirSim.
    """
    print(msg)
    client.simPrintLogMessage(msg)


class HRI_AirSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # read config file
        config_file = './config/config_main.ini'
        config = configparser.ConfigParser()
        config.read(config_file)

        main_setup = config['DEFAULT']
        uas_setup = config['UAS']
        map_setup = config['MAP']

        # create folder to save data based on current date and time
        self.env_time = time.strftime("%Y_%m_%d_%H_%M_%S")
        self.name_folder = './data/' + self.env_time
        print('[*] Saving images and logs at ', self.name_folder)
        os.makedirs(Path(self.name_folder), exist_ok=True)
        os.makedirs(Path(self.name_folder + '/depth'), exist_ok=True) # for depth data

        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.episode = -1  # current episode number
        # counter increases when 'reset' is called
        self.t = 0  # increases with the number of steps
        self.epi_t = 0 # counts episode timesteps
        self.initial_human_data_size = 0
        # also works as ID for images and numerical data

        # pos, vel, acc (x,y,z: linear and ang)
        # check self._parse_imu() for labels
        self.imu = np.zeros(18)
        self.vehicle_has_collided = False
        self.imu_timestamp = 0
        self.C = np.zeros((2,2))

        # initialize action vector
        self.act_x = 0.
        self.act_y = 0.
        self.act_z = 0.
        self.act_w = 0.
        self.actions = np.array([self.act_x, self.act_y, self.act_z, self.act_w])

        # parameters
        self.n_episodes = main_setup.getint('n_episodes')
        self.n_max_steps = main_setup.getint('n_max_steps')
        self.n_hold_action = main_setup.getint('n_hold_action')

        self.n_act_mode = main_setup.getint('n_act_mode')
        if self.n_act_mode == 5 or self.n_act_mode == 3:
            self.n_act = 3
        elif self.n_act_mode == 4 or self.n_act_mode == 2 or self.n_act_mode == 6:
            self.n_act = 2
        elif self.n_act_mode == 1 or self.n_act_mode == 7:
            self.n_act = 1
        elif self.n_act_mode == 8:
            self.n_act = 4

        self.action_level = main_setup.getint('action_level')
        self.mission = main_setup['mission']
        self.use_gps = main_setup.getboolean('use_gps')
        self.reward_function = main_setup['reward_function']
        self.prev_shaping = 0
        self.use_perception = main_setup.getboolean('use_perception')

        # wind settings
        self.use_wind = main_setup.getboolean('use_wind')
        self.have_wind = False
        self.Vm = 0.25 # max wind amplitude
        self.wind_direction = np.zeros(self.n_act)
        self.wind_prob = 0.45
        self.wind_steps = 0
        self.wind_curr_step = 0
        self.min_wind_steps = 5
        self.max_wind_steps = 15

        self.feature_level = main_setup['feature_level']

        self.map_max_x = map_setup.getfloat('max_x')
        self.map_min_x = map_setup.getfloat('min_x')

        self.alt = uas_setup.getfloat('alt')
        self.initial_alt = self.alt
        self.const_act_x = uas_setup.getfloat('const_act_x')
        self.const_act_z = uas_setup.getfloat('const_act_z')

        self.initial_human_control = uas_setup.getboolean(
            'initial_human_control')  # human or agent starts controlling

        # continuous actions
        self.action_ub = 1  # upper bound
        self.action_lb = -1  # lower bound
        self.action_space = spaces.Box(self.action_lb, self.action_ub,
                                       shape=(self.n_act,))

        # observations: frontal rgb camera
        # set camera parameters in AirSim's settings
        # file (~/Documents/AirSim/settings.json)
        self.flat_images = uas_setup.getboolean('flat_images')
        self.camera_mode = uas_setup['camera_mode']
        self.save_training_image_data = uas_setup.getboolean('save_training_image_data')

        self.scale_input = uas_setup.getboolean('scale_input')
        self.sf = uas_setup.getfloat('scale_factor')

        # correct observation space if desired to scale images
        if self.scale_input:
            print('[*] Working with scaled inputs.')
            screen_width = int(self.sf*uas_setup.getint('screen_width'))
            screen_height = int(self.sf*uas_setup.getint('screen_height'))
        else:
            screen_width = uas_setup.getint('screen_width')
            screen_height = uas_setup.getint('screen_height')

        if self.camera_mode == 'rgba':
            channels = 4
        elif self.camera_mode == 'rgb':
            channels = 3
        elif self.camera_mode == 'grayscale':
            channels = 1
        elif self.camera_mode == 'depth':
            channels = 1
        else:
            sys.exit('Invalid camera mode. \
                Check config file (default: config_main.ini )')

        # change observation space according to high or low (config_main file)
        if self.feature_level == 'high':
            self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(screen_height, screen_width, channels))
        elif self.feature_level == 'low':
            if self.mission == 'colission_avoidance':
                self.sess_vae = tf.Session()

                # vae address and dimensions
                self.VAE_INPUT_DIM = int(36*64)
                N_FEATURES_LOW = 36
                VAE_ADDR = '../data/vae_model'
                new_saver = tf.train.import_meta_graph(VAE_ADDR + '/model.ckpt.meta')
                new_saver.restore(self.sess_vae, tf.train.latest_checkpoint(VAE_ADDR))

                graph = tf.get_default_graph()
                # input: depth images
                self.vae_input_image = graph.get_tensor_by_name("input_image:0")
                self.vae_input_encoded = graph.get_tensor_by_name("code:0")

            elif self.mission == 'landing':
                N_FEATURES_LOW = 3 # x, y, and radius of pad

            else:
                sys.exit('Invalid mission mode. \
                            Check config file (default: config_main.ini )')

            # imu parameters
            if self.use_gps:
                N_FEATURES_IMU = 12
            else:
                N_FEATURES_IMU = 10

            # create low-dimensional observation space
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(N_FEATURES_IMU + N_FEATURES_LOW,))

            # create log file to save low level features
            self.low_obs_log_file = open(
                '{}/low_obs_log.csv'.format(self.name_folder), 'w')

            # PyQT GUI variables
            self.use_pyqt = main_setup.getboolean('use_pyqt')
            self.confidence = [np.array(0)]
            self.control = 'reset'
            self.display_img = np.zeros(shape=(640,480))
            self.ts = []
            self.qval_t = np.array(0)
            self.qval_lb = np.array(0)
            self.qval_ub = np.array(0)
            self.qvals = []
            self.qvals_lb = []
            self.qvals_ub = []

        else:
            print('ERROR: Please check feature_level at config/config_main.ini')
            sys.exit(0)

        # create file to save logs
        self.log_file = open('{}/log.csv'.format(self.name_folder), 'w')
        self.human_log_file = open('{}/human_log.csv'.format(self.name_folder), 'w')
        self.reward_log_file = open('{}/reward_log.csv'.format(self.name_folder), 'w')
        self._label_log()
        self.i = 0

        # display useful info
        self._print_env_info()

        # get initial tank position
        self.initial_moving_vehicle_pose = self.client.simGetObjectPose('vehicle0')

    def _preload_csv(self,initial_human_addr, initial_human_epi):
        print('[*] Preloading data from ', initial_human_addr)
        # load complete csv file with previous human data
        previous_human_data = np.genfromtxt(initial_human_addr + '/human_log.csv', delimiter=',')

        # find the time step index for the desired episode
        # the -1 corrects the episode index (log file starts on 0)
        previous_log_data = np.genfromtxt(initial_human_addr + '/log.csv', delimiter=',')
        epi_idx = np.where(previous_log_data[:,1] == (initial_human_epi-1))
        last_epi_idx = epi_idx[0][-1]
        timestep_idx = previous_log_data[last_epi_idx,0]

        # copy everything up to this time step index
        human_epi_idx = np.where(previous_human_data[:,0] <= timestep_idx)
        human_epi_idx = human_epi_idx[0][-1]

        # grab human data up to the desired timestep and feed to current human_log.csv
        initial_human_data = previous_human_data[:human_epi_idx,:]
        self.initial_human_data_size = initial_human_data.shape[0]
        print('[*] Using {} samples.'.format(self.initial_human_data_size))

        for i in range(self.initial_human_data_size):
            # grab row of data
            obs_row = initial_human_data[i,:]
            self.t = int(obs_row[0])
            self.reward = obs_row[1]
            self.act_y = obs_row[2]
            self.act_x = obs_row[3]
            self.act_z = obs_row[4]
            self.act_w = obs_row[5]
            obs = obs_row[6:]

            # write to csv
            self._log_low_obs(obs, log_human=True)


    def _print_env_info(self):
        print('\n=========================================')
        print('==     HRI_AirSim ENVIRONMENT INFO     ==')
        print('=========================================\n')
        print('[*] Mission: ', self.mission)
        print('[*] Feature level: ', self.feature_level)
        print('[*] Observation space: ', self.observation_space)
        print('[*] Action space: ', self.action_space)
        print('[*] Action level: ', self.action_level)
        print('[*] Camera mode: ', self.camera_mode)
        print('[*] Human begin controlling: ', self.initial_human_control)
        print('[*] Saving image data: ', self.save_training_image_data)
        print('\n[*] Episodes: ', self.n_episodes)
        print('[*] Maximum number of steps: ', self.n_max_steps)
        print('\n=========================================\n')

    def _low_lvl_cmd(self):
        """Move vehicle by sending low-level angle commands (roll, pitch, yaw.)
        """
        # send the action (converting to correct format)
        self.act_x = np.float64(self.act_x)
        self.act_y = np.float64(self.act_y)
        sc = 2
        yaw = self.imu[5] + self.act_w/(2*sc)
        self.alt += self.act_z/(2*sc)
        self.client.moveByAngle(-self.act_x, self.act_y, self.alt, yaw, 1)

    def _high_lvl_cmd(self):
        """Move vehicle by sending high-level velocity commands (vx, vy, vz).
        """
        # send the action (converting to correct format: np.float64)
        duration = 0.01
        vx = -4*self.act_x
        vy = 4*self.act_y
        vz = 40*self.act_z

        # translate from inertial to body frame
        th = self.imu[5]
        self.C[0,0] = np.cos(th)
        self.C[0,1] = -np.sin(th)
        self.C[1,0] = -self.C[0,1]
        self.C[1,1] = self.C[0,0]
        vb = self.C.dot(np.array([vx,vy]))

        # send commands
        self.client.moveByVelocityZAsync(vb[0], vb[1], self.alt, duration,
                                airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, vz)).join()

    def _higher_lvl_cmd(self):
        """Move vehicle by sending high-level position commands (pos_x, etc).
        """
        # changes in command
        # self.imu[0] = pos_x
        # self.imu[1] = pos_y
        x = self.imu[0] - 4*self.act_x
        y = self.imu[1] - 4*self.act_y
        print(x, y)
        velocity = 5 # m/s

        self.client.moveToPositionAsync(x, y, self.alt, velocity, max_wait_seconds = .1,
        drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode = airsim.YawMode(),
        lookahead = -1, adaptive_lookahead = 1).join()

    def _capture_obs(self):
        """Capture observation (depth, image input, etc)
        """
        # get data
        image = self._take_pic()
        depth = self._read_depth()

        # vehicle imu
        self._parse_imu()

        if self.feature_level == 'high': # image data
            if self.camera_mode == 'depth':
                obs = depth
            else:
                obs = image

            # scale if necessary
            if self.scale_input:
                obs = cv2.resize(obs, None, fx=self.sf, fy=self.sf)

            # return obs as flat arrays, instead of 2D image format
            if self.flat_images:
                obs = obs.flatten()

        elif self.feature_level == 'low': # encoded data
            # scale depth image for vae
            depth = cv2.resize(depth, None, fx=0.25, fy=0.25)

            # pass depth through VAE and extract low level features
            feed_dict = {self.vae_input_image: depth.reshape((1,self.VAE_INPUT_DIM))}
            code = self.sess_vae.run(self.vae_input_encoded, feed_dict=feed_dict)

            # concatenate it to imu values
            obs = np.hstack((self.imu,code[0]))

            # log low level observation
            self._log_low_obs(obs, log_obs=True)

        return obs

    def step(self, action):
        # check action format and clip to maximum allowed
        if DEBUG_STATEMENTS:
            debug('action received: {} | shape: {} | \
                class: {} | action[0].__class__: {}'.format(action,
                                                            action.shape, action.__class__, action[0].__class__))

        # clip receive angles
        action = np.clip(action, self.action_lb, self.action_ub)

        # read joystick
        rcdata = airsim.RCData()
        rcdata = self._apply_deadband(rcdata)
        self.total_steps += 1

        # switch between human and agent by holding top left trigger
        if self.rcdata_switch5:
            self.human_control = True
        else:
            self.human_control = False

        # collect continuous reward function from human when
        # the right trigger is held
        if self.rcdata_switch6:
            self.human_reward = True
        else:
            self.human_reward = False

        # process ACTIONS
        # check if human or agent controlling
        if self.human_control:
            # HUMAN CONTROLLING

            if self.n_act_mode == 4:
                # send roll to rotate vehicle, instead of turning
                self.act_z = self.rcdata_yaw
                self.act_y = 0
                # centers joystick between [-1,1]
                self.act_x = -(self.rcdata_throttle-0.75)*2
                self.act_w = 0

            elif self.n_act_mode == 3:
                self.act_z = self.rcdata_roll
                self.act_y = self.rcdata_yaw
                # centers joystick between [-1,1]
                self.act_x = -(self.rcdata_throttle-0.75)*2
                self.act_w = 0

            elif self.n_act_mode == 2:
                self.act_z = 0
                self.act_y = self.rcdata_yaw
                # centers joystick between [-1,1]
                self.act_x = -(self.rcdata_throttle-0.75)*2
                self.act_w = 0

            elif self.n_act_mode == 1:
                self.act_z = 0
                self.act_y = self.rcdata_yaw
                self.act_x = self.const_act_x
                self.act_w = 0

        else:
            # AGENT CONTROLLING

            # act (holding action based on counter)
            if self.count_action == self.n_hold_action:
                self.count_action = 1

                # check how many actions agent is controlling and act
                if self.n_act_mode == 4:
                    # send roll to rotate vehicle, instead of turning
                    self.act_z = action[0]
                    self.act_y = 0
                    self.act_x = action[1]
                    self.act_w = 0

                elif self.n_act_mode == 3:
                    self.act_z = action[2]
                    self.act_y = action[0]
                    self.act_x = action[1]
                    self.act_w = 0

                elif self.n_act_mode == 2:
                    self.act_z = 0
                    self.act_y = action[0]
                    self.act_x = action[1]
                    self.act_w = 0

                elif self.n_act_mode == 1:
                    self.act_z = 0
                    self.act_y = action[0]
                    self.act_x = self.const_act_x
                    self.act_w = 0
            else:
                self.count_action += 1

        # decide level or action control
        self.actions = np.array([self.act_x, self.act_y, self.act_z, self.act_w])
        if self.action_level == 0: #'low':
            self._low_lvl_cmd()
        elif self.action_level == 1: #'high':
            self._high_lvl_cmd()
        elif self.action_level == 2: #'higher':
            self._higher_lvl_cmd()
        else:
            sys.exit('Invalid action_level. Check config file.')


        # NEXT STATES:
        # next states (camera data) based on current camera mode
        # (no matter the mode, we always save image and depth data)
        obs = self._capture_obs()
        pos_x = self.imu[0]


        # REWARD:
        if self.reward_source == 'custom':
            reward = self._custom_reward(pos_x)

        else:
            reward = 0

        if DEBUG_STATEMENTS:
            debug('Reward: %.2f' % (reward))


        # DONE:
        # 'select' button (switch7), completed track, or collision
        done = self.rcdata_switch7
        if ((pos_x < self.map_min_x) or
                (self.vehicle_has_collided == True)):
            #print('\n[*] Crash!')
            done = 1

        if (pos_x > self.map_max_x):
            print('\n[*] Completed course!')
            done = 1

        # additional info
        info = {}

        # update log
        self._log_data(reward, done)
        if self.feature_level == 'low':
            self._log_low_obs(obs, log_obs=True)

            # log human intervention data
            if self.human_control:
                self._log_low_obs(obs, log_human=True)

        # increase data counter
        self.t += 1
        self.epi_t += 1

        return obs, reward, done, info

    def _custom_reward(self, pos_x):
        """Custom reward function.
        """
        # find current distance traveled (on x direction)
        return 1#pos_x/self.map_max_x

    def _joystick_reward(self, rcdata):
        """Compute reward based on joystick buttons pressed.
        """
        # continuous reward from left stick
        continuous_rew = self.rcdata_pitch

        #### VGG July 30, 2018
        ### NEED TO FIND BUTTONS FOR IT
        ### INTERVENTION TOOK TOP LEFT TRIGGER
        # binary reward from top buttons
        if self.rcdata_switch5:
            binary_rew = -1
        elif self.rcdata_switch6:
            binary_rew = 1
        else:
            binary_rew = 0

        # sum them
        reward = continuous_rew + binary_rew

        return reward

    def _query_reward_network(self, obs):
        """Query a reward cloning network to estimate reward.
        """
        # reshape to fit network (needs batch dimension)
        obs = obs.reshape(np.hstack((1,self.observation_space.shape)))

        # query network
        reward = self.reward_network.predict(obs)

        return reward[0][0]

    # make sure it returns on a given amount of time
    def _full_takeoff(self):
        # re-initialize airsim
        if not self.client.isApiControlEnabled():
            self.client.enableApiControl(True)
        self.client.armDisarm(True)

        print("[*] Taking off...")

        # # AirSim < v1.0
        # return self.client.takeoffAsync().join()

        return True

    def reset(self, initial_x = None, initial_y = None):
        currentDT = datetime.datetime.now()
        print('[*] RESET TIME: {}'.format(currentDT.strftime("%Y-%m-%d %H:%M:%S")))
        # force reset
        reset_ok = False

        while True:
            try:
                # wait for a given amount of seconds to reset
                # if takes too long, retry
                #with time_limit(5): # seconds
                if not self.client.isApiControlEnabled():
                    self.client.enableApiControl(True)
                self.client.armDisarm(True)
                self.client.reset()
                reset_ok = True

                if reset_ok:
                    break

            except: #TimeoutException as e:
                print("[*] Reset failed")
                # disarm everything
                self.client.armDisarm(False)
                self.client.enableApiControl(False)

        # reset episode parameters
        self.alt = self.initial_alt
        self.episode += 1
        self.epi_t = 0
        self.total_steps = 0
        self.count_action = 0
        self.human_control = False
        self.reward = 0
        self.count_action = self.n_hold_action

        # reset GUI
        if self.use_pyqt:
            self.ts = []
            self.qvals = []
            self.qvals_lb = []
            self.qvals_ub = []
            self.qval_t = np.array(0)
            self.qval_lb = np.array(0)
            self.qval_ub = np.array(0)
            self.confidence = [np.array(0)]
            self.control = 'reset'
            self.display_img = np.zeros(shape=(640,480))

        # try to handle error during takeoff
        takeoff_ok = False

        while True:
            try:
                # wait for a given amount of seconds while takeoff
                # if takes too long, retry
                #with time_limit(15): # seconds
                takeoff_ok = self._full_takeoff()

                if takeoff_ok:
                    break
            except:
                print("[*] Takeoff failed")
                self.client.armDisarm(False)
                self.client.enableApiControl(False)

        # # change initial position
        # CURRENTLY FREEZES AIRSIM
        # self.client.simSetPose(Pose(Vector3r(0, 6, -3),
        #                       self.client.toQuaternion(0, 0, 0)), True)

        # RANDOM INITIAL POSITION (LANDING)
        if self.mission == 'landing':
            print('[*] Moving to random initial position...')
            # if initial_x == None:
            #     initial_x = 1.0*(2*np.random.rand()-1)
            # if initial_y == None:
            #     initial_y = 2.0*(2*np.random.rand()-1)

            # random initial location
            initial_x = 1.0*(2*np.random.rand()-1)
            initial_y = 2.0*(2*np.random.rand()-1)
    
            print('[*] X: {:6.4f} | Y: {:6.4f}'.format(initial_x,initial_y))
            self.client.moveToPositionAsync(
                x = initial_x,
                y = initial_y,
                z = self.initial_alt,
                velocity = 5).join()
            time.sleep(3)
            print('[*] At initial position.')

        # next states (camera data) based on current camera mode
        # (no matter the mode, we always save image and depth data)
        obs = self._capture_obs()

        # reset tank (landing pad) to initial position
        self.client.simSetObjectPose(
            object_name='vehicle0',
            pose=self.initial_moving_vehicle_pose,
            teleport = True)

        if DEBUG_STATEMENTS:
            debug('obs.shape: {}'.format(obs.shape))

        return obs

    def render(self, mode='human', close=False):
        # raise NotImplementedError()
        pass

    def close(self):
        self.client.reset()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        self.log_file.close()

        if self.feature_level == 'low':
            self.low_obs_log_file.close()
            self.human_log_file.close()
            self.reward_log_file.close()

    def _take_pic(self):
        """ Take picture and convert to Numpy array
        """
        if self.camera_mode == 'rgb':
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
            response = responses[0]

            # get numpy array
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)

            # reshape array to 4 channel image array H X W X 4
            img_rgba = img1d.reshape(response.height, response.width, 4)

            # # original image is fliped vertically
            # img_rgba = np.flipud(img_rgba)

            # convert to rgb
            img = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2RGB)
            if self.save_training_image_data:
                cv2.imwrite('{}/{}.png'.format(self.name_folder, self.t), img)

        elif self.camera_mode == 'grayscale':
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene,
                             pixels_as_float=True,
                             compress=False)])
            response = responses[0]

            # get numpy array
            img1d = np.array(response.image_data_float, dtype=np.float32)

            # reshape array to 1 channel image array H X W X 1
            img_rgba = img1d.reshape(response.height, response.width)
            img = np.array(img_rgba * 255, dtype=np.uint8)

            # # original image is fliped vertically
            # img_rgba = np.flipud(img_rgba)
            if self.save_training_image_data:
                cv2.imwrite('{}/{}.png'.format(self.name_folder, self.t), img)

        # if rgba or depth mode -> grab RGBA data
        else:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
            response = responses[0]

            # get numpy array
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)

            # reshape array to 4 channel image array H X W X 4
            img_rgba = img1d.reshape(response.height, response.width, 4)

            # original image is fliped vertically
            img = np.flipud(img_rgba)

            # write data of this time step to file
            # write image to a png file
            if self.save_training_image_data:
                airsim.write_png(os.path.normpath('{}/{}.png'.format(
                    self.name_folder, self.t)), img)

        return img

    def _toEulerianAngle(self, q):
        z = q.z_val
        y = q.y_val
        x = q.x_val
        w = q.w_val
        ysqr = y * y

        # roll (x-axis rotation)
        t0 = +2.0 * (w*x + y*z)
        t1 = +1.0 - 2.0*(x*x + ysqr)
        roll = math.atan2(t0, t1)

        # pitch (y-axis rotation)
        t2 = +2.0 * (w*y - z*x)
        if (t2 > 1.0):
            t2 = 1
        if (t2 < -1.0):
            t2 = -1.0
        pitch = math.asin(t2)

        # yaw (z-axis rotation)
        t3 = +2.0 * (w*z + x*y)
        t4 = +1.0 - 2.0 * (ysqr + z*z)
        yaw = math.atan2(t3, t4)

        return (pitch, roll, yaw)

    def _parse_imu(self):
        """ Parse vehicle states (IMU).

        self.imu = [
        pos_x, pos_y, pos_z,
        pitch, roll, yaw,
        vel_x, vel_y, vel_z,
        pitch_vel, roll_vel, yaw_vel,
        acc_x, acc_y, acc_z,
        pitch_acc, roll_vel, yaw_acc
        ]
        """
        # get vehicle data
        vehicle_states = self.client.getMultirotorState()
        self.vehicle_has_collided = self.client.simGetCollisionInfo().has_collided
        self.imu_timestamp = vehicle_states.timestamp

        # parse joystick inputs
        # [-1]: A
        # [-2]: B
        # [-3]: X
        # [-4]: Y
        # [-5]: top left
        # [-6]: top right
        # [-7]: left tiny button
        # [-8]: right tiny button
        rcdata_switches = str(np.binary_repr(vehicle_states.rc_data.switches, width=16))
        self.rcdata_switch5 = float(rcdata_switches[-5])
        self.rcdata_switch6 = float(rcdata_switches[-6])
        self.rcdata_switch7 = float(rcdata_switches[-7])
        self.rcdata_switch8 = float(rcdata_switches[-8])

        self.rcdata_pitch = float(vehicle_states.rc_data.pitch)
        self.rcdata_roll = float(vehicle_states.rc_data.roll)
        self.rcdata_yaw = float(vehicle_states.rc_data.yaw)
        self.rcdata_throttle = float(vehicle_states.rc_data.throttle)


        # convert from quaternion to euler angles
        (pitch, roll, yaw) = self._toEulerianAngle(vehicle_states.kinematics_estimated.orientation)

        # parse position level
        self.imu[0] = vehicle_states.kinematics_estimated.position.x_val
        self.imu[1] = vehicle_states.kinematics_estimated.position.y_val
        self.imu[2] = vehicle_states.kinematics_estimated.position.z_val
        self.imu[3] = pitch
        self.imu[4] = roll
        self.imu[5] = yaw

        # parse velocity level
        self.imu[6] = vehicle_states.kinematics_estimated.linear_velocity.x_val
        self.imu[7] = vehicle_states.kinematics_estimated.linear_velocity.y_val
        self.imu[8] = vehicle_states.kinematics_estimated.linear_velocity.z_val
        self.imu[9] = vehicle_states.kinematics_estimated.angular_velocity.x_val
        self.imu[10] = vehicle_states.kinematics_estimated.angular_velocity.y_val
        self.imu[11] = vehicle_states.kinematics_estimated.angular_velocity.z_val

        # parse acceleration level
        self.imu[12] = vehicle_states.kinematics_estimated.linear_acceleration.x_val
        self.imu[13] = vehicle_states.kinematics_estimated.linear_acceleration.y_val
        self.imu[14] = vehicle_states.kinematics_estimated.linear_acceleration.z_val
        self.imu[15] = vehicle_states.kinematics_estimated.angular_acceleration.x_val
        self.imu[16] = vehicle_states.kinematics_estimated.angular_acceleration.y_val
        self.imu[17] = vehicle_states.kinematics_estimated.angular_acceleration.z_val

    def _label_log(self):
        """ Defines log labels
        """
        # define labels
        log_labels = '{},{},{},{},{},{},{},{},{}, \
                     {},{},{},{},{},{},{},{},{},{}, \
                     {},{},{},{},{},{},{},{},{},{}\n'.format('id',
                                                          'episode',
                                                          'action_level',
                                                          'action_y',
                                                          'action_x',
                                                          'action_z',
                                                          'action_w',
                                                          'reward',
                                                          'done',
                                                          'timestamp',
                                                          'collision_status',
                                                          'vehicle_pos_x',
                                                          'vehicle_pos_y',
                                                          'vehicle_pos_z',
                                                          'pitch',
                                                          'roll',
                                                          'yaw',
                                                          'vehicle_vel_x',
                                                          'vehicle_vel_y',
                                                          'vehicle_vel_z',
                                                          'pitch_vel',
                                                          'roll_vel',
                                                          'yaw_vel',
                                                          'vehicle_acc_x',
                                                          'vehicle_acc_y',
                                                          'vehicle_acc_z',
                                                          'pitch_acc',
                                                          'roll_acc',
                                                          'yaw_acc')

        # write to files
        self.log_file.write(log_labels)

    def _log_data(self, reward, done):
        """ Log learning and vehicle data.
        """
        # learning data
        learning_data = '{},{},{},{},{},{},{},{},{}'.format(self.t,
                                                   self.episode,
                                                   self.action_level,
                                                   self.act_y,
                                                   self.act_x,
                                                   self.act_z,
                                                   self.act_w,
                                                   reward,
                                                   done)

        # vehicle data
        if self.vehicle_has_collided:
            collision_status = 1
        else:
            collision_status = 0

        vehicle_pos = '{},{},{}'.format(self.imu[0], self.imu[1], self.imu[2])
        vehicle_ori = '{},{},{}'.format(self.imu[3], self.imu[4], self.imu[5])
        vehicle_vel = '{},{},{}'.format(self.imu[6], self.imu[7], self.imu[8])
        vehicle_ang_vel = '{},{},{}'.format(self.imu[9], self.imu[10], self.imu[11])
        vehicle_acc = '{},{},{}'.format(self.imu[12], self.imu[13], self.imu[14])
        vehicle_ang_acc = '{},{},{}'.format(self.imu[15], self.imu[16], self.imu[17])

        # write log
        self.log_file.write('{},{},{},{},{},{},{},{},{}\n'.format(learning_data,
                                                                  self.imu_timestamp,
                                                                  collision_status,
                                                                  vehicle_pos,
                                                                  vehicle_ori,
                                                                  vehicle_vel,
                                                                  vehicle_ang_vel,
                                                                  vehicle_acc,
                                                                  vehicle_ang_acc))


    def _read_depth(self, n_depth=1):
        """ Read depth sensor all around vehicle.

        * Args *
        n_depth: number of depth cameras to access
                 1. only front cameras
                 2. front + rear cameras
                 4. front + rear + lateral cameras
        """
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)])

        if n_depth == 1:  # only front cameras
            response = responses[0]

            # convert to grayscale
            depth = np.array(response.image_data_float, dtype=np.float32)
            depth = depth.reshape(response.height, response.width)
            depth = np.array(depth * 255, dtype=np.uint8)

            # save pic
            if self.save_training_image_data:
                cv2.imwrite('{}/depth/{}.png'.format(self.name_folder, self.t), depth)

        elif n_depth == 2:  # front + rear cameras
            pass
            # response = responses[0]
            # # get numpy array
            # img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)

            # # reshape array to 4 channel image array H X W X 4
            # depth = img1d.reshape(response.height, response.width, 4)

            # # original image is fliped vertically
            # depth = np.flipud(depth)

            # # save pic
            # if self.save_training_image_data:
            #     cv2.imwrite('{}/depth/{}.png'.format(self.name_folder, self.t), depth)

        return depth

    def _apply_deadband(self, rcdata):
        """ Create a deadband to smooth the data coming from the joysticks
        """
        dbv = 0.01 # deadband value
        if self.rcdata_yaw < dbv and self.rcdata_yaw > -dbv:
            self.rcdata_yaw = 0.0

        if self.rcdata_roll < dbv and self.rcdata_roll > -dbv:
            self.rcdata_roll = 0.0

        if self.rcdata_pitch < dbv and self.rcdata_pitch > -dbv:
            self.rcdata_pitch = 0.0

        if self.rcdata_throttle < (dbv+0.75) and self.rcdata_throttle > (0.75-dbv):
            self.rcdata_throttle = 0.75


        return rcdata

    def _log_low_obs(
        self, low_obs, log_obs=False, log_human=False, log_reward=False):
        """ Log low level observations """
        if log_obs:
            self.low_obs_log_file.write(
                '{},{}\n'.format(self.t,
                                 str(low_obs.tolist())[1:-1] ))

        # if logging human data, append obs with actions taken
        if log_human:
            self.human_log_file.write(
            '{},{},{},{},{},{},{}\n'.format(self.t,
                self.reward,
                self.act_y,
                self.act_x,
                self.act_z,
                self.act_w,
                str(low_obs.tolist())[1:-1]))

        # also log reward from joystick
        if log_reward:
            self.reward_log_file.write(
            '{},{},{},{},{},{},{}\n'.format(self.t,
                self.joystick_reward,
                self.act_y,
                self.act_x,
                self.act_z,
                self.act_w,
                str(low_obs.tolist())[1:-1]))

class HRI_AirSim_Landing(HRI_AirSim):
    """ Implements HRI AirSim Landing environment.

    In this task it is desired to land a quadrotor on a landing pad.

    States:
        - RGB/Depth center downward camera (AirSim camera ID: 3)

    Actions:
        - Roll, pitch, throttle/altitude
        (Problem can be simplified so the learning algorithm can be tested with
        one, two, or three actions)

    """
    def _take_pic(self):
        """ Take picture and convert to Numpy array
        """
        if self.camera_mode == 'rgb':
            responses = self.client.simGetImages([
                airsim.ImageRequest("3", airsim.ImageType.Scene, False, False)])
            response = responses[0]

            # get numpy array
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)

            # reshape array to 4 channel image array H X W X 4
            if os.name == 'nt': ## Windows
                img_bgr = img1d.reshape(response.height, response.width, 3)
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            else:
                img_rgba = img1d.reshape(response.height, response.width, 4)
                img = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)

            # # original image is fliped vertically
            # img_rgba = np.flipud(img_rgba)

            # update GUI image
            if self.use_pyqt:
                self.display_img = np.rot90(img, k=3)

            # save if desired
            if self.save_training_image_data:
                cv2.imwrite('{}/{}.png'.format(self.name_folder, self.t), img)

        elif self.camera_mode == 'grayscale':
            responses = self.client.simGetImages([
                airsim.ImageRequest("3", airsim.ImageType.Scene,
                             pixels_as_float=True,
                             compress=False)])
            response = responses[0]

            # get numpy array
            img1d = np.array(response.image_data_float, dtype=np.float32)

            # reshape array to 1 channel image array H X W X 1
            img_rgba = img1d.reshape(response.height, response.width)
            img = np.array(img_rgba * 255, dtype=np.uint8)

            # # original image is fliped vertically
            # img_rgba = np.flipud(img_rgba)
            if self.save_training_image_data:
                cv2.imwrite('{}/{}.png'.format(self.name_folder, self.t), img)

        # if rgba or depth mode -> grab RGBA data
        else:
            responses = self.client.simGetImages([
                airsim.ImageRequest("3", airsim.ImageType.Scene, False, False)])
            response = responses[0]

            # get numpy array
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)

            # reshape array to 4 channel image array H X W X 4
            img_rgba = img1d.reshape(response.height, response.width, 4)

            # original image is fliped vertically
            img = np.flipud(img_rgba)

            # write data of this time step to file
            # write image to a png file
            if self.save_training_image_data:
                airsim.write_png(os.path.normpath('{}/{}.png'.format(
                    self.name_folder, self.t)), img)

        return img

    def _read_depth(self, n_depth=1):
        """ Read depth sensor all around vehicle.

        * Args *
        n_depth: number of depth cameras to access
                 1. only front cameras
                 2. front + rear cameras
                 4. front + rear + lateral cameras
        """
        responses = self.client.simGetImages([
            airsim.ImageRequest("3", airsim.ImageType.DepthVis, True)])

        if n_depth == 1:  # only front cameras
            response = responses[0]

            # convert to grayscale
            depth = np.array(response.image_data_float, dtype=np.float32)
            depth = depth.reshape(response.height, response.width)
            depth = np.array(depth * 255, dtype=np.uint8)

            # save pic
            if self.save_training_image_data:
                cv2.imwrite('{}/depth/{}.png'.format(self.name_folder, self.t), depth)

        elif n_depth == 2:  # front + rear cameras
            pass
            # response = responses[0]
            # # get numpy array
            # img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)

            # # reshape array to 4 channel image array H X W X 4
            # depth = img1d.reshape(response.height, response.width, 4)

            # # original image is fliped vertically
            # depth = np.flipud(depth)

            # # save pic
            # if self.save_training_image_data:
            #     cv2.imwrite('{}/depth/{}.png'.format(self.name_folder, self.t), depth)

        return depth

    def step(self, action):
        # check action format and clip to maximum allowed
        if DEBUG_STATEMENTS:
            debug('action received: {} | shape: {} | \
                class: {} | action[0].__class__: {}'.format(action,
                                                            action.shape, action.__class__, action[0].__class__))

        # clip receive angles
        action = np.clip(action, self.action_lb, self.action_ub)

        if self.use_wind:
            # stop wind if reached max number of steps
            if self.wind_curr_step == self.wind_steps:
                self.have_wind = False

            # compute chance of having a discrete wind gust, if no gust is happening now
            if np.random.rand() < self.wind_prob and self.have_wind == False:
                self.have_wind = True
                self.wind_steps = np.random.randint(
                    low=self.min_wind_steps, high=self.max_wind_steps)
                self.wind_curr_step = 0
                self.wind_direction = np.random.randint(low=-1, high=2, size=self.n_act)
            else:
                action_wind = 0

            # update wind if already have it
            if self.have_wind:
                action_wind = self._compute_wind(self.wind_curr_step, self.wind_steps)
                self.wind_curr_step += 1
        
        # read joystick
        rcdata = self.client.getMultirotorState().rc_data
        rcdata = self._apply_deadband(rcdata)
        self.total_steps += 1

        # switch between human and agent by holding top left trigger
        if self.rcdata_switch5:
            self.human_control = True
        else:
            self.human_control = False

        # collect continuous reward function from human when
        # the right trigger is held
        if self.rcdata_switch6:
            self.human_reward = True
        else:
            self.human_reward = False

        # process ACTIONS
        # check if human or agent controlling
        if self.human_control:
            # HUMAN CONTROLLING

            if self.n_act_mode == 5:
                # roll and pitch using right stick
                # throttle using left stick

                # centers joystick between [-1,1]
                self.act_z = -(self.rcdata_throttle-0.75)*2
                self.act_y = self.rcdata_roll
                self.act_x = self.rcdata_pitch
                self.act_w = self.rcdata_yaw

            elif self.n_act_mode == 6:
                # roll and pitch using right stick
                # constant throttle

                # centers joystick between [-1,1]
                self.act_z = self.const_act_z
                self.act_y = self.rcdata_roll
                self.act_x = self.rcdata_pitch
                self.act_w = self.rcdata_yaw

            elif self.n_act_mode == 7:
                # no roll or pitch control
                # throttle using left stick

                # centers joystick between [-1,1]
                self.act_z = -(self.rcdata_throttle-0.75)*2
                self.act_y = 0
                self.act_x = 0
                self.act_w = self.rcdata_yaw

            elif self.n_act_mode == 8:
                # roll and pitch using right stick
                # throttle using left stick

                # centers joystick between [-1,1]
                self.act_z = -(self.rcdata_throttle-0.75)*2
                self.act_y = self.rcdata_roll
                self.act_x = self.rcdata_pitch
                self.act_w = self.rcdata_yaw

            else:
                sys.exit('Invalid n_act_mode. Check config file.')

        else:
            # AGENT CONTROLLING

            # act (holding action based on counter)
            if self.count_action == self.n_hold_action:
                self.count_action = 1

                # check how many actions agent is controlling and act
                if self.n_act_mode == 5:
                    self.act_z = action[2]
                    self.act_y = action[0]
                    self.act_x = action[1]
                    self.act_w = self.rcdata_yaw

                elif self.n_act_mode == 6:
                    self.act_z = self.const_act_z
                    self.act_y = action[0]
                    self.act_x = action[1]
                    self.act_w = self.rcdata_yaw

                elif self.n_act_mode == 7:
                    self.act_z = action[0]
                    self.act_y = 0
                    self.act_x = 0
                    self.act_w = self.rcdata_yaw

                elif self.n_act_mode == 8:
                    # roll and pitch using right stick
                    # throttle using left stick

                    # centers joystick between [-1,1]
                    self.act_z = action[2]
                    self.act_y = action[0]
                    self.act_x = action[1]
                    self.act_w = action[3]

                else:
                    sys.exit('Invalid n_act_mode. Check config file.')
            else:
                self.count_action += 1

        # apply wind
        if self.use_wind:
            wind_effect = action_wind*self.wind_direction
            self.act_x += wind_effect[0]
            self.act_y += wind_effect[1]
            self.act_z += wind_effect[2]
            self.act_w += wind_effect[3]

        # decide level or action control
        self.actions = np.array([self.act_x, self.act_y, self.act_z, self.act_w])
        if self.action_level == 0: #'low':
            self._low_lvl_cmd()
        elif self.action_level == 1: #'high':
            self._high_lvl_cmd()
        elif self.action_level == 2: #'higher':
            sys.exit('Invalid action_level when using HRI_AirSim_Landing. Check config file.')
        else:
            sys.exit('Invalid action_level. Check config file.')


        # NEXT STATES:
        # next states (camera data) based on current camera mode
        # (no matter the mode, we always save image and depth data)
        obs = self._capture_obs()
        pos_z = self.imu[2]

        # DONE:
        # 'select' button (switch7)
        done = self.rcdata_switch7
        # check for crashes
        if self.vehicle_has_collided:
            print('[*] Crash!\n')
            done = 1

        # check if above upper and lower bounds for states
        # (specifically yaw rate)
        curr_yaw_rate = obs[-4]
        if curr_yaw_rate > 1.0 or curr_yaw_rate < -1.0:
            print('[*] Uncontrolled spinning.\n')
            done = 2

        # check max number of steps
        if self.epi_t >= self.n_max_steps - 2: # corrects for step zero and final
            print('[*] Reached max number of simulation steps.\n')
            done = 1

        # check landed state
        if pos_z > 0.8:
            print('[*] Reached minimum altitude.\n')
            done = 1

        # REWARD:
        moving_vehicle_position = self.client.simGetObjectPose('vehicle0').position
        pad_x_loc = -moving_vehicle_position.x_val
        pad_y_loc = moving_vehicle_position.y_val
        pad_z_loc = moving_vehicle_position.z_val-1.65 # 1.65 = height of pad on top of car
        
        dist_to_pad_2d = np.sqrt(( (self.imu[0] - pad_x_loc)**2 + (self.imu[1] - pad_y_loc)**2))
        dist_to_pad_3d = np.sqrt((
            (self.imu[0] - pad_x_loc)**2 + (self.imu[1] - pad_y_loc)**2 + (self.imu[2] - pad_z_loc)**2))
        
        # select between 2d or 3d distance depending on number of actions
        if self.n_act_mode == 6: # 2 actions
            dist_to_pad = dist_to_pad_2d
        elif self.n_act_mode == 8: # 4 actions
            dist_to_pad = dist_to_pad_3d

        # prevents propagation of nan distance values
        if np.isnan(dist_to_pad):
            dist_to_pad = 100.

        # Check if using either sparse or dense reward
        if self.reward_function == 'sparse':
            # reward at the end of episode
            if done == 1:
                # evaluate reward based on final distance
                dist_threshold = 0.75
                if dist_to_pad <= dist_threshold:
                    reward = 1/(1+dist_to_pad**2)
                else:
                    reward = 1/(1+dist_to_pad**2)
            else:
                reward = 0

        elif self.reward_function == 'dense':
            # Reward: distance to landing pad
            reward = 0
            shaping = \
                - 100*dist_to_pad \
                - 100*np.sqrt(self.imu[6]**2 + self.imu[7]**2 + self.imu[8]**2)
                
            if self.epi_t == 0:
                # corrects for fist step
                reward = 0
            else:
                reward = shaping - self.prev_shaping
            self.prev_shaping = shaping
            
            reward -= 0.5  # less fuel spent is better

            if done == 1:
                # evaluate reward based on final distance
                dist_threshold = 0.75
                if dist_to_pad <= dist_threshold:
                    reward += 100
                else:
                    reward += -100

        # print end of episode info
        if done == 1:
            # evaluate reward based on final distance
            dist_threshold = 0.75
            if dist_to_pad <= dist_threshold:
                print('[*] SUCCESS! ')
            else:
                print('[*] FAIL! ')
            # display results
            print('[*] (distance to pad: {:6.4f} | threshold: {} | {} steps)\n'.format(
                dist_to_pad, dist_threshold, self.epi_t))
        
        # record env reward internally so it can be used in other functions
        self.reward = reward

        ## OVERWRITE REWARD
        # use reward collected using the joystick when right trigger is held
        if self.human_reward:
            self.joystick_reward = self.rcdata_pitch
            self._log_low_obs(obs, log_reward=True)
        else:
            self.joystick_reward = 0

        ## August 17: adding env to TAMER reward
        # allows human eval to improve deep rl
        reward += self.joystick_reward

        # additional info
        info = {}

        # update log
        self._log_data(reward, done)
        if self.feature_level == 'low':
            self._log_low_obs(obs, log_obs=True)

            # log human intervention data
            if self.human_control:
                self._log_low_obs(obs, log_human=True)

        # increase data counter
        self.t += 1
        self.epi_t += 1

        return obs, reward, done, info

    def _compute_wind(self, current_step, total_steps):
        """
        Adds discrete gust wind model to AirSim.

        Reference:
        https://www.mathworks.com/help/aeroblks/discretewindgustmodel.html

        The only modification is that wind intensity is computed based on time
        (time step) and not distance.
        """
        # compute current wind velocity
        V_wind = self.Vm/2*(1-np.cos(np.pi*current_step/total_steps))

        return V_wind

    def _high_lvl_cmd(self):
        """Move vehicle by sending high-level velocity commands (vx, vy, vz).
        """
        # send the action (converting to correct format: np.float64)
        sc = 2.5 # scales joystick inputs
        duration = 0.1
        vx = sc*self.act_x
        vy = sc*self.act_y
        vz = 10*sc*self.act_w
        self.alt += self.act_z/(2*sc)

        # translate from inertial to body frame
        th = self.imu[5]
        self.C[0,0] = np.cos(th)
        self.C[0,1] = -np.sin(th)
        self.C[1,0] = -self.C[0,1]
        self.C[1,1] = self.C[0,0]
        vb = self.C.dot(np.array([vx,vy]))

        # send commands
        self.client.moveByVelocityZAsync(vb[0], vb[1], self.alt, duration,
                                airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, vz)).join()

    def _capture_obs(self):
        """Capture observation (depth, image input, etc)
        """
        # get data
        image = self._take_pic()
        if self.camera_mode == 'depth':
            depth = self._read_depth()

        # update GUI variables
        if self.use_pyqt:
            if self.human_control:
                self.control = 'human'
            else:
                self.control = 'agent'
            self.qvals.append(self.qval_t)
            self.qvals_lb.append(self.qval_lb)
            self.qvals_ub.append(self.qval_ub)
            self.ts.append(self.epi_t)

        # vehicle imu
        self._parse_imu()

        if self.feature_level == 'high': # image data
            if self.camera_mode == 'depth':
                obs = depth
            else:
                obs = image

            # scale if necessary
            if self.scale_input:
                obs = cv2.resize(obs, None, fx=self.sf, fy=self.sf)

            # return obs as flat arrays, instead of 2D image format
            if self.flat_images:
                obs = obs.flatten()

        elif self.feature_level == 'low': # landing pad coordinates
            # scale depth image for vae
            x_pad, y_pad, r_pad = self._find_landing_pad(image)

            # check if using perception module or ground truth for the pad
            if not self.use_perception:
                # check pad location and replace perception-based values
                moving_vehicle_position = self.client.simGetObjectPose('vehicle0').position
                x_pad = -moving_vehicle_position.x_val
                y_pad = moving_vehicle_position.y_val

            # remove GPS and acceleration level IMU measurements
            if self.use_gps:
                mod_imu = self.imu[:12]
            else:
                mod_imu = self.imu[2:12]

            # concatenate it to imu values
            obs = np.hstack((mod_imu,x_pad,y_pad,r_pad))

        return obs


    def _find_landing_pad(
        self, image,
        MIN_THRESH=(150.0, 0.0, 0.0, 0.0),
        MAX_THRESH=(255.0, 255.0, 255.0, 0.0)):
        """ Takes an image as input and return its binarized version
        according to a color-based filter
        """
        def thresholded_image(image, MIN_THRESH, MAX_THRESH):
            # convert image to hsv
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # threshold the image
            image_threshed = cv2.inRange(image_hsv, MIN_THRESH, MAX_THRESH)

            return image_threshed

        # smooth the image
        image_smoothed = image.copy()
        image_smoothed = cv2.blur(image,(5,5))

        # threshold the smoothed image
        image_threshed = thresholded_image(image_smoothed, MIN_THRESH, MAX_THRESH)

        # blobify
        image_threshed = cv2.dilate(image_threshed, None, 18)
        image_threshed = cv2.erode(image_threshed, None, 10)

        blobContour = None

        # extract the edges from our binary image
        cnts = cv2.findContours(image_threshed.copy(),
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        if len(cnts) == 0:
            return (0, 0, 0)
        else:
            # loop over the contours and find largest
            largest_area = 0
            largest_cnt_idx = 0
            cnt_idx = 0

            for c in cnts:
                area = cv2.contourArea(c)
                if area > largest_area:
                    largest_area = area
                    largest_cnt_idx = cnt_idx
                cnt_idx += 1

            # compute the center of the largest contour
            try:
                # fit circle around contour area
                (x,y), radius = cv2.minEnclosingCircle(cnts[largest_cnt_idx])
                cX = int(x)
                cY = int(y)

            except:
                print('Target too small')
                cX = 0
                cY = 0
                radius = 0

            return (cX, cY, radius)
