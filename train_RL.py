# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os, psutil
# make sure mujoco and nvidia will be found
# os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + \
#                                 ':/workspace/.mujoco/mujoco210/bin:/usr/local/nvidia/lib:/usr/lib/nvidia'
# os.environ['MUJOCO_PY_MUJOCO_PATH'] = '/workspace/.mujoco/mujoco210/'
import numpy as np
import shutil
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
import platform
os.environ['MUJOCO_GL'] = 'egl'
# set to glfw if trying to render locally with a monitor
# os.environ['MUJOCO_GL'] = 'glfw'
#os.environ['EGL_DEVICE_ID'] = '0' 

from distutils.dir_util import copy_tree
import torchvision.transforms as transforms
from pathlib import Path
import cv2
import imageio

import hydra
import torch
from dm_env import StepType, TimeStep, specs

from utils_folder import utils_vrl3 as utils
from logger import Logger
from video import TrainVideoRecorder, VideoRecorder
import joblib
import pickle
import time

torch.backends.cudnn.benchmark = True

# TODO this part can be done during workspace setup
ENV_TYPE = 'adroit'
import mj_envs
from mjrl.utils.gym_env import GymEnv
from rrl_local.rrl_utils import make_basic_env, make_dir
from adroit import AdroitEnv

IS_ADROIT = True if ENV_TYPE == 'adroit' else False

def make_agent(obs_spec, obs_sensor_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.obs_sensor_shape = obs_sensor_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)

def print_time_est(time_used, curr_n_frames, total_n_frames):
    time_per_update = time_used / curr_n_frames
    est_total_time = time_per_update * total_n_frames
    est_time_remaining = est_total_time - time_used
    print("Stage 3 [{:.2f}%]. Frames:[{:.0f}/{:.0f}]K. Time:[{:.2f}/{:.2f}]hrs. Overall FPS: {}.".format(
        curr_n_frames / total_n_frames * 100, curr_n_frames/1000, total_n_frames/1000,
        time_used / 3600, est_total_time / 3600, int(curr_n_frames / time_used)))

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print("\n=== Training log stored to: ===")
        print(f'workspace: {self.work_dir}')
        self.direct_folder_name = os.path.basename(self.work_dir)

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.observation_sensor_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        warnings.filterwarnings('ignore', category=DeprecationWarning)

        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        env_name = self.cfg.task_name
        env_type = 'adroit' if env_name in ('hammer-v0',
                                            'hammer_light-v0',
                                            'hammer_color-v0',
                                            'door-v0',
                                            'pen-v0',
                                            'pen_light-v0',
                                            'pen_color-v0',
                                            'relocate-v0', 
                                            'door_light-v0',
                                            'door_color-v0') else NotImplementedError
        # assert env_name in ('hammer-v0','door-v0','pen-v0','relocate-v0',)

        if self.cfg.agent.encoder_lr_scale == 'auto':
            if env_name == 'relocate-v0':
                self.cfg.agent.encoder_lr_scale = 0.01
            else:
                self.cfg.agent.encoder_lr_scale = 1

        self.env_feature_type = self.cfg.env_feature_type
        
        # reward rescale can either be added in the env or in the agent code when reward is used
        self.train_env = AdroitEnv(env_name, 
                                   test_image=False, 
                                   num_repeats=self.cfg.action_repeat,
                                   num_frames=self.cfg.frame_stack, 
                                   env_feature_type=self.env_feature_type,
                                   device=self.device, 
                                   reward_rescale=self.cfg.reward_rescale)
        
        self.eval_env = AdroitEnv(env_name, 
                                  test_image=False, 
                                  num_repeats=self.cfg.action_repeat,
                                  num_frames=self.cfg.frame_stack, 
                                  env_feature_type=self.env_feature_type,
                                  device=self.device, 
                                  reward_rescale=self.cfg.reward_rescale)

        data_specs = (self.train_env.observation_spec(),
                    self.train_env.observation_sensor_spec(),
                    self.train_env.action_spec(),
                    specs.Array((1,), np.float32, 'reward'),
                    specs.Array((1,), np.float32, 'discount'),
                    specs.Array((1,), np.int8, 'n_goal_achieved'),
                    specs.Array((1,), np.float32, 'time_limit_reached'),
                    )

        # create replay buffer
        self.replay_buffer = hydra.utils.instantiate(self.cfg.replay_buffer, data_specs=data_specs)

        self.video_recorder = VideoRecorder(self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(self.work_dir if self.cfg.save_train_video else None)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        total_success = 0.0

        while eval_until_episode(episode):
            n_goal_achieved_total = 0
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    observation = time_step.observation
                    action = self.agent.act(observation,
                                            self.global_step,
                                            eval_mode=True,
                                            obs_sensor=time_step.observation_sensor)
                    
                time_step = self.eval_env.step(action)
                n_goal_achieved_total += time_step.n_goal_achieved
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            if self.cfg.task_name == 'pen-v0':
                threshold = 20
            else:
                threshold = 25

            if n_goal_achieved_total > threshold:
                total_success += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        success_rate_standard = total_success / self.cfg.num_eval_episodes
        episode_reward_standard = total_reward / episode
        episode_length_standard = step*self.cfg.action_repeat / episode

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', episode_reward_standard)
            log('success_rate', success_rate_standard)
            log('episode_length', episode_length_standard)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def get_data_folder_path(self):
        pretrained_encoder_path = self.work_dir.parents[4]
        return str(pretrained_encoder_path)
    
    def get_pretrained_model_path(self, stage1_model_name):
        # given a stage1 model name, return the path to the pretrained model
        data_folder_path = self.get_data_folder_path()
        model_folder_path = os.path.join(data_folder_path, "pretrained_encoders")
        model_path = os.path.join(model_folder_path, stage1_model_name + '_checkpoint.pth.tar')
        return model_path
    
    def get_demo_path(self, env_name):
        # given a environment name (with the -v0 part), return the path to its demo file
        data_folder_path = self.get_data_folder_path()
        demo_folder_path = os.path.join(data_folder_path, "demonstrations")
        demo_path = os.path.join(demo_folder_path, env_name + "_demos.pickle")
        return demo_path
    
    def load_demo(self, replay_buffer, env_name, verbose=True):
        # will load demo data and put them into a replay storage
        demo_path = self.get_demo_path(env_name)
        if verbose:
            print("Trying to get demo data from:", demo_path)

        # get the raw state demo data, a list of length 25
        demo_data = pickle.load(open(demo_path, 'rb'))
        if self.cfg.num_demo >= 0:
            demo_data = demo_data[:self.cfg.num_demo]

        """
        the adroit demo data is in raw state, so we need to convert them into image data
        we init an env and run episodes with the stored actions in the demo data
        then put the image and sensor data into the replay buffer, also need to clip actions here 
        this part is basically following the RRL code
        """
        demo_env = AdroitEnv(env_name, test_image=False, num_repeats=1, num_frames=self.cfg.frame_stack,
                env_feature_type=self.env_feature_type, device=self.device, reward_rescale=self.cfg.reward_rescale)
        demo_env.reset()

        total_data_count = 0
        for i_path in range(len(demo_data)):
            path = demo_data[i_path]
            demo_env.reset()
            demo_env.set_env_state(path['init_state_dict'])
            time_step = demo_env.get_current_obs_without_reset()
            replay_buffer.add(time_step)

            ep_reward = 0
            ep_n_goal = 0
            for i_act in range(len(path['actions'])):
                total_data_count += 1
                action = path['actions'][i_act]
                action = action.astype(np.float32)
                # when action is put into the environment, they will be clipped.
                action[action > 1] = 1
                action[action < -1] = -1

                # when they collect the demo data, they actually did not use a timelimit...
                if i_act == len(path['actions']) - 1:
                    force_step_type = 'last'
                else:
                    force_step_type = 'mid'

                time_step = demo_env.step(action, force_step_type=force_step_type)
                replay_buffer.add(time_step)

                reward = time_step.reward
                ep_reward += reward

                goal_achieved = time_step.n_goal_achieved
                ep_n_goal += goal_achieved
            if verbose:
                print('demo trajectory %d, len: %d, return: %.2f, goal achieved steps: %d' %
                      (i_path, len(path['actions']), ep_reward, ep_n_goal))
        if verbose:
            print("Demo data load finished, total data count:", total_data_count)

    def train(self):
        training_start_time = time.time()

        """========================================= LOAD DATA ========================================="""
        if self.cfg.load_demo:
            self.load_demo(self.replay_buffer, self.cfg.task_name)
        print("Model and data loading finished in %.2f hours." % ((time.time()-training_start_time) / 3600))

        print("\n=== Training started! ===")
        """=================================== LOAD PRETRAINED MODEL ==================================="""
        if self.cfg.pretrained_encoder:
            self.agent.load_pretrained_encoder(self.get_pretrained_model_path(self.cfg.pretrained_encoder_model_name))
        self.agent.switch_to_RL_stages()

        training_start_time = time.time()
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames, self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames, self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_buffer.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None

        episode_step_since_log, episode_reward_list, episode_frame_list = 0, [0], [0]
        self.timer.reset()
        while train_until_step(self.global_step):
            # if 1000 steps passed, do some logging
            if self.global_step % 1000 == 0 and metrics is not None:
                elapsed_time, total_time = self.timer.reset()
                episode_frame_since_log = episode_step_since_log * self.cfg.action_repeat
                with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                        log('fps', episode_frame_since_log / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', np.mean(episode_reward_list))
                        log('episode_length', np.mean(episode_frame_list))
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_buffer))
                        log('step', self.global_step)
                episode_step_since_log, episode_reward_list, episode_frame_list = 0, [0], [0]

            if self.cfg.show_computation_time_est and self.global_step > 0 and self.global_step % self.cfg.show_time_est_interval == 0:
                print_time_est(time.time() - training_start_time, self.global_frame + 1, self.cfg.num_train_frames)

            # if reached end of episode
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')

                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    episode_step_since_log += episode_step
                    episode_reward_list.append(episode_reward)
                    episode_frame = episode_step * self.cfg.action_repeat
                    episode_frame_list.append(episode_frame)

                # reset env
                time_step = self.train_env.reset()
                self.replay_buffer.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step, episode_reward = 0, 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
                self.eval()

            # sample action
            obs_sensor = time_step.observation_sensor
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                    self.global_step,
                                    eval_mode=False,
                                    obs_sensor=obs_sensor)

            # update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_buffer, self.global_step, stage=3, use_sensor=IS_ADROIT)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_buffer.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

            """here save model for later"""
            if self.cfg.save_models:
                if self.global_frame in (2, 100000, 500000, 1000000, 2000000, 4000000):
                    self.save_snapshot(suffix=f"{self.cfg.task_name}_action_repeat={self.cfg.action_repeat}_frame_stack={self.cfg.frame_stack}_{self.global_frame}")

        print("Training finished in %.2f hours." % ((time.time()-training_start_time) / 3600))
        print(self.work_dir)

    def save_snapshot(self, suffix=None):
        if suffix is None:
            save_name = 'snapshot.pt'
        else:
            save_name = 'snapshot_' + suffix + '.pt'
        snapshot = self.work_dir / save_name
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)
        print("snapshot saved to:", str(snapshot))

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

@hydra.main(config_path='config_folder', config_name='config_RL')
def main(cfg):
    from train_RL import Workspace as W
    workspace = W(cfg)
    workspace.train()

if __name__ == '__main__':
    main()