# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import cv2
import imageio
import numpy as np

class VideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, env, enabled=True):
        self.env_name = env.env_name

        if self.env_name == 'relocate-v0':
            self.cam_1 = []
            self.cam_2 = []
            self.cam_3 = []
        else:
            self.frames = []

        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            if hasattr(env, 'physics'):
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=0)
            else:
                frame = env.get_pixels_with_width_height(self.render_size, self.render_size)
                frame = frame.transpose((1, 2, 0))

            if self.env_name == 'relocate-v0':
                self.cam_1.append(frame[:,:,:3])
                self.cam_2.append(frame[:,:,3:6])
                self.cam_3.append(frame[:,:,6:])

            else:
                self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:

            if self.env_name == 'relocate-v0':
                imageio.mimsave(str(self.save_dir) + '/cam_1_' + str(file_name), self.cam_1, fps=self.fps)
                imageio.mimsave(str(self.save_dir) + '/cam_2_' + str(file_name), self.cam_2, fps=self.fps)
                imageio.mimsave(str(self.save_dir) + '/cam_3_' + str(file_name), self.cam_3, fps=self.fps)

            else:
                path = self.save_dir / file_name
                imageio.mimsave(str(path), self.frames, fps=self.fps)


class TrainVideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / 'train_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            frame = cv2.resize(obs[-3:].transpose(1, 2, 0),
                               dsize=(self.render_size, self.render_size),
                               interpolation=cv2.INTER_CUBIC)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
