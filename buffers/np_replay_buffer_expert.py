import numpy as np
from buffers.replay_buffer import AbstractReplayBuffer

class EfficientReplayBuffer(AbstractReplayBuffer):
    def __init__(self, buffer_size, batch_size, nstep, discount, frame_stack, frame_stack_sample,
                 data_specs=None):
        self.buffer_size = buffer_size
        self.data_dict = {}
        self.index = -1
        self.traj_index = 0
        self.frame_stack = frame_stack
        self.frame_stack_sample = frame_stack_sample
        self._recorded_frames = frame_stack + 1
        self.batch_size = batch_size
        self.nstep = nstep
        self.discount = discount
        self.full = False
        # fixed since we can only sample transitions that occur nstep earlier
        # than the end of each episode or the last recorded observation
        self.discount_vec = np.power(discount, np.arange(nstep)).astype('float32')
        self.next_dis = discount**nstep

    def _initial_setup(self, time_step):
        self.index = 0
        self.obs_shape = list(time_step.observation.shape)
        self.ims_channels = self.obs_shape[0] // self.frame_stack
        self.sample_obs_shape = list(time_step.observation.shape)
        self.sample_obs_shape[0] = self.ims_channels*self.frame_stack_sample
        self.act_shape = time_step.action.shape
        self.obs_sensor_shape = time_step.observation_sensor.shape

        self.obs = np.zeros([self.buffer_size, self.ims_channels, *self.obs_shape[1:]], dtype=np.uint8)
        self.obs_sensor = np.zeros([self.buffer_size, *self.obs_sensor_shape], dtype=np.float32)
        self.act = np.zeros([self.buffer_size, *self.act_shape], dtype=np.float32)
        self.rew = np.zeros([self.buffer_size], dtype=np.float32)
        self.dis = np.zeros([self.buffer_size], dtype=np.float32)
        # which timesteps can be validly sampled (Not within nstep from end of
        # an episode or last recorded observation)
        self.valid = np.zeros([self.buffer_size], dtype=np.bool_)

    def add_data_point(self, time_step):
        first = time_step.first()
        latest_obs = time_step.observation[-self.ims_channels:]
        latest_obs_sensor = time_step.observation_sensor

        if first:
            # if first observation in a trajectory, record frame_stack copies of it
            end_index = self.index + self.frame_stack
            end_invalid = end_index + self.frame_stack + 1
            if end_invalid > self.buffer_size:
                if end_index > self.buffer_size:
                    end_index = end_index % self.buffer_size
                    self.obs[self.index:self.buffer_size] = latest_obs
                    self.obs[0:end_index] = latest_obs
                    self.obs_sensor[self.index:self.buffer_size] = latest_obs_sensor
                    self.obs_sensor[0:end_index] = latest_obs_sensor
                    self.full = True
                else:
                    self.obs[self.index:end_index] = latest_obs
                    self.obs_sensor[self.index:end_index] = latest_obs_sensor
                end_invalid = end_invalid % self.buffer_size
                self.valid[self.index:self.buffer_size] = False
                self.valid[0:end_invalid] = False
            else:
                self.obs[self.index:end_index] = latest_obs
                self.obs_sensor[self.index:end_index] = latest_obs_sensor
                self.valid[self.index:end_invalid] = False
            self.index = end_index
            self.traj_index = 1
        else:
            np.copyto(self.obs[self.index], latest_obs)
            np.copyto(self.act[self.index], time_step.action)
            np.copyto(self.obs_sensor[self.index], latest_obs_sensor)
            self.rew[self.index] = time_step.reward
            self.dis[self.index] = time_step.discount
            self.valid[(self.index + self.frame_stack) % self.buffer_size] = False
            if self.traj_index >= self.nstep:
                self.valid[(self.index - self.nstep + 1) % self.buffer_size] = True
            self.index += 1
            self.traj_index += 1
            if self.index == self.buffer_size:
                self.index = 0
                self.full = True

    def add(self, time_step):
        if self.index == -1:
            self._initial_setup(time_step)
        self.add_data_point(time_step)

    def __next__(self, ):
        # sample only valid indices
        indices = np.random.choice(self.valid.nonzero()[0], size=self.batch_size)
        return self.gather_nstep_indices(indices)

    def gather_nstep_indices(self, indices):
        n_samples = indices.shape[0]
        all_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack_sample, indices[i] + self.nstep)
                                  for i in range(n_samples)], axis=0) % self.buffer_size
        gather_ranges = all_gather_ranges[:, self.frame_stack_sample:] # bs x nstep
        obs_gather_ranges = all_gather_ranges[:, :self.frame_stack_sample]
        nobs_gather_ranges = all_gather_ranges[:, -self.frame_stack_sample:]

        all_rewards = self.rew[gather_ranges]

        # Could implement reward computation as a matmul in pytorch for
        # marginal additional speed improvement
        rew = np.sum(all_rewards * self.discount_vec, axis=1, keepdims=True)

        obs = np.reshape(self.obs[obs_gather_ranges], [n_samples, *self.sample_obs_shape])
        nobs = np.reshape(self.obs[nobs_gather_ranges], [n_samples, *self.sample_obs_shape])

        act = self.act[indices]
        obs_sensor = self.obs_sensor[obs_gather_ranges[:, -1]]
        next_obs_sensor = self.obs_sensor[nobs_gather_ranges[:, -1]]

        dis = np.expand_dims(self.next_dis * self.dis[nobs_gather_ranges[:, -1]], axis=-1)

        ret = (obs, obs_sensor, act, rew, dis, nobs, next_obs_sensor)
        return ret
    
    def gather_images(self):
        indices = np.random.choice(self.valid.nonzero()[0], size=self.batch_size)
        return self.obs[indices, :, :, :]

    def __len__(self):
        if self.full:
            return self.buffer_size
        else:
            return self.index
