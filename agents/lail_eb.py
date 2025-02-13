import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as torchd
from torch import autograd
from torch.nn.utils import spectral_norm 
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms.functional import rgb_to_grayscale

from utils_folder import utils
from utils_folder.utils_dreamer import Bernoulli
from utils_folder.resnet import BasicBlock, ResNet84

import os

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class NoAug(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
class Identity(nn.Module):
    def __init__(self, input_placeholder=None):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
class EventBasedTransform(nn.Module):
    def __init__(self, theta=0.15, noise_sigma=0.005):
        super().__init__()
        self.theta = theta  # Brightness change threshold (in log intensity).
        self.noise_sigma = noise_sigma  # Standard deviation of Gaussian noise added to delta_log

    def forward(self, obs, ref_log=None):
        """
        :param obs: (B, C, H, W), where C = 3 * num_frames (RGB frames concatenated in channels)
        :param ref_log: (B, 1, H, W) or None. If None, it's initialized with the first frame.
        :return: event_frames (B, 3*(T-1), H, W)
        """
        b, c, h, w = obs.size()
        num_frames = c // 3  # RGB stacked along the channel dimension

        # Initialize reference log intensity if not provided
        if ref_log is None:
            ref_log = torch.log(rgb_to_grayscale(obs[:, :3, :, :]) / 255.0 + 1e-5)  # (B, 1, H, W)
        else:
            ref_log = ref_log.clone()  # Prevent modifying the external ref_log

        event_frames = []

        for i in range(num_frames - 1):
            frame1 = rgb_to_grayscale(obs[:, 3*i:3*i+3, :, :])  # (B, 1, H, W)
            frame2 = rgb_to_grayscale(obs[:, 3*(i+1):3*(i+1)+3, :, :])  # (B, 1, H, W)

            curr_log = torch.log(frame2 / 255.0 + 1e-5)
            delta_log = curr_log - ref_log

            # True event masks
            true_positive_events = delta_log > self.theta  # (B, 1, H, W)
            true_negative_events = delta_log < -self.theta  # (B, 1, H, W)

            # Add Gaussian noise
            noisy_delta = delta_log + torch.randn_like(delta_log) * self.noise_sigma  # (B, 1, H, W)

            # Noisy event decisions
            positive_events = noisy_delta > self.theta  # (B, 1, H, W)
            negative_events = noisy_delta < -self.theta  # (B, 1, H, W)

            # Create event visualization frame (batch-wise, RGB order)
            event_frame = torch.zeros((b, 3, h, w), dtype=torch.uint8, device=obs.device)  # (B, 3, H, W)
            event_frame[:, 1, :, :].masked_fill_(positive_events.squeeze(1), 255)  # Green for positive events
            event_frame[:, 0, :, :].masked_fill_(negative_events.squeeze(1), 255)  # Red for negative events

            event_frames.append(event_frame)  # Store per-frame event visualization

            # Update ref_log only using true events
            ref_log = torch.where(true_positive_events, curr_log, ref_log)
            ref_log = torch.where(true_negative_events, curr_log, ref_log)

        return torch.cat(event_frames, dim=1)  # (B, 3*(T-1), H, W)
    
class PretrainedEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, model_name, device):
        super().__init__()
        # a wrapper over a non-RL encoder model
        self.device = device
        assert len(obs_shape) == 3
        self.n_input_channel = obs_shape[0]-3
        assert self.n_input_channel % 3 == 0
        self.n_images = self.n_input_channel // 3
        self.model = self.init_model(model_name)
        self.model.fc = Identity()
        self.repr_dim = self.model.get_feature_size()

        self.normalize_op = transforms.Normalize((0.485, 0.456, 0.406),
                                                 (0.229, 0.224, 0.225))
        self.channel_mismatch = True

        self.trunk = nn.Sequential(nn.Linear(self.repr_dim, feature_dim),
                        nn.LayerNorm(feature_dim), nn.Tanh())
        
        self.apply(utils.weight_init)

    def init_model(self, model_name):
        # model name is e.g. resnet6_32channel
        n_layer_string, n_channel_string = model_name.split('_')

        layer_string_to_layer_list = {
            'resnet6': [0, 0, 0, 0],
            'resnet10': [1, 1, 1, 1],
            'resnet18': [2, 2, 2, 2],
        }

        channel_string_to_n_channel = {
            '32channel': 32,
            '64channel': 64,
        }

        layer_list = layer_string_to_layer_list[n_layer_string]
        start_num_channel = channel_string_to_n_channel[n_channel_string]
        return ResNet84(BasicBlock, layer_list, start_num_channel=start_num_channel).to(self.device)

    def expand_first_layer(self):
        # convolutional channel expansion to deal with input mismatch
        multiplier = self.n_images
        self.model.conv1.weight.data = self.model.conv1.weight.data.repeat(1,multiplier,1,1) / multiplier
        means = (0.485, 0.456, 0.406) * multiplier
        stds = (0.229, 0.224, 0.225) * multiplier
        self.normalize_op = transforms.Normalize(means, stds)
        self.channel_mismatch = False

    def freeze_bn(self):
        # freeze batch norm layers (VRL3 ablation shows modifying how
        # batch norm is trained does not affect performance)
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()

    def get_parameters_that_require_grad(self):
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                params.append(param)
        return params

    def transform_obs_tensor_batch(self, obs):
        # transform obs batch before put into the pretrained resnet
        new_obs = self.normalize_op(obs.float()/255)
        return new_obs

    def _forward_impl(self, x):
        x = self.model.get_features(x)
        return x

    def forward(self, obs):
        o = self.transform_obs_tensor_batch(obs)
        h = self._forward_impl(o)
        z = self.trunk(h)
        return z

class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        self.feature_dim = (32,35,35)

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0]-3, 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        
        self.trunk = nn.Sequential(nn.Linear(self.repr_dim, feature_dim),
                                nn.LayerNorm(feature_dim), nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        z = self.trunk(h)
        return z
        
class Discriminator(nn.Module):
    def __init__(self, input_net_dim, hidden_dim, spectral_norm_bool=False, dist=None):
        super().__init__()
                
        self.dist = dist
        self._shape = (1,)

        if spectral_norm_bool:
            self.net = nn.Sequential(spectral_norm(nn.Linear(input_net_dim, hidden_dim)),
                                    nn.ReLU(inplace=True),
                                    spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
                                    nn.ReLU(inplace=True),
                                    spectral_norm(nn.Linear(hidden_dim, 1)))  

        else:
            self.net = nn.Sequential(nn.Linear(input_net_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, 1))  
        
        self.apply(utils.weight_init)

    def forward(self, transition):
        d = self.net(transition)

        if self.dist == 'binary':
            return Bernoulli(torchd.independent.Independent(torchd.bernoulli.Bernoulli(logits=d), len(self._shape)))
        else:
            return d 

class Actor(nn.Module):
    def __init__(self, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        mu = self.policy(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist

class Critic(nn.Module):
    def __init__(self, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2

class LailEbAgent:
    def __init__(self, 
                 obs_shape, 
                 obs_sensor_shape,
                 action_shape, 
                 device, 
                 lr, 
                 feature_dim,
                 hidden_dim, 
                 critic_target_tau, 
                 num_expl_steps,
                 update_every_steps, 
                 stddev_schedule, 
                 stddev_clip, 
                 use_tb, 
                 reward_d_coef, 
                 discriminator_lr, 
                 spectral_norm_bool, 
                 check_every_steps, 
                 pretrained_encoder_path, 
                 encoder_lr_scale, 
                 pretrained_encoder=False, 
                 pretrained_encoder_model_name = 'resnet6_32channel', 
                 GAN_loss='bce', 
                 from_dem=False, 
                 add_aug=True, 
                 RL_plus_IL = False):
        
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.GAN_loss = GAN_loss
        self.from_dem = from_dem
        self.RL_plus_IL = RL_plus_IL
        self.check_every_steps = check_every_steps
        self.eb_transform = EventBasedTransform()
        
        if pretrained_encoder:
            self.encoder = PretrainedEncoder(obs_shape, feature_dim, pretrained_encoder_model_name, device).to(device)
            self.load_pretrained_encoder(pretrained_encoder_path)
            self.encoder.expand_first_layer()
            print("Convolutional channel expansion finished: now can take in %d images as input." % self.encoder.n_images)
            encoder_lr = lr * encoder_lr_scale

        else:
            self.encoder = Encoder(obs_shape, feature_dim).to(device)
            encoder_lr = lr 

        downstream_input_dim = feature_dim+obs_sensor_shape[0]

        self.actor = Actor(action_shape, downstream_input_dim, hidden_dim).to(device)
        self.critic = Critic(action_shape, downstream_input_dim, hidden_dim).to(device)
        self.critic_target = Critic(action_shape, downstream_input_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # added model
        if from_dem:
            if self.GAN_loss == 'least-square':
                self.discriminator = Discriminator(feature_dim+action_shape[0], hidden_dim, spectral_norm_bool).to(device)
                self.reward_d_coef = reward_d_coef

            elif self.GAN_loss == 'bce':
                self.discriminator = Discriminator(feature_dim+action_shape[0], hidden_dim, spectral_norm_bool, dist='binary').to(device)
            else:
                NotImplementedError

        else:
            if self.GAN_loss == 'least-square':
                self.discriminator = Discriminator(2*feature_dim, hidden_dim, spectral_norm_bool).to(device)
                self.reward_d_coef = reward_d_coef

            elif self.GAN_loss == 'bce':
                self.discriminator = Discriminator(2*feature_dim, hidden_dim, spectral_norm_bool, dist='binary').to(device)
            else:
                NotImplementedError

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), lr=discriminator_lr)

        # data augmentation
        if add_aug:
            self.aug = RandomShiftsAug(pad=4)
        else:
            self.aug = NoAug()

        self.train()
        self.critic_target.train()

    def load_pretrained_encoder(self, model_path, verbose=True):
        if verbose:
            print("Trying to load pretrained model from:", model_path)

        checkpoint = torch.load(model_path, map_location=torch.device(self.device))
        state_dict = checkpoint['state_dict']

        pretrained_dict = {}
        # remove `module.` if model was pretrained with distributed mode
        for k, v in state_dict.items():
            if 'module.' in k:
                name = k[7:]
            else:
                name = k
            pretrained_dict[name] = v

        self.encoder.model.load_state_dict(pretrained_dict, strict=False)

        if verbose:
            print("Pretrained model loaded!")

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.discriminator.train(training)

    def act(self, obs, obs_sensor, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(self.eb_transform(obs.unsqueeze(0))) #Eb transformation

        obs_sensor = torch.as_tensor(obs_sensor, device=self.device)
        obs_sensor = obs_sensor.unsqueeze(0)
        obs_combined = torch.cat([obs, obs_sensor], dim=1)

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs_combined, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics
    
    def compute_reward(self, obs_a, next_a, reward_a):
        metrics = dict()

        # augment
        if self.from_dem:
            obs_a = self.aug(obs_a.float())
        else:
            obs_a = self.aug(obs_a.float())
            next_a = self.aug(next_a.float())
        
        # encode
        with torch.no_grad():
            if self.from_dem:
                obs_a = self.encoder(obs_a)
            else:
                obs_a = self.encoder(obs_a)
                next_a = self.encoder(next_a)
        
            self.discriminator.eval()
            transition_a = torch.cat([obs_a, next_a], dim = -1)

            d = self.discriminator(transition_a)

            if self.GAN_loss == 'least-square':
                reward_d = self.reward_d_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)

            elif self.GAN_loss == 'bce':
                reward_d = d.mode()
            
            if self.RL_plus_IL:
                reward = reward_d + reward_a

            else:
                reward = reward_d

            if self.use_tb:
                metrics['reward_d'] = reward_d.mean().item()
    
            self.discriminator.train()
            
        return reward, metrics
    
    def compute_discriminator_grad_penalty_LS(self, obs_e, next_e, lambda_=10):
        
        expert_data = torch.cat([obs_e, next_e], dim=-1)
        expert_data.requires_grad = True
        
        d = self.discriminator(expert_data)

        ones = torch.ones(d.size(), device=self.device)
        grad = autograd.grad(outputs=d, inputs=expert_data, grad_outputs=ones, create_graph=True,
                             retain_graph=True, only_inputs=True)[0]
        
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def compute_discriminator_grad_penalty_bce(self, obs_a, next_a, obs_e, next_e, lambda_=10):

        agent_feat = torch.cat([obs_a, next_a], dim=-1)
        alpha = torch.rand(agent_feat.shape[:1]).unsqueeze(-1).to(self.device)
        expert_data = torch.cat([obs_e, next_e], dim=-1)
        disc_penalty_input = alpha*agent_feat + (1-alpha)*expert_data

        disc_penalty_input.requires_grad = True

        d = self.discriminator(disc_penalty_input).mode()

        ones = torch.ones(d.size(), device=self.device)
        grad = autograd.grad(outputs=d, inputs=disc_penalty_input, grad_outputs=ones, create_graph=True,
                             retain_graph=True, only_inputs=True)[0]
        
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen
        
    def update_discriminator(self, obs_a, next_a, obs_e, next_e):
        metrics = dict()

        transition_a = torch.cat([obs_a, next_a], dim=-1)
        transition_e = torch.cat([obs_e, next_e], dim=-1)
        
        agent_d = self.discriminator(transition_a)
        expert_d = self.discriminator(transition_e)

        if self.GAN_loss == 'least-square':
            expert_labels = 1.0
            agent_labels = -1.0

            expert_loss = F.mse_loss(expert_d, expert_labels*torch.ones(expert_d.size(), device=self.device))
            agent_loss = F.mse_loss(agent_d, agent_labels*torch.ones(agent_d.size(), device=self.device))
            grad_pen_loss = self.compute_discriminator_grad_penalty_LS(obs_e.detach(), next_e.detach())
            loss = 0.5*(expert_loss + agent_loss) + grad_pen_loss
        
        elif self.GAN_loss == 'bce':
            expert_loss = (expert_d.log_prob(torch.ones_like(expert_d.mode()).to(self.device))).mean()
            agent_loss = (agent_d.log_prob(torch.zeros_like(agent_d.mode()).to(self.device))).mean()
            grad_pen_loss = self.compute_discriminator_grad_penalty_bce(obs_a.detach(), next_a.detach(), obs_e.detach(), next_e.detach())
            loss = -(expert_loss+agent_loss) + grad_pen_loss

        self.discriminator_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.discriminator_opt.step()
        
        if self.use_tb:
            metrics['discriminator_expert_loss'] = expert_loss.item()
            metrics['discriminator_agent_loss'] = agent_loss.item()
            metrics['discriminator_loss'] = loss.item()
            metrics['discriminator_grad_pen'] = grad_pen_loss.item()
        
        return metrics        

    def update(self, replay_iter, replay_iter_expert, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics
        
        batch = next(replay_iter)
        obs, obs_sensor, action, reward_a, discount, next_obs, next_obs_sensor = utils.to_torch(batch, self.device)
        
        batch_expert = next(replay_iter_expert)
        obs_e_raw, _, action_e, _, _, next_obs_e_raw, _ = utils.to_torch(batch_expert, self.device)

        if step % self.check_every_steps == 0:
            self.check_aug(obs, next_obs, obs_e_raw, next_obs_e_raw, "raw_images", step)

        obs = self.eb_transform(obs)
        next_obs = self.eb_transform(next_obs)
        obs_e_raw = self.eb_transform(obs_e_raw)
        next_obs_e_raw = self.eb_transform(next_obs_e_raw)

        if step % self.check_every_steps == 0:
            self.check_aug(obs, next_obs, obs_e_raw, next_obs_e_raw, "learning_buffer", step)
        
        obs_e = self.aug(obs_e_raw.float())
        next_obs_e = self.aug(next_obs_e_raw.float())
        obs_a = self.aug(obs.float())
        next_obs_a = self.aug(next_obs.float())

        with torch.no_grad():
            obs_e = self.encoder(obs_e)
            next_obs_e = self.encoder(next_obs_e)
            obs_a = self.encoder(obs_a)
            next_obs_a = self.encoder(next_obs_a)

        # update critic
        if self.from_dem:
            metrics.update(self.update_discriminator(obs_a, action, obs_e, action_e))

            if self.RL_plus_IL:
                reward, metrics_r = self.compute_reward(obs, action, reward_a)
            else:
                reward, metrics_r = self.compute_reward(obs, action, reward_a=0)

        else:
            metrics.update(self.update_discriminator(obs_a, next_obs_a, obs_e, next_obs_e))

            if self.RL_plus_IL:
                reward, metrics_r = self.compute_reward(obs, next_obs, reward_a)
            else:
                reward, metrics_r = self.compute_reward(obs, next_obs, reward_a=0)

        metrics.update(metrics_r)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward_a.mean().item()

        # combine encoded image with sensors observations
        obs_combined = torch.cat([obs, obs_sensor], dim=1)
        next_obs_combined = torch.cat([next_obs, next_obs_sensor], dim=1)

        # update critic
        metrics.update(self.update_critic(obs_combined, action, reward, discount, next_obs_combined, step))

        # update actor
        metrics.update(self.update_actor(obs_combined.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics
    
    def check_aug(self, obs, next_obs, obs_e, next_obs_e, type, step):

        if not os.path.exists(f'checkimages_{type}'):
            os.makedirs(f"checkimages_{type}")

        obs = obs/255
        next_obs = next_obs/255
        obs_e = obs_e/255
        next_obs_e = next_obs_e/255

        obs = torch.cat([obs, next_obs], dim=0)
        obs_e = torch.cat([obs_e, next_obs_e])
        rand_idx = torch.randperm(obs.shape[0])
        imgs1 = obs[rand_idx[:9]]
        imgs2 = obs[rand_idx[-9:]]
        imgs3 = obs_e[rand_idx[9:18]]
        imgs4 = obs_e[rand_idx[-18:-9]]
                
        saved_imgs = torch.cat([imgs1[:,:3,:,:], imgs2[:,:3,:,:], imgs3[:,:3,:,:], imgs4[:,:3,:,:]], dim=0)
        save_image(saved_imgs, f"./checkimages_{type}/{step}.png", nrow=9)
