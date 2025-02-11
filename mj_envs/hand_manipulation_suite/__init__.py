from gym.envs.registration import register
from mjrl.envs.mujoco_env import MujocoEnv

# Swing the door open
register(
    id='door-v0',
    entry_point='mj_envs.hand_manipulation_suite:DoorEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.door_v0 import DoorEnvV0

# Hammer a nail into the board
register(
    id='hammer-v0',
    entry_point='mj_envs.hand_manipulation_suite:HammerEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.hammer_v0 import HammerEnvV0

register(
    id='hammer_light-v0',
    entry_point='mj_envs.hand_manipulation_suite:HammerEnvV0_light',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.hammer_light import HammerEnvV0_light

register(
    id='hammer_color-v0',
    entry_point='mj_envs.hand_manipulation_suite:HammerEnvV0_color',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.hammer_color import HammerEnvV0_color

# Reposition a pen in hand
register(
    id='pen-v0',
    entry_point='mj_envs.hand_manipulation_suite:PenEnvV0',
    max_episode_steps=100,
)
from mj_envs.hand_manipulation_suite.pen_v0 import PenEnvV0

register(
    id='pen_light-v0',
    entry_point='mj_envs.hand_manipulation_suite:PenEnvV0_light',
    max_episode_steps=100,
)
from mj_envs.hand_manipulation_suite.pen_light import PenEnvV0_light

register(
    id='pen_color-v0',
    entry_point='mj_envs.hand_manipulation_suite:PenEnvV0_color',
    max_episode_steps=100,
)
from mj_envs.hand_manipulation_suite.pen_color import PenEnvV0_color

# Relcoate an object to the target
register(
    id='relocate-v0',
    entry_point='mj_envs.hand_manipulation_suite:RelocateEnvV0',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0

register(
    id='door_light-v0',
    entry_point='mj_envs.hand_manipulation_suite:DoorEnvV0_light',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.door_light import DoorEnvV0_light

register(
    id='door_color-v0',
    entry_point='mj_envs.hand_manipulation_suite:DoorEnvV0_color',
    max_episode_steps=200,
)
from mj_envs.hand_manipulation_suite.door_color import DoorEnvV0_color
