import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='chakra-v0',
    entry_point='rlpa.envs:chakra',
    timestep_limit=40,
)

register(
    id='visham-v0',
    entry_point='rlpa.envs:visham',
    timestep_limit=40,
)

register(
    id='gridworldA-v0',
    entry_point='rlpa.envs:GridWorldEnvA',
)


register(
    id='gridworldB-v0',
    entry_point='rlpa.envs:GridWorldEnvB',
)


register(
    id='gridworldC-v0',
    entry_point='rlpa.envs:GridWorldEnvC',
)