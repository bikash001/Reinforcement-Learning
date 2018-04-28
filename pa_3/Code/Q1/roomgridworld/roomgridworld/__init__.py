import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='RoomGridWorld-v0',
    entry_point='roomgridworld.envs:RoomGridWorldA'
)
register(
    id='RoomGridWorld-v1',
    entry_point='roomgridworld.envs:RoomGridWorldB'
)
