from gym import Env
from gym.envs.registration import register
from gym.utils import seeding
from gym import spaces
import numpy as np


class vishamC(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __calculate_reward__(self, pos):
        gamma = 10.
        return 0.5*pos[0]*pos[0] + gamma*0.5*pos[1]*pos[1]
    
    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        # if abs(action[0]) > 0.025:
        #     action[0] = 0.025 * np.sign(action[0])
        # if abs(action[1]) > 0.025:
        #     action[1] = 0.025 * np.sign(action[1])
        norm = np.linalg.norm(action)
        if norm > 0.025:
            action[0] = action[0]/norm * 0.025
            action[1] = action[1]/norm * 0.025
        
        self.state = np.array([self.state[0]+action[0], self.state[1]+action[1]])
        if self.state[0]<-1 or self.state[0]>1 or self.state[1]<-1 or self.state[1]>1:
            self.reset()

        reward = -self.__calculate_reward__(self.state)
        done = bool(self.state[0] == 0. and self.state[1] == 0.)
        # Return the next state and the reward, along with 2 additional quantities : False, {}
        return np.array(self.state), reward, done, {}

    def reset(self):
        while True:
            self.state = self.np_random.uniform(low=-1, high=1, size=(2,))
            # Sample states that are far away
            if np.linalg.norm(self.state) > 0.9:
                break
        return np.array(self.state)

    # method for rendering

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 600

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # o = rendering.make_polyline([(0,0), (0,screen_height), (screen_width,screen_height), (screen_width,0)])
            # o.set_color(0,0,0)
            # self.viewer.add_geom(o)
            # r = np.linspace(0, 1, 20)
            # for x in r:
            #     o = rendering.make_circle(min(screen_height, screen_width) * x, filled=False)
            #     o.add_attr(rendering.Transform(translation=(screen_width/2, screen_height/2)))
            #     o.set_color(0,0,0)
            #     self.viewer.add_geom(o)

           
            agent = rendering.make_circle(
                min(screen_height, screen_width) * 0.03)
            origin = rendering.make_circle(
                min(screen_height, screen_width) * 0.03)
            trans = rendering.Transform(translation=(0, 0))
            agent.add_attr(trans)
            self.trans = trans
            agent.set_color(1, 0, 0)
            origin.set_color(0, 0, 0)
            origin.add_attr(rendering.Transform(
                translation=(screen_width // 2, screen_height // 2)))
            self.viewer.add_geom(agent)
            self.viewer.add_geom(origin)

        # self.trans.set_translation(0, 0)
        self.trans.set_translation(
            (self.state[0] + 1) / 2 * screen_width,
            (self.state[1] + 1) / 2 * screen_height,
        )

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()

# register(
#     'vishamC-v0',
#     entry_point='rlpa.vishamC:vishamC',
#     timestep_limit=40,
# )
