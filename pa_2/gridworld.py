import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class GridWorldEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(1, 12, shape=(2,), dtype=np.int)
        self.success_prob = 0.9
        self.wind_prob = 0.5
        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        position = self.state
        actions = [0, 1, 2, 3]

        x, y = position
        
        if not self.np_random.rand() < self.success_prob:    
            actions.remove(action)
            action = self.np_random.choice(actions)
        
        if action == 0:
            x += 1
        elif action == 1:
            x -= 1
        elif action == 2:
            y += 1
        else:
            y -= 1

        if self.np_random.rand() < self.wind_prob:
            x += 1

        position = (x, y)
        if not (x < 0 or x > 12 or y < 0 or y > 12):
            self.state = position
        
        done = bool(position == self.goal_position)
        reward = 0
        if done:
            reward = 10

        if (x == 6 and y>=6 and y<=8) or (x==7 and y==8):
            reward -= 3
        elif (x>=5 and x<=7 and y>=5 and y<=9) or (x==8 and y>=7 and y<=9):
            reward -= 2
        elif (x>=4 and x<=8 and y>=4 and y<=10) or (x==9 and y>=6 and y<=10):
            reward -= 1

        return np.array(self.state), reward, done, {}

    def reset(self):
        y = [1,2,6,7]
        self.state = (1, self.np_random.choice(y))
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        world_width = 12
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
