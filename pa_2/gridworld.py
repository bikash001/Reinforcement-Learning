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

    def __init__(self, v='A', wind_prob=0.):
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 11, shape=(2,), dtype=np.int)
        self.success_prob = 0.9
        self.wind_prob = wind_prob
        self.seed()
        self.viewer = None
        self.state = None
        self.start_pos = None
        self.state_rewards = np.zeros((12,12))
        self.__fill_rewards__()
        
        if v == 'A':
            self.goal_position = (11, 11)
        elif v == 'B':
            self.goal_position = (9, 9)
        else:
            self.goal_position = (7, 5)

    def __fill_rewards__(self):
        for i in range(3,10):
            self.state_rewards[i][3] = -1
        for i in range(3, 9):
            self.state_rewards[9][i] = -1
        for i in range(5, 10):
            self.state_rewards[i][8] = -1
        for i in range(3, 6):
            self.state_rewards[i][7] = -1
        for i in range(3, 8):
            self.state_rewards[3][i] = -1

        for i in range(4,9):
            self.state_rewards[i][4] = -2
        for i in range(4, 8):
            self.state_rewards[8][i] = -2
        for i in range(6, 9):
            self.state_rewards[i][7] = -2
        for i in range(4, 7):
            self.state_rewards[i][6] = -2
        for i in range(4, 7):
            self.state_rewards[4][i] = -2

        for i in range(5,8):
            self.state_rewards[i][5] = -3
        self.state_rewards[7][6] = -3

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
        
        # move east
        if action == 0:
            x += 1
        elif action == 1:   # move west
            x -= 1
        elif action == 2: # move north
            y += 1
        else:   # move south
            y -= 1

        if self.np_random.rand() < self.wind_prob:
            x += 1

        position = (x, y)
        if not (x < 0 or x >= 12 or y < 0 or y >= 12):
            self.state = position
        else:
            position = self.state
        done = bool(position == self.goal_position)
        reward = 0
        if done:
            reward = 10

        reward += self.state_rewards[position[0], position[1]]
        # if (x == 6 and y>=6 and y<=8) or (x==7 and y==8):
        #     reward -= 3
        # elif (x>=5 and x<=7 and y>=5 and y<=9) or (x==8 and y>=7 and y<=9):
        #     reward -= 2
        # elif (x>=4 and x<=8 and y>=4 and y<=10) or (x==9 and y>=6 and y<=10):
        #     reward -= 1

        return np.array(self.state), reward, done, {}

    def reset(self):
        y = [0,1,5,6]
        self.state = (0, self.np_random.choice(y))
        self.start_pos = self.state
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 500
        screen_height = 500
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
        
            l = np.linspace(10, screen_height-10, 13)
            w = np.linspace(10, screen_width-10, 13)
            self.__del_x = w[1] - w[0]
            self.__del_y = l[1] - l[0]

            arr = []
            for i in range(len(l)-1):
                a = []
                for j in range(len(w)-1):
                    a.append([(w[j], l[i]), (w[j], l[i+1]), (w[j+1], l[i+1]), (w[j+1], l[i])])
                arr.append(a)

            # make grid
            for i in range(len(arr)):
                for j in range(len(arr[i])):
                    # check if obstacle
                    if self.state_rewards[i][j] == 0:
                        cart = rendering.make_polygon(arr[i][j], False)
                        cart.set_color(0,0,0)
                    elif self.state_rewards[i][j] == -1:
                        cart = rendering.make_polygon(arr[i][j], True)
                        cart.set_color(0.7,0.7,0.7)
                    elif self.state_rewards[i][j] == -2:
                        cart = rendering.make_polygon(arr[i][j], True)
                        cart.set_color(0.4,0.4,0.4)
                    elif self.state_rewards[i][j] == -3:
                        cart = rendering.make_polygon(arr[i][j], True)
                        cart.set_color(0,0,0)
                    self.viewer.add_geom(cart)

            #goal position
            el = arr[self.goal_position[1]][self.goal_position[0]]
            xdiff = el[3][0] - el[0][0]
            flagx = xdiff/2 + el[0][0] - 10
            
            ydiff = el[1][1] - el[0][1] - 20
            flagy1 = el[0][1] + 10
            flagy2 = flagy1 + ydiff

            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(1.0,0,0)
            self.viewer.add_geom(flag)

            # agent position
            agent = rendering.FilledPolygon(self.__margin__(arr[self.state[1]][self.state[0]], 10))
            agent.set_color(0,0.,1.)
            agent.add_attr(rendering.Transform(translation=(0, 0)))
            self.agenttrans = rendering.Transform()
            agent.add_attr(self.agenttrans)
            self.viewer.add_geom(agent)
        else:
            self.agenttrans.set_translation(self.__del_x*(self.state[0]-self.start_pos[0]), self.__del_y*(self.state[1]-self.start_pos[1]))

        # print(self.state)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def __margin__(self, els, val):
        return [(els[0][0]+val, els[0][1]+val),
                (els[1][0]+val, els[1][1]-val),
                (els[2][0]-val, els[2][1]-val),
                (els[3][0]-val, els[3][1]+val)]

    def close(self):
        if self.viewer: self.viewer.close()
