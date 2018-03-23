import gym
import time
import numpy as np
from gridworld import GridWorldEnv

if __name__ == '__main__':
    env = GridWorldEnv('C')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            # print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()