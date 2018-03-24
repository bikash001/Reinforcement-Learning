import numpy as np
import gym
from chakra import chakra as ChakraEnv
import time

if __name__ == '__main__':
    env = ChakraEnv()
    for i_episode in range(1):
        observation = env.reset()
        for t in range(40):
            env.render()
            time.sleep(2)
            # print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()