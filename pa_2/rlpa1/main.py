import gym
import time
import numpy as np
from gridworld import GridWorldEnv
import sys
import warnings

# warnings.filterwarnings('error')

# np.seterr(all='raise')

def q_learning():
    alpha = 0.1
    eps = 0.1
    gamma = 0.9
    q = np.zeros((12*12,4))
    env = GridWorldEnv('A')

    for i_episode in range(200):
        state = env.reset()
        t = 0
        while True:
            # env.render()
            if np.random.rand() < (1 - eps):
                action = np.argmax(q[state[0]*12+state[1]])
            else:
                action = np.random.randint(4)
            s, r, done,_ = env.step(action)
            q[state[0]*12+state[1], action] += alpha * (r+gamma*np.max(q[s[0]*12+s[1]])-q[state[0]*12+state[1], action])
            state = s

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
            t += 1

    for _ in range(2):
        state = env.reset()
        env.render()
        t = 0
        while True:
            time.sleep(1)
            action = np.argmax(q[state[0]*12+state[1]])
            s, r, done,_ = env.step(action)
            state = s
            env.render()
            
            if done:
                time.sleep(1)
                print("Episode finished after {} timesteps".format(t+1))
                break
            t += 1
    env.close()

def sarsa():
    alpha = 0.1
    eps = 0.1
    gamma = 0.9
    q = np.zeros((12*12,4))
    env = GridWorldEnv('A')

    for i_episode in range(300):
        state = env.reset()
        t = 0
        if np.random.rand() < (1 - eps):
            action = np.argmax(q[state[0]*12+state[1]])
        else:
            action = np.random.randint(4)
        
        while True:
            # env.render()
            s, r, done,_ = env.step(action)
            if np.random.rand() < (1 - eps):
                a = np.argmax(q[s[0]*12+s[1]])
            else:
                a = np.random.randint(4)

            q[state[0]*12+state[1], action] += alpha * (r+gamma*q[s[0]*12+s[1], a]-q[state[0]*12+state[1], action])
            state, action = s, a

            if done:
                # print("Episode finished after {} timesteps".format(t+1))
                break
            t += 1

    # np.savetxt('probs.csv', q, delimiter=',')
    # return
    for i in range(10):
        state = env.reset()
        print('start: ', state)
        env.render()
        t = 0
        while True:
            time.sleep(0.1)
            action = np.argmax(q[state[0]*12+state[1]])
            s, r, done, debug = env.step(action)
            state = s
            env.render()
            print(debug)

            if done:
                time.sleep(0.1)
                print("Episode finished after {} timesteps".format(t+1))
                break
            t += 1
    env.close()

def sarsa_lambda(lmd):
    alpha = 0.1
    eps = 0.1
    gamma = 0.9
    q = np.zeros((12*12,4))
    et = np.zeros((12*12, 4))
    env = GridWorldEnv('A')

    for i_episode in range(100):
        state = env.reset()
        t = 0
        if np.random.rand() < (1 - eps):
            action = np.argmax(q[state[0]*12+state[1]])
        else:
            action = np.random.randint(4)
        
        while True:
            # env.render()
            s, r, done,_ = env.step(action)
            if np.random.rand() < (1 - eps):
                a = np.argmax(q[s[0]*12+s[1]])
            else:
                a = np.random.randint(4)

            # modify eligibility
            et = (gamma*lmd)*et
            for i in range(et.shape[1]):
                et[state[0]*12+state[1]][i] = 0.    
            et[state[0]*12+state[1]][action] = 1.

            # print('max: %e, min: %e' %(np.max(q),np.min(q)))
            delta = r+gamma*q[s[0]*12+s[1], a]-q[state[0]*12+state[1], action]
            q += (alpha*delta)*et

            state, action = s, a

            if done:
                # print("Episode finished after {} timesteps".format(t+1))
                break
            t += 1

    # np.savetxt('probs.csv', q, delimiter=',')
    # return
    for i in range(10):
        state = env.reset()
        # print('start: ', state)
        env.render()
        t = 0
        while True:
            time.sleep(0.1)
            action = np.argmax(q[state[0]*12+state[1]])
            s, r, done, debug = env.step(action)
            state = s
            env.render()
            # print(debug)

            if done:
                time.sleep(0.1)
                print("Episode finished after {} timesteps".format(t+1))
                break
            t += 1
    env.close()

if __name__ == '__main__':
    # q_learning()
    sarsa_lambda(0.3)
    # env = GridWorldEnv('C')
    # for i_episode in range(20):
    #     observation = env.reset()
    #     for t in range(100):
    #         env.render()
    #         # print(observation)
    #         action = env.action_space.sample()
    #         observation, reward, done, info = env.step(action)
    #         if done:
    #             print("Episode finished after {} timesteps".format(t+1))
    #             break
    # env.close()