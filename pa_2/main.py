import gym
import time
import numpy as np
import rlpa
import sys
from gym.envs.classic_control import rendering
import matplotlib.pyplot as plt
from tqdm import tqdm
    

def view(q, goal_position):
    state_rewards = np.zeros((12,12))
    for i in range(3,10):
        state_rewards[i][3] = -1
    for i in range(3, 9):
        state_rewards[9][i] = -1
    for i in range(5, 10):
        state_rewards[i][8] = -1
    for i in range(3, 6):
        state_rewards[i][7] = -1
    for i in range(3, 8):
        state_rewards[3][i] = -1

    for i in range(4,9):
        state_rewards[i][4] = -2
    for i in range(4, 8):
        state_rewards[8][i] = -2
    for i in range(6, 9):
        state_rewards[i][7] = -2
    for i in range(4, 7):
        state_rewards[i][6] = -2
    for i in range(4, 7):
        state_rewards[4][i] = -2

    for i in range(5,8):
        state_rewards[i][5] = -3
    state_rewards[7][6] = -3


    screen_width = 500
    screen_height = 500
    
    viewer = rendering.Viewer(screen_width, screen_height)

    l = np.linspace(10, screen_height-10, 13)
    w = np.linspace(10, screen_width-10, 13)
    dx = w[1] - w[0]
    dy = l[1] - l[0]

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
            if state_rewards[i][j] == 0:
                cart = rendering.make_polygon(arr[i][j], False)
                cart.set_color(0,0,0)
            elif state_rewards[i][j] == -1:
                cart = rendering.make_polygon(arr[i][j], True)
                cart.set_color(0.7,0.7,0.7)
            elif state_rewards[i][j] == -2:
                cart = rendering.make_polygon(arr[i][j], True)
                cart.set_color(0.4,0.4,0.4)
            elif state_rewards[i][j] == -3:
                cart = rendering.make_polygon(arr[i][j], True)
                cart.set_color(0,0,0)
            viewer.add_geom(cart)

    #goal position
    el = arr[goal_position[1]][goal_position[0]]
    xdiff = el[3][0] - el[0][0]
    flagx = xdiff/2 + el[0][0] - 10
    
    ydiff = el[1][1] - el[0][1] - 20
    flagy1 = el[0][1] + 10
    flagy2 = flagy1 + ydiff

    flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
    viewer.add_geom(flagpole)
    flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
    flag.set_color(1.0,0,0)
    viewer.add_geom(flag)

    for i in range(q.shape[0]):
        if i != goal_position[0]*12+goal_position[1]:
            x, y = arr[i%12][i//12][0]
            dn = np.argmax(q[i])
            if dn == 0:
                obj = rendering.Line((x, y + dy/2.), (x+dx, y + dy/2.))
                obj.set_color(0,0,1.)
                viewer.add_geom(obj)
                obj = rendering.FilledPolygon([(x+dx/2, y+dy-10), (x+dx, y+dy/2.), (x+dx/2, y+10)])
                obj.set_color(0,0,1.)
                viewer.add_geom(obj)
            elif dn == 1:
                obj = rendering.Line((x, y + dy/2.), (x+dx, y + dy/2.))
                obj.set_color(0,0,1.)
                viewer.add_geom(obj)
                obj = rendering.FilledPolygon([(x, y+dy/2.), (x+dx/2., y+dy-10), (x+dx/2., y+10)])
                obj.set_color(0,0,1.)
                viewer.add_geom(obj)
            elif dn == 2:
                obj = rendering.Line((x+dx/2., y), (x+dx/2., y+dy))
                obj.set_color(0,0,1.)
                viewer.add_geom(obj)
                obj = rendering.FilledPolygon([(x+10, y+dy/2.), (x+dx/2., y+dy), (x+dx-10, y+dy/2.)])
                obj.set_color(0,0,1.)
                viewer.add_geom(obj)
            else:
                obj = rendering.Line((x+dx/2., y), (x+dx/2., y+dy))
                obj.set_color(0,0,1.)
                viewer.add_geom(obj)
                obj = rendering.FilledPolygon([(x+10, y+dy/2.), (x+dx/2., y), (x+dx-10, y+dy/2.)])
                obj.set_color(0,0,1.)
                viewer.add_geom(obj)


    viewer.render(return_rgb_array = False)
    return viewer


def q_learning(model='A', max_episode=None):
    batch_size = 50
    alpha = 0.1
    alpha_k = 0.999
    eps = 0.1
    eps_k = 0.9999
    gamma = 0.9
    q = np.zeros((12*12,4))
    env = gym.make('gridworld'+model+'-v0')

    ep_no = 0   
    rewards = []
    steps = []
    while True:
        ep_no += 1
        prev_q = np.array(q)
        state = env.reset()
        t = 0
        total_r = 0.
        while True:
            t += 1
            if np.random.rand() < (1 - eps):
                action = np.argmax(q[state[0]*12+state[1]])
            else:
                action = np.random.randint(4)
            s, r, done,_ = env.step(action)
            q[state[0]*12+state[1], action] += alpha * (r+gamma*np.max(q[s[0]*12+s[1]])-q[state[0]*12+state[1], action])
            state = s
            total_r += r
            if done:
                # print("Episode finished after {} timesteps with rewards {}".format(t+1, total_r))
                rewards.append(total_r)
                steps.append(t)
                break
        
        if max_episode is not None:
            if ep_no >= max_episode:
                break
        elif ep_no % batch_size == 0:
            alpha = alpha * alpha_k
            eps = eps * eps_k

        if max_episode is None and (np.linalg.norm(prev_q-q) <= 1e-8):
            break

    return rewards, steps
    # print('alpha %f, eps %f' %(alpha, eps))
    # np.save('q_learning', q)
    # input('enter no.')
    # if model == 'A':
    #     v = view(q, (11, 11))
    # elif model == 'B':
    #     v = view(q, (9, 9))
    # else:
    #     v = view(q, (7, 5))
    # input('enter to exit')
    # v.close()
    # for _ in range(2):
    #     state = env.reset()
    #     env.render()
    #     input()
    #     t = 0
    #     rewards = []
    #     while True:
    #         time.sleep(1)
    #         action = np.argmax(q[state[0]*12+state[1]])
    #         s, r, done,_ = env.step(action)
    #         state = s
    #         rewards.append(r)
    #         env.render()
            
    #         if done:
    #             time.sleep(0.1)
    #             print("Episode finished after {} timesteps with rewards {}".format(t+1, np.sum(rewards)))
    #             break
    #         t += 1
    # env.close()

def sarsa(model='A', max_episode=500):
    alpha = 0.1
    eps = 0.1
    gamma = 0.9
    q = np.zeros((12*12,4))
    env = gym.make('gridworld'+model+'-v0')
    rewards, steps = [], []
    for i_episode in range(max_episode):
        state = env.reset()
        t = 0
        if np.random.rand() < (1 - eps):
            action = np.argmax(q[state[0]*12+state[1]])
        else:
            action = np.random.randint(4)
        total_r = 0.
        while True:
            t += 1
            s, r, done,_ = env.step(action)
            if np.random.rand() < (1 - eps):
                a = np.argmax(q[s[0]*12+s[1]])
            else:
                a = np.random.randint(4)
            total_r += r
            q[state[0]*12+state[1], action] += alpha * (r+gamma*q[s[0]*12+s[1], a]-q[state[0]*12+state[1], action])
            state, action = s, a

            if done:
                rewards.append(total_r)
                steps.append(t)
                # print("Episode finished after {} timesteps".format(t+1))
                break
    
    # return rewards, steps            
    print('alpha %f, eps %f' %(alpha, eps))
    np.save('q_learning', q)
    input('enter no.')
    if model == 'A':
        v = view(q, (11, 11))
    elif model == 'B':
        v = view(q, (9, 9))
    else:
        v = view(q, (7, 5))
    input('enter to exit')
    v.close()
    
    # np.savetxt('probs.csv', q, delimiter=',')
    # return
    # for i in range(10):
    #     state = env.reset()
    #     env.render()
    #     t = 0
    #     rewards = []
    #     while True:
    #         time.sleep(0.1)
    #         action = np.argmax(q[state[0]*12+state[1]])
    #         s, r, done, debug = env.step(action)
    #         state = s
    #         rewards.append(r)
    #         env.render()

    #         if done:
    #             time.sleep(0.1)
    #             print("Episode finished after {} timesteps with rewards {}".format(t+1, np.sum(rewards)))
    #             break
    #         t += 1
    # env.close()

def sarsa_lambda(lmd, model='A', max_episode=500):
    alpha = 0.1
    eps = 0.1
    gamma = 0.9
    q = np.zeros((12*12,4))
    et = np.zeros((12*12, 4))
    env = gym.make('gridworld'+model+'-v0')

    rewards, steps = [], []
    for i_episode in range(max_episode):
        state = env.reset()
        t = 0
        if np.random.rand() < (1 - eps):
            action = np.argmax(q[state[0]*12+state[1]])
        else:
            action = np.random.randint(4)
        total_r = 0.
        while True:
            t += 1
            s, r, done,_ = env.step(action)
            if np.random.rand() < (1 - eps):
                a = np.argmax(q[s[0]*12+s[1]])
            else:
                a = np.random.randint(4)

            total_r += r
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
                rewards.append(total_r)
                steps.append(t)
                break
            
    # return rewards, steps
    # print('alpha %f, eps %f' %(alpha, eps))
    # np.save('sarsa_lambda', q)
    input('enter no.')
    if model == 'A':
        v = view(q, (11, 11))
    elif model == 'B':
        v = view(q, (9, 9))
    else:
        v = view(q, (7, 5))
    input('enter to exit')
    v.close()
    
    # np.savetxt('probs.csv', q, delimiter=',')
    # return
    # for i in range(10):
    #     state = env.reset()
    #     # print('start: ', state)
    #     env.render()
    #     t = 0
    #     rewards = []
    #     while True:
    #         time.sleep(0.1)
    #         action = np.argmax(q[state[0]*12+state[1]])
    #         s, r, done, debug = env.step(action)
    #         state = s
    #         rewards.append(r)
    #         env.render()
    #         # print(debug)

    #         if done:
    #             time.sleep(0.1)
    #             print("Episode finished after {} timesteps with reward {}".format(t+1, np.sum(rewards)))
    #             break
    #         t += 1
    # env.close()


def plot1(algo):
    if algo == 'Q-learning':
        model = q_learning
    elif algo == 'Sarsa':
        model = sarsa
    else:
        raise NotImplementedError()

    for goal in ['C']:
        max_episode = 500
        rewards, steps = [], []
        for i in tqdm(range(10)):
            r, s = model(goal, max_episode)
            rewards.append(r)
            steps.append(s)

        r_y = np.mean(np.array(rewards), 0)
        s_y = np.mean(np.array(steps), 0)
        plt.title(algo)
        plt.xlabel('episodes')
        plt.ylabel('average rewards')
        plt.plot(np.linspace(1, max_episode, max_episode),r_y)
        plt.savefig(algo+'_'+goal+'_rewardss.png')
        plt.clf()

        plt.title(algo)
        plt.xlabel('episodes')
        plt.ylabel('average steps')
        plt.plot(np.linspace(1, max_episode, max_episode),s_y)
        plt.savefig(algo+'_'+goal+'_steps.png')
        plt.clf()

def plot2():
    for goal in ['A', 'B']:
        for lmd in [0, 0.3, 0.5, 0.9, 0.99, 1.0]:
            max_episode = 500
            rewards, steps = [], []
            for i in tqdm(range(50)):
                r, s = sarsa_lambda(lmd, goal, max_episode)
                rewards.append(r)
                steps.append(s)

            r_y = np.mean(np.array(rewards), 0)
            s_y = np.mean(np.array(steps), 0)
            plt.title('Sarsa(λ)')
            plt.xlabel('episodes')
            plt.ylabel('average rewards')
            plt.plot(np.linspace(1, max_episode, max_episode),r_y)
            plt.savefig('sarsa-'+str(lmd)+'-'+goal+'-rewards.png')
            plt.clf()

            plt.title('Sarsa(λ)')
            plt.xlabel('episodes')
            plt.ylabel('average steps')
            plt.plot(np.linspace(1, max_episode, max_episode),s_y)
            plt.savefig('sarsa-'+str(lmd)+'-'+goal+'_steps.png')
            plt.clf()

def plot3():
    l_values = np.linspace(0,1, 21)
    for goal in ['A', 'B']:
        rewards, steps = [], []
        for lmd in l_values:
            max_episode = 26
            rs, ss = [], []
            for i in tqdm(range(100)):
                r, s = sarsa_lambda(lmd, goal, max_episode)
                rs.append(r[-1])
                ss.append(s[-1])
            rewards.append(np.mean(rs))
            steps.append(np.mean(ss))

        r_y = rewards
        s_y = steps
        plt.title('Sarsa(λ)')
        plt.xlabel('λ')
        plt.ylabel('average rewards')
        plt.plot(l_values,r_y)
        plt.savefig('ss-'+goal+'-rewards.png')
        plt.clf()

        plt.title('Sarsa(λ)')
        plt.xlabel('episodes')
        plt.ylabel('average steps')
        plt.plot(l_values,s_y)
        plt.savefig('ss-'+goal+'_steps.png')
        plt.clf()

if __name__ == '__main__':
    # plot3()
    # plot1('Sarsa')
    plot1('Q-learning')
    # plot2()
    # q_learning('C', 50000)
    # sarsa('C', 10000)
    # sarsa_lambda(lmd=0., model='C', max_episode=1000)
    # v = view(None)
    # input()
    # v.close()

    # q_learning(max_episode=500)
    # sarsa()
    # sarsa_lambda(0.3)
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