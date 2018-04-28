import gym
import roomgridworld
import numpy as np
import sys
from tqdm import tqdm

global_lookup = [
					[[1,1,1,1,2], [1,1,1,1,2], [1,1,1,1,1], [1,1,1,1,0], [1,1,1,1,0]],
					[[2,2,2,2,2], [2,2,2,2,2], [2,2,2,2,2], [2,2,2,2,2], [1,2,3,3,3]],
					[[2,2,2,2,2], [2,2,2,2,2], [2,2,2,2,2], [2,2,2,2,2], [2,2,2,2,2], [1,1,2,3,3]],
					[[2,2,2,2,2], [2,2,2,2,2], [3,3,3,3,3], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],
					[[2,2,2,2,2], [2,2,2,2,2], [3,3,3,3,3], [0,0,0,0,0]],
					[[1,1,0,3,3], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],
					[[1,0,3,3,3], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],
					[[2,2,2,2,2], [2,2,2,2,2], [2,2,2,2,2], [1,1,1,1,1], [0,0,0,0,0]]
				]
def run_intra_option(env, option, state):
	experience = []
	# print 'option executing state: {}'.format(state)
	count = 0
	done = False
	if state < 25:
		if option == 4:
			lookup = global_lookup[0]
			goal = 25
		else:
			lookup = global_lookup[1]
			goal = 102
		start, end = 0, 25
	elif state == 25:
		if option == 4:
			goal = 56
			lookup = global_lookup[2]
			action = 1
			start, end = 26, 56 
		else:
			lookup = global_lookup[1]
			goal = 102
			action = 3
			start, end = 0, 25

		while state == 25:
			count += 1
			s, r, done, _ = env.step(action)
			experience.append((state, option, r, s))
			state = s
	elif state < 56:
		if option == 4:
			lookup = global_lookup[2]
			goal = 56
		else:
			lookup = global_lookup[3]
			goal = 25
		start, end = 26, 56

	elif state == 56:
		if option == 4:
			lookup = global_lookup[4]
			goal = 77
			action = 2
			start, end = 57, 77
		else:
			lookup = global_lookup[3]
			goal = 25
			action = 0
			start, end = 26, 56

		while state == 56:
			count += 1
			s, r, done, _ = env.step(action)
			experience.append((state, option, r, s))
			state = s

	elif state < 77:
		if option == 4:
			lookup = global_lookup[4]
			goal = 77
		else:
			lookup = global_lookup[5]
			goal = 56
		start, end = 57, 77
	elif state == 77:
		if option == 4:
			lookup =global_lookup[6]
			goal = 102
			action = 3
			start, end = 78, 102
		else:
			lookup = global_lookup[5]
			goal = 56
			action = 1
			start, end = 57, 77

		while state == 77:
			count += 1
			s, r, done, _ = env.step(action)
			experience.append((state, option, r, s))
			state = s

	elif state < 103:
		if option == 4:
			lookup = global_lookup[6]
			goal = 102
		else:
			goal = 77
			lookup = global_lookup[7]
		start, end = 78, 102
	else:
		if option == 4:
			lookup = global_lookup[0]
			goal = 25
			action = 0
			start, end = 0, 25
		else:
			lookup = global_lookup[7]
			goal = 77
			action = 2
			start, end = 78, 102

		while state == 103:
			count += 1
			s, r, done, _ = env.step(action)
			experience.append((state, option, r, s))
			state = s
	# print 'goal: {}, state: {}'.format(goal, state)
	while not (done or (state == goal) or (state<start) or (state>=end)):
		count += 1
		s, r, done, _ = env.step(lookup[(state-start) // 5][(state-start) % 5])
		experience.append((state, option, r, s))
		state = s
		# input()
	
	return state, done, count, experience


def run_option(env, option, gamma, state):
	# print 'option executing state: {}'.format(state)
	count = 0
	done = False
	ret = 0
	discount = 1
	if state < 25:
		if option == 4:
			lookup = global_lookup[0]
			goal = 25
		else:
			lookup = global_lookup[1]
			goal = 102
		start, end = 0, 25
	elif state == 25:
		if option == 4:
			goal = 56
			lookup = global_lookup[2]
			action = 1
			start, end = 26, 56 
		else:
			lookup = global_lookup[1]
			goal = 102
			action = 3
			start, end = 0, 25

		while state == 25:
			count += 1
			s, r, done, _ = env.step(action)
			ret += r*discount
			discount *= gamma
			state = s
	elif state < 56:
		if option == 4:
			lookup = global_lookup[2]
			goal = 56
		else:
			lookup = global_lookup[3]
			goal = 25
		start, end = 26, 56

	elif state == 56:
		if option == 4:
			lookup = global_lookup[4]
			goal = 77
			action = 2
			start, end = 57, 77
		else:
			lookup = global_lookup[3]
			goal = 25
			action = 0
			start, end = 26, 56

		while state == 56:
			count += 1
			s, r, done, _ = env.step(action)
			ret += r*discount
			discount *= gamma
			state = s

	elif state < 77:
		if option == 4:
			lookup = global_lookup[4]
			goal = 77
		else:
			lookup = global_lookup[5]
			goal = 56
		start, end = 57, 77
	elif state == 77:
		if option == 4:
			lookup =global_lookup[6]
			goal = 102
			action = 3
			start, end = 78, 102
		else:
			lookup = global_lookup[5]
			goal = 56
			action = 1
			start, end = 57, 77

		while state == 77:
			count += 1
			s, r, done, _ = env.step(action)
			ret += r*discount
			discount *= gamma
			state = s

	elif state < 103:
		if option == 4:
			lookup = global_lookup[6]
			goal = 102
		else:
			goal = 77
			lookup = global_lookup[7]
		start, end = 78, 102
	else:
		if option == 4:
			lookup = global_lookup[0]
			goal = 25
			action = 0
			start, end = 0, 25
		else:
			lookup = global_lookup[7]
			goal = 77
			action = 2
			start, end = 78, 102

		while state == 103:
			count += 1
			s, r, done, _ = env.step(action)
			ret += r*discount
			discount *= gamma
			state = s
	# print 'goal: {}, state: {}'.format(goal, state)
	while not (done or (state == goal) or (state<start) or (state>=end)):
		count += 1
		s, r, done, _ = env.step(lookup[(state-start) // 5][(state-start) % 5])
		state = s
		ret = ret + r*discount
		discount *= gamma
		# input()
	
	return state, ret, done, discount/gamma, count

def smdp_Qlearning(max_episode=100):
	env = gym.make('RoomGridWorld-v1')

	total_actions = 6
	q_values =  np.zeros((106, total_actions))
	gamma = 0.9
	eps = 0.1
	alpha = 0.1

	for i in tqdm(xrange(max_episode)):
		state = env.reset()
		start = state
		done = False
		steps = 0
		
		while not done:
		# for i in tqdm(xrange(10000)):
			# if done:
			# 	break
			if np.random.rand() < eps:
				# exploration step
				action = np.random.randint(0, total_actions)
			else:
				# greedily choose the arm
				# randomly select if the more than 1 arms have maximum value
				max_indices = np.where(q_values[state] == np.max(q_values[state]))[0]
				action = max_indices[np.random.randint(0, max_indices.shape[0])]

			if action < 4:
				s, r, done, _ = env.step(action)
				discount = gamma
				steps += 1
			else:
				s, r, done, discount, k = run_option(env, action, gamma, state)
				steps += k

			q_values[state, action] += alpha * (r + discount * max(q_values[s]) - q_values[state, action])
			state = s
		print('Episode steps: {}, start: {}'.format(steps, start))

	for i in xrange(10):
		state = env.reset()
		start = state
		done = False
		steps = 0
		while not done:
			if np.random.rand() < eps:
				# exploration step
				action = np.random.randint(0, total_actions)
			else:
				# greedily choose the arm
				# randomly select if the more than 1 arms have maximum value
				max_indices = np.where(q_values[state] == np.max(q_values[state]))[0]
				action = max_indices[np.random.randint(0, max_indices.shape[0])]

			if action < 4:
				s, r, done, _ = env.step(action)
				discount = gamma
				steps += 1
			else:
				s, r, done, discount, ss = run_option(env, action, gamma, state)
				steps += ss
			state = s
		print('start: %d,  steps: %d' %(start, steps))


def intra_option_Qlearning(max_episode=100):
	env = gym.make('RoomGridWorld-v0')

	total_actions = 6
	q_values =  np.zeros((106, total_actions))
	gamma = 0.9
	eps = 0.1
	alpha = 0.5
	beta = 0.99

	for i in tqdm(xrange(max_episode)):
		state = env.reset()
		start = state
		done = False
		steps = 0
		
		while not done:
		# for i in tqdm(xrange(10000)):
			# if done:
			# 	break
			if np.random.rand() < eps:
				# exploration step
				action = np.random.randint(0, total_actions)
			else:
				# greedily choose the arm
				# randomly select if the more than 1 arms have maximum value
				max_indices = np.where(q_values[state] == np.max(q_values[state]))[0]
				action = max_indices[np.random.randint(0, max_indices.shape[0])]

			if action < 4:
				s, r, done, _ = env.step(action)
				experiences = [(state,action,r,s)]
				steps += 1
			else:
				s, done, k, experiences = run_intra_option(env, action, state)
				steps += k

			for x in reversed(experiences):
				q_hat = (1-beta)*q_values[x[3], x[1]] + beta*max(q_values[x[3]])
				q_values[x[0], x[1]] += alpha * (x[2] + gamma*q_hat - q_values[x[0], x[1]])
			state = s
		print('Episode steps: {}, start: {}'.format(steps, start))

	for i in xrange(10):
		eps = 0
		state = env.reset()
		start = state
		done = False
		steps = 0
		while not done:
			if np.random.rand() < eps:
				# exploration step
				action = np.random.randint(0, total_actions)
			else:
				# greedily choose the arm
				# randomly select if the more than 1 arms have maximum value
				max_indices = np.where(q_values[state] == np.max(q_values[state]))[0]
				action = max_indices[np.random.randint(0, max_indices.shape[0])]

			if action < 4:
				s, r, done, _ = env.step(action)
				discount = gamma
				steps += 1
			else:
				s, done, k, _ = run_intra_option(env, action, state)
				steps += k
			state = s
		print('start: %d,  steps: %d' %(start, steps))


if __name__ == '__main__':
	# smdp_Qlearning()
	intra_option_Qlearning()