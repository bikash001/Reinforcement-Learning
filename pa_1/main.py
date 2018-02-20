import numpy as np
import matplotlib.pyplot as plt

class TestBed(object):
	"""docstring for TestBed"""
	def __init__(self, k=10, epsilon=0., algo='egreedy', init_val=None, means=None):
		super(TestBed, self).__init__()
		self.arms = k
		if means is not None:
			self.arm_means = np.array(means)
		else:
			self.arm_means = np.random.normal(size=k)
		self.optimum_arm = np.argmax(self.arm_means)
		self.values = [0.] * k
		self.count = np.zeros(k)
		self.epsilon = epsilon
		self.steps = 0

		# if init_val is None:
		# 	# pull each arm once
		# 	for i in range(k):
		# 		self.values.append(np.random.normal(self.arm_means[i]))
		# else:
		# 	self.values = [init_val] * k
		self.values = np.array(self.values)

		if algo == 'egreedy':
			self.__algo = self.__epsilon_greedy
		elif algo == 'ucb':
			self.__algo = self.__ucb
		elif algo == 'softmax':
			self.__algo = self.__softmax
		else:
			raise 'Invalid algo'

	def start_simulation(self):
		rewards = []
		arms = []
		for i in range(1000):
			reward, arm = self.__algo()
			rewards.append(reward)
			arms.append(arm)

		total = 0.
		total_selection = 0.
		for i in range(1, len(rewards)+1):
			total += rewards[i-1]
			rewards[i-1] = total / i
			total_selection += 1. if arms[i-1] == self.optimum_arm else 0.
			arms[i-1] = total_selection / i

		return np.array(rewards), np.array(arms)
	

	def __epsilon_greedy(self):
		# exploration step
		if np.random.rand() < self.epsilon:
			next_arm = np.random.randint(0, self.arms)
		else:
			max_indices = np.where(self.values == np.max(self.values))[0]
			next_arm = max_indices[np.random.randint(0, max_indices.shape[0])]

		reward = np.random.normal(self.arm_means[next_arm])
		self.count[next_arm] += 1
		self.values[next_arm] += (reward - self.values[next_arm])/self.count[next_arm]
		self.steps += 1
		return reward, next_arm
			

	def __softmax(self):
		self.steps += 1
		denom = np.sum(np.exp(self.prefs))
		for i in range(self.prefs.shape[0]):
			self.probs[i] = np.exp(self.prefs[i])/denom

		min_indices = np.where(self.prefs == np.max(self.prefs))[0]
		next_arm = min_indices[np.random.randint(0, min_indices.shape[0])]
		reward = np.random.normal(self.arm_means[next_arm])
		self.avg_reward +=  (reward - self.avg_reward)/self.steps

		for i in range(0, next_arm):
			self.prefs[i] -= (self.alpha * (reward - self.avg_reward)) * self.probs[i]

		self.prefs[next_arm] += self.alpha * (reward - self.avg_reward) * (1 - self.probs[next_reward])

		for i in range(next_arm+1, self.arms):
			self.prefs[i] -= (self.alpha * (reward - self.avg_reward)) * self.probs[i]
		return reward, next_arm

	def __ucb(self):
		min_indices = np.where(self.values == (self.values + self.c * np.sqrt(np.log(self.steps)/self.count)))[0]
		next_arm = min_indices[np.random.randint(0, min_indices.shape[0])]
		reward = np.random.normal(self.arm_means[next_arm])
		self.values[next_arm] += (reward - self.values[next_arm])/self.count[next_arm]
		self.count[next_arm] += 1
		self.steps += 1
		return reward, next_arm

	def __median_elimination(self):
		arms = np.array([i for i in range(0, self.arms)])
		eps = self.epsilon
		delta = self.delta

		while arms.shape[0] > 1:
			times = np.square(2/eps) * np.log(3/delta)
			means = []
			for i in arms:
				means.append(np.mean(np.random.normal(self.means[i],size=times)))

			# values = 

	def next_action(self):
		return self.__algo()


def plot():
	eps = [0., 0.01, 0.1]
	rewards = [None] * 3
	arms = [None] * 3

	for i in range(2000):
		means = np.random.normal(size=10)

		for x, y in enumerate(eps):
			tb = TestBed(epsilon=y, means=means)
			if rewards[x] is not None:
				avg_rewards, avg_arm = tb.start_simulation()
				rewards[x] += avg_rewards
				arms[x] += avg_arm
			else:
				rewards[x], arms[x] = tb.start_simulation()

	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	for i, x in enumerate(eps):
		rewards[i] = rewards[i] / 2000
		arms[i] = arms[i] / 20
		ax1.plot(list(range(0, len(rewards[i]))), rewards[i], label='eps = '+str(x))
		ax2.plot(list(range(0, len(arms[i]))), arms[i], label='eps = '+str(x))
	
	ax1.legend()
	ax2.legend()
	ax1.set_xlabel('Steps')
	ax1.set_ylabel('Average reward')
	ax2.set_xlabel('Steps')
	ax2.set_ylabel('% Optimal action')
	ax2.set_ylim(0, 100)
	ax1.set_ylim(0)
	plt.show()



if __name__ == '__main__':
	plot()