import numpy as np
import matplotlib.pyplot as plt

class TestBed(object):
	"""docstring for TestBed"""
	def __init__(self, k=10, epsilon=0., algo='egreedy', init_val=None):
		super(TestBed, self).__init__()
		self.arms = k
		self.arm_means = np.random.normal(size=k)
		self.values = []
		self.count = np.ones(k)
		self.epsilon = epsilon
		self.steps = k

		if init_val is None:
			# pull each arm once
			for i in range(k):
				self.values.append(np.random.normal(self.arm_means[i]))
		else:
			self.values = [init_val] * k
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
		for i in range(1000):
			reward = self.__algo()
			rewards.append(reward)

		total = rewards[0]
		for i in range(1, len(rewards)+1):
			total += rewards[i-1]
			rewards[i-1] = total / i

		return np.array(rewards)
	

	def __epsilon_greedy(self):
		# exploration step
		if np.random.randn() < self.epsilon:
			next_arm = np.random.randint(0, self.arms)
		else:
			min_indices = np.where(self.values == np.max(self.values))[0]
			next_arm = min_indices[np.random.randint(0, min_indices.shape[0])]

		reward = np.random.normal(self.arm_means[next_arm])
		self.values[next_arm] += (reward - self.values[next_arm])/self.count[next_arm]
		self.count[next_arm] += 1
		self.steps += 1
		return reward
			

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
		return reward

	def __ucb(self):
		min_indices = np.where(self.values == (self.values + self.c * np.sqrt(np.log(self.steps)/self.count)))[0]
		next_arm = min_indices[np.random.randint(0, min_indices.shape[0])]
		reward = np.random.normal(self.arm_means[next_arm])
		self.values[next_arm] += (reward - self.values[next_arm])/self.count[next_arm]
		self.count[next_arm] += 1
		self.steps += 1
		return reward

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
		pass


if __name__ == '__main__':
	for eps in [0., 0.1, 0.01]:
		rewards = None
		for i in range(2000):
			tb = TestBed(epsilon=eps)
			if rewards is not None:
				avg_rewards = tb.start_simulation()
				rewards += avg_rewards
			else:
				rewards = tb.start_simulation()

		rewards = rewards / 2000
		plt.plot(list(range(0, len(rewards))), rewards, label='eps = '+str(eps))
	
	plt.legend()
	plt.show()