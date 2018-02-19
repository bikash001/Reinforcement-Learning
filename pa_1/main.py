import numpy as np

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

	def __softmax(self):
		denom = np.sum(np.exp(self.prefs))
		for i in range(self.prefs.shape[0]):
			self.probs[i] = np.exp(self.prefs[i])/denom

		min_indices = np.where(self.prefs == np.max(self.prefs))[0]
		next_arm = min_indices[np.random.randint(0, min_indices.shape[0])]
		reward = np.random.normal(self.arm_means[next_arm])

		for i in range(0, next_arm):
			self.prefs[i] -= (self.alpha * (reward - self.avg_reward)) * self.probs[i]

		self.prefs[next_arm] += self.alpha * (reward - self.avg_reward) * (1 - self.probs[next_reward])

		for i in range(next_arm, self.arms):
			self.prefs[i] -= (self.alpha * (reward - self.avg_reward)) * self.probs[i]

	def __ucb(self):
		min_indices = np.where(self.values == (self.values + self.c * np.sqrt(np.log(self.steps)/self.count)))[0]
		next_arm = min_indices[np.random.randint(0, min_indices.shape[0])]
		reward = np.random.normal(self.arm_means[next_arm])
		self.values[next_arm] += (reward - self.values[next_arm])/self.count[next_arm]
		self.count[next_arm] += 1
		self.steps += 1


	def __median_elimination(self):
		pass

	def next_action(self):
		pass