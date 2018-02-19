import numpy as np

class TestBed(object):
	"""docstring for TestBed"""
	def __init__(self, k=10, epsilon=0., init_val=None):
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
			next_arm = np.argmax(self.values)

		reward = np.random.normal(self.arm_means[next_arm])
		self.values[next_arm] += (reward - self.values[next_arm])/self.count[next_arm]
		self.count[next_arm] += 1
		self.steps += 1

	def __softmax(self):
		pass

	def __ucb(self):
		next_arm = np.argmax(self.values + self.c * np.sqrt(np.log(self.steps)/self.count))
		reward = np.random.normal(self.arm_means[next_arm])
		self.values[next_arm] += (reward - self.values[next_arm])/self.count[next_arm]
		self.count[next_arm] += 1
		self.steps += 1


	def __median_elimination(self):
		pass

	def next_action(self):
		pass