import numpy as np
import matplotlib.pyplot as plt
import sys


class TestBed(object):
	"""This class is used for simulating k-arm bandit using different action selection algorithms."""
	def __init__(self, k=10, epsilon=0.1, delta=0.1, algo='egreedy', t=0.1, c=2., means=None):
		super(TestBed, self).__init__()
		self.arms = k 							# no. of arms in the bandit
		
		# set the true values of the arms
		# if not given, initialize randomly from normal distribution
		if means is not None:
			self.arm_means = np.array(means)
		else:
			self.arm_means = np.random.normal(size=k)

		self.optimum_arm = np.argmax(self.arm_means) 		# optimal arm
		self.values = [0.] * k 			# sample mean
		self.count = np.zeros(k) 		# no. of times a particular arm pulled
		self.epsilon = epsilon 			# parameter for e-greedy algorithm and median elimination
		self.steps = 0 			# no. of steps
		self.t = t 		# temperature for softmax algo
		self.c = c 		# exploration parameter of ucb
		self.delta = delta

		self.values = np.array(self.values)

		if algo == 'egreedy':
			self.__algo = self.__epsilon_greedy
		elif algo == 'ucb':
			self.__algo = self.__ucb
		elif algo == 'softmax':
			self.__algo = self.__softmax
		elif algo == 'me':
			self.__algo = self.__median_elimination
		else:
			raise 'Invalid algo'

	# method to simulate k-arm bandit
	# pulls 1000 times
	def start_simulation(self):
		# call median elimination algorithm
		if self.__algo == self.__median_elimination:
			return self.__algo()

		rewards = [] 	# rewards received in each step
		arms = [] 		# set if optimal arm is pulled
		
		# pull arm for 1000 times
		for i in range(1000):
			reward, arm = self.__algo() 	# pull according to given method
			rewards.append(reward)

			# check if optimal arm is pulled
			if arm == self.optimum_arm:
				arms.append(1)
			else:
				arms.append(0)

		
		return np.array(rewards), np.array(arms)
	

	# e-greedy algorithm
	def __epsilon_greedy(self):
		if np.random.rand() < self.epsilon:
			# exploration step
			next_arm = np.random.randint(0, self.arms)
		else:
			# greedily choose the arm
			# randomly select if the more than 1 arms have maximum value
			max_indices = np.where(self.values == np.max(self.values))[0]
			next_arm = max_indices[np.random.randint(0, max_indices.shape[0])]

		reward = np.random.normal(self.arm_means[next_arm]) 	# get reward
		self.count[next_arm] += 1 		# increment count for selected arm
		self.values[next_arm] += (reward - self.values[next_arm])/self.count[next_arm] 		# change sample mean
		self.steps += 1 	# increment steps
		return reward, next_arm
			

	# softmax algorithm
	def __softmax(self):
		# apply softmax formula to each arm values
		prefs = np.exp(self.values/self.t) 	
		denom = np.sum(prefs)
		prefs = prefs / denom

		# pick arm using the calculated distribution
		prefs = np.cumsum(prefs)
		next_arm = 0
		prob = np.random.rand()
		for x in prefs:
			if prob < x:
				break
			next_arm += 1

		next_arm = np.min((next_arm, self.arms-1)) 	# arm should be from 0 to self.arms
		
		reward = np.random.normal(self.arm_means[next_arm]) # get reward
		self.count[next_arm] += 1
		self.values[next_arm] += (reward - self.values[next_arm])/self.count[next_arm]
		self.steps += 1
		return reward, next_arm


	# ucb-1 algorithm
	def __ucb(self):
		self.steps += 1

		# check if values of any arm is 0
		# i.e if arm is not picked for a single time
		max_indices = []
		for i, x in enumerate(self.count):
			if x == 0:
				max_indices.append(i)

		max_indices = np.array(max_indices)
		
		# if all arm are picked once then pick according to the ucb-1 rule
		if max_indices.shape[0] == 0:
			prefs = self.values + self.c * np.sqrt(np.log(self.steps)/(self.count))
			max_indices = np.where(prefs == np.max(prefs))[0]
		
		next_arm = max_indices[np.random.randint(0, max_indices.shape[0])]
		reward = np.random.normal(self.arm_means[next_arm])
		self.count[next_arm] += 1
		self.values[next_arm] += (reward - self.values[next_arm])/self.count[next_arm]
		return reward, next_arm


	# median elimination algorithm
	def __median_elimination(self):
		rewards = []
		actions = []

		max_steps = 10000 	# maximum no. of steps
		arms = [i for i in range(0, self.arms)] 	# arm nos.
		eps = self.epsilon / 4
		delta = self.delta / 2
		values = [0] * self.arms 	# initial values of each arm
		total_times = 0 	# no. of times arm pulled

		while len(arms) > 1:
			times = int(np.ceil(np.square(2/eps) * np.log(3/delta))) 	# calculate no. of times to pull an arm
			
			# pull each arm fot the above calculated no. of times
			for i, x in enumerate(arms):
				r = np.random.normal(self.arm_means[x],size=times).tolist()
				rewards += r 		# store the generated rewards
				actions += [x]*times 	# store the pulled arm
				values[i] = (total_times * values[i] + np.sum(r)) / (total_times + times) 	# update sample mean

			total_times += times
			median = np.median(values) 		# find median

			new_arms = []
			new_values = []

			# discard the arms whose values are less than median
			for i in range(len(arms)):
				if values[i] >= median:
					new_arms.append(arms[i])
					new_values.append(values[i])

			values = new_values
			arms = new_arms
			eps = 3./4. * eps
			delta = delta / 2.

			if len(rewards) >= max_steps:
				break

		# pick the only remaining arm for the remaining steps
		if len(rewards) < max_steps:
			rewards += np.random.normal(self.arm_means[arms[0]], size=(max_steps-len(rewards))).tolist()
			actions += [arms[0]]*(max_steps-len(rewards))

		return np.array(rewards[:max_steps]), np.array(actions[:max_steps])


	def next_action(self):
		return self.__algo()


# plot for part epsilon greedy algorithm
def plot1():
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


# plot for softmax algorithm
def plot2():
	temperature = [0.1, 1., 10.]
	rewards = [None] * 3
	arms = [None] * 3

	for i in range(2000):
		means = np.random.normal(size=10)

		for x, y in enumerate(temperature):
			tb = TestBed(algo='softmax', means=means, t=y)
			if rewards[x] is not None:
				avg_rewards, avg_arm = tb.start_simulation()
				rewards[x] += avg_rewards
				arms[x] += avg_arm
			else:
				rewards[x], arms[x] = tb.start_simulation()

	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	for i, x in enumerate(temperature):
		rewards[i] = rewards[i] / 2000
		arms[i] = arms[i] / 20
		ax1.plot(list(range(0, len(rewards[i]))), rewards[i], label='t = '+str(x))
		ax2.plot(list(range(0, len(arms[i]))), arms[i], label='t = '+str(x))
	
	ax1.legend()
	ax2.legend()
	ax1.set_xlabel('Steps')
	ax1.set_ylabel('Average reward')
	ax2.set_xlabel('Steps')
	ax2.set_ylabel('% Optimal action')
	ax2.set_ylim(0, 100)
	ax1.set_ylim(0)
	plt.show()


# plot for ucb-1 algorithm
def plot3():
	algos = ['ucb']
	rewards = [None] * 3
	arms = [None] * 3

	for i in range(2000):
		means = np.random.normal(size=10)

		for x, y in enumerate(algos):
			tb = TestBed(algo=y, means=means)
			if rewards[x] is not None:
				avg_rewards, avg_arm = tb.start_simulation()
				rewards[x] += avg_rewards
				arms[x] += avg_arm
			else:
				rewards[x], arms[x] = tb.start_simulation()

	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	for i, x in enumerate(algos):
		rewards[i] = rewards[i] / 2000
		arms[i] = arms[i] / 20
		ax1.plot(list(range(0, len(rewards[i]))), rewards[i], label=x)
		ax2.plot(list(range(0, len(arms[i]))), arms[i], label=x)
	
	ax1.legend()
	ax2.legend()
	ax1.set_xlabel('Steps')
	ax1.set_ylabel('Average reward')
	ax2.set_xlabel('Steps')
	ax2.set_ylabel('% Optimal action')
	ax2.set_ylim(0, 100)
	ax1.set_ylim(0)
	plt.show()	


# plot for median elimination algorithm
def plot4():
	algos = ['me']
	rewards = [None] * 3
	arms = [None] * 3

	for i in range(2000):
		means = np.random.normal(size=10)

		for x, y in enumerate(algos):
			tb = TestBed(algo=y, epsilon=0.01, delta=0.05, means=means)
			if rewards[x] is not None:
				avg_rewards, avg_arm = tb.start_simulation()
				rewards[x] += avg_rewards
				arms[x] += avg_arm
			else:
				rewards[x], arms[x] = tb.start_simulation()

	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	for i, x in enumerate(algos):
		rewards[i] = rewards[i] / 2000
		arms[i] = arms[i] / 20
		ax1.plot(list(range(0, len(rewards[i]))), rewards[i], label=x)
		ax2.plot(list(range(0, len(arms[i]))), arms[i], label=x)
	
	ax1.legend()
	ax2.legend()
	ax1.set_xlabel('Steps')
	ax1.set_ylabel('Average reward')
	ax2.set_xlabel('Steps')
	ax2.set_ylabel('% Optimal action')
	ax2.set_ylim(0, 100)
	ax1.set_ylim(0)
	plt.show()	


# plot for 1000 arm bandit
def plot5():
	algos = ['me']
	rewards = [None] * 3
	arms = [None] * 3

	for i in range(2000):
		means = np.random.normal(size=1000)

		for x, y in enumerate(algos):
			tb = TestBed(k=1000, algo=y, epsilon=0.1, delta=0.05, means=means)
			if rewards[x] is not None:
				avg_rewards, avg_arm = tb.start_simulation()
				rewards[x] += avg_rewards
				arms[x] += avg_arm
			else:
				rewards[x], arms[x] = tb.start_simulation()

	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	for i, x in enumerate(algos):
		rewards[i] = rewards[i] / 2000
		arms[i] = arms[i] / 20
		ax1.plot(list(range(0, len(rewards[i]))), rewards[i], label=x)
		ax2.plot(list(range(0, len(arms[i]))), arms[i], label=x)
	
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
	np.random.seed(0)
	plot4()