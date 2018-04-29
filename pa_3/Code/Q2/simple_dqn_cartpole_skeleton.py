import gym
import random
import os, sys
import time

import numpy as np
import tensorflow as tf

class DQN:
	
	REPLAY_MEMORY_SIZE = 10000 			# number of tuples in experience replay  
	EPSILON = 0.5 						# epsilon of epsilon-greedy exploation
	EPSILON_DECAY = 0.99 				# exponential decay multiplier for epsilon
	HIDDEN1_SIZE = 128 					# size of hidden layer 1
	HIDDEN2_SIZE = 128 					# size of hidden layer 2
	EPISODES_NUM = 2000 				# number of episodes to train on. Ideally shouldn't take longer than 2000
	MAX_STEPS = 200 					# maximum number of steps in an episode 
	LEARNING_RATE = 0.0001 				# learning rate and other parameters for SGD/RMSProp/Adam
	MINIBATCH_SIZE = 10 				# size of minibatch sampled from the experience replay
	DISCOUNT_FACTOR = 0.9 				# MDP's gamma
	TARGET_UPDATE_FREQ = 100 			# number of steps (not episodes) after which to update the target networks 
	LOG_DIR = './logs' 					# directory wherein logging takes place


	# Create and initialize the environment
	def __init__(self, env, hidden, batch_size=10, lr=0.0001, eps=0.5, episode=2000, model_dir='model/'):
		self.env = gym.make(env)
		assert len(self.env.observation_space.shape) == 1
		self.input_size = self.env.observation_space.shape[0]		# In case of cartpole, 4 state features
		self.output_size = self.env.action_space.n					# In case of cartpole, 2 actions (right/left)
		if not os.path.exists(model_dir):
			os.makedirs(model_dir)

		self.model_dir = model_dir
		self.hidden = [self.input_size] + hidden + [self.output_size]
		# print self.hidden

		self.epsilon = eps
		self.batch_size = batch_size
		self.gamma = self.DISCOUNT_FACTOR
		self.epsilon_decay = self.EPSILON_DECAY
		self.lr = lr
		self.lmd = 0.0001
		# self.max_episode = self.EPISODES_NUM
		self.max_episode = episode

	def _build_graph(self, target):
		# placeholder for the state-space input to the q-network
		x = tf.placeholder(tf.float32, [None, self.input_size], name='input_placeholder')
		############################################################
		# Design your q-network here.
		# 
		# Add hidden layers and the output layer. For instance:
		# 
		# with tf.name_scope('output'):
		#	W_n = tf.Variable(
		# 			 tf.truncated_normal([self.HIDDEN_n-1_SIZE, self.output_size], 
		# 			 stddev=0.01), name='W_n')
		# 	b_n = tf.Variable(tf.zeros(self.output_size), name='b_n')
		# 	self.Q = tf.matmul(h_n-1, W_n) + b_n
		#
		#############################################################
		# if target:
		# 	w1_inp = tf.zeros([self.input_size, self.hidden[0]])
		# 	w2_inp = tf.zeros([self.hidden[0], self.hidden[1]])
		# 	w3_inp = tf.zeros([self.hidden[1], self.output_size])
		# else:
		# 	w1_inp = tf.truncated_normal([self.input_size, self.hidden[0]], stddev=0.01)
		# 	w2_inp = tf.truncated_normal([self.hidden[0], self.hidden[1]], stddev=0.01)
		# 	w3_inp = tf.truncated_normal([self.hidden[1], self.output_size], stddev=0.01)

		inp = x
		params = []
		# print self.hidden
		for i in xrange(1, len(self.hidden), 1):
			if target:
				w_inp = tf.zeros([self.hidden[i-1], self.hidden[i]])
			else:
				w_inp = tf.random_normal([self.hidden[i-1], self.hidden[i]], stddev=2./(self.hidden[i-1]+self.hidden[i]))
			w = tf.Variable(w_inp, name='w'+str(i))
			b = tf.Variable(tf.zeros(self.hidden[i]), name='b'+str(i))
			a = tf.matmul(inp, w) + b
			params = params + [w]
			if i < len(self.hidden)-1:
				h = tf.nn.leaky_relu(a)
				# h = tf.tanh(a)
				inp = h
		out = a
		saver = tf.train.Saver(
					tf.global_variables(), max_to_keep=20)
		if target:
			out = tf.reduce_max(out, 1)
			# self.target_saver = tf.train.Saver(
			# 		tf.global_variables(), max_to_keep=20)
			self.target_saver = saver
		else:
			self.y = tf.placeholder(tf.float32, [None, self.output_size], name='output_placeholder')
			y = self.y
			############################################################
			# Next, compute the loss.
			#
			# First, compute the q-values. Note that you need to calculate these
			# for the actions in the (s,a,s',r) tuples from the experience replay's minibatch
			#
			# Next, compute the l2 loss between these estimated q-values and 
			# the target (which is computed using the frozen target network)
			#
			############################################################
			self.mask = tf.placeholder(tf.float32, [None, self.output_size], name='mask_placeholder')
			y_hat = tf.multiply(out, self.mask)
			# y_hat = out
			# y = tf.multiply(self.y, self.mask)
			loss = tf.losses.mean_squared_error(y, y_hat)
			# loss = tf.reduce_mean(tf.squared_difference(y_hat, y))
			# l2 = tf.norm(params[0]) + tf.norm(params[1]) + tf.norm(params[2])
			# for x in params[1:]:
			# 	l2 = l2 + tf.norm(x)
			# loss = loss + self.lmd * l2
			############################################################
			# Finally, choose a gradient descent algorithm : SGD/RMSProp/Adam. 
			#
			# For instance:
			# optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
			# global_step = tf.Variable(0, name='global_step', trainable=False)
			# self.train_op = optimizer.minimize(self.loss, global_step=global_step)
			#
			############################################################

			# Your code here
			optimizer = tf.train.AdamOptimizer(self.lr)
			global_step = tf.Variable(0, name='global_step', trainable=False)
			self.train_op = optimizer.minimize(loss, global_step=global_step)
			# self.policy_saver = tf.train.Saver(
			# 		tf.global_variables(), max_to_keep=20)
			self.policy_saver = saver
			self.loss = loss
			############################################################
		return x, out, params

	# Create the Q-network
  	def initialize_network(self):
  		self.target_graph = tf.Graph()
  		with self.target_graph.as_default():
  			self.tx, self.tout, self.tparams = self._build_graph(True)
  		
  		self.policy_graph = tf.Graph()
  		with self.policy_graph.as_default():
  			self.x, self.out, self.params = self._build_graph(False)
  		
	def _change_target(self, step):
		ckpt = self.policy_saver.save(self.policy_sess, self.model_dir+'ckpt-%d'%(step))
		self.target_saver.restore(self.target_sess, ckpt)

	def train(self, episodes_num=EPISODES_NUM):
		
		# Initialize summary for TensorBoard 						
		summary_writer = tf.summary.FileWriter(self.LOG_DIR, self.policy_graph)	
		summary = tf.Summary()
		# Alternatively, you could use animated real-time plots from matplotlib 
		# (https://stackoverflow.com/a/24228275/3284912)
		
		# Initialize the TF session
		self.policy_sess = tf.Session(graph=self.policy_graph)
		with self.policy_graph.as_default():	
			self.policy_sess.run(tf.global_variables_initializer())
		
		self.target_sess = tf.Session(graph=self.target_graph)
		with self.target_graph.as_default():
			self.target_sess.run(tf.global_variables_initializer())
		############################################################
		# Initialize other variables (like the replay memory)
		############################################################

		# Your code here
		memory = []
		total_steps = 0
		############################################################
		# Main training loop
		# 
		# In each episode, 
		#	pick the action for the given state, 
		#	perform a 'step' in the environment to get the reward and next state,
		#	update the replay buffer,
		#	sample a random minibatch from the replay buffer,
		# 	perform Q-learning,
		#	update the target network, if required.
		#
		#
		#
		# You'll need to write code in various places in the following skeleton
		#
		############################################################
		for episode in xrange(self.max_episode):
		  
			state = self.env.reset()
			steps = 0
			loss = 0.
			############################################################
			# Episode-specific initializations go here.
			############################################################
			#
			# Your code here
			#
			############################################################

			while True:
				steps += 1
				total_steps += 1
				############################################################
				# Pick the next action using epsilon greedy and and execute it
				############################################################

				# Your code here
				if random.random() < self.epsilon:
					action = np.random.randint(self.output_size)
				else:
					out = self.policy_sess.run(self.out, feed_dict={self.x: np.array(state).reshape([1,self.input_size])})
					action = np.argmax(out)
					
				############################################################
				# Step in the environment. Something like: 
				# next_state, reward, done, _ = self.env.step(action)
				############################################################

				# Your code here
				next_state, reward, done, _ = self.env.step(action)
				if done:
					reward = -10
				############################################################
				# Update the (limited) replay buffer. 
				#
				# Note : when the replay buffer is full, you'll need to 
				# remove an entry to accommodate a new one.
				############################################################

				# Your code here
				if len(memory) == self.REPLAY_MEMORY_SIZE:
					del memory[0]
				memory.append((state, action, reward, next_state))
				state = next_state
				############################################################
				# Sample a random minibatch and perform Q-learning (fetch max Q at s') 
				#
				# Remember, the target (r + gamma * max Q) is computed    
				# with the help of the target network.
				# Compute this target and pass it to the network for computing 
				# and minimizing the loss with the current estimates
				#
				############################################################

				# Your code here
				if len(memory) >= self.batch_size:
					actions = []
					states = []
					next_states = []
					rewards = []
					batch = random.sample(memory, self.batch_size)
					for p in batch:
						states.append(p[0])
						actions.append(p[1])
						rewards.append(p[2])
						next_states.append(p[3])

					next_states = np.array(next_states)
					out = self.target_sess.run(self.tout, feed_dict={self.tx: next_states})
					
					targets = []
					masks = []
					for p, r, i in zip(out, rewards, actions):
						c = [0, 0]
						c[i] = 1.
						targets.append([r+self.gamma*p, 0] if i==0 else [0, r+self.gamma*p])
						masks.append(c)

					states = np.array(states)
					targets = np.array(targets)
					masks = np.array(masks)

					# print 'states: {}, targets: {}, masks: {}'.format(states.shape, targets.shape, masks.shape)
					# _, l = self.policy_sess.run([self.train_op, self.loss],
					# 	feed_dict={self.x: states, self.y: targets,
					# 	self.mask: masks})
					self.policy_sess.run(self.train_op, feed_dict={self.x: states, self.y: targets,
						self.mask: masks})

					if total_steps % self.TARGET_UPDATE_FREQ == 0:
						self._change_target(total_steps)
						
				############################################################
			  	# Update target weights. 
			  	#
			  	# Something along the lines of:
				# if total_steps % self.TARGET_UPDATE_FREQ == 0:
				# 	target_weights = self.session.run(self.weights)
				############################################################

				# Your code here

				############################################################
				# Break out of the loop if the episode ends
				#
				# Something like:
				# if done or (episode_length == self.MAX_STEPS):
				# 	break
				#
				############################################################

				# Your code here
				if done:
					break

			############################################################
			# Logging. 
			#
			# Very important. This is what gives an idea of how good the current
			# experiment is, and if one should terminate and re-run with new parameters
			# The earlier you learn how to read and visualize experiment logs quickly,
			# the faster you'll be able to prototype and learn.
			#
			# Use any debugging information you think you need.
			# For instance :
			if self.epsilon > 0.01: # and total_steps % 10 == 0:
				self.epsilon *= self.epsilon_decay
			print("Training: Episode = %d, Length = %d, Global step = %d, epsilon: %f" % (episode, steps, total_steps, self.epsilon))
			summary.value.add(tag="episode length", simple_value=steps)
			summary_writer.add_summary(summary, episode)


	# Simple function to visually 'test' a policy
	def playPolicy(self):
		
		done = False
		steps = 0
		state = self.env.reset()
		
		# we assume the CartPole task to be solved if the pole remains upright for 200 steps
		while not done and steps < 200: 	
			self.env.render()
			q_vals = self.policy_sess.run(self.out, feed_dict={self.x: np.array(state).reshape([1,self.input_size])})
			action = np.argmax(q_vals)
			state, _, done, _ = self.env.step(action)
			steps += 1
		
		return steps


if __name__ == '__main__':

	# Create and initialize the model
	hidden_layers = [128, 128]
	dqn = DQN('CartPole-v0', hidden_layers, episode=1000)
	dqn.initialize_network()
	print("\nStarting training...\n")
	dqn.train()
	print("\nFinished training...\nCheck out some demonstrations\n")

	# Visualize the learned behaviour for a few episodes
	results = []
	for i in range(50):
		episode_length = dqn.playPolicy()
		print("Test steps = ", episode_length)
		results.append(episode_length)

	print("Mean steps = ", sum(results) / len(results))	
	print("\nFinished.")
	print("\nCiao, and hasta la vista...\n")