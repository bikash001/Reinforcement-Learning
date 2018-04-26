import gym
import random

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
	def __init__(self, env):
		self.env = gym.make(env)
		assert len(self.env.observation_space.shape) == 1
		self.input_size = self.env.observation_space.shape[0]		# In case of cartpole, 4 state features
		self.output_size = self.env.action_space.n					# In case of cartpole, 2 actions (right/left)
	
	# Create the Q-network
  	def initialize_network(self):

  		# placeholder for the state-space input to the q-network
		self.x = tf.placeholder(tf.float32, [None, self.input_size])

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

		# Your code here

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

		# Your code here

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

		############################################################

	def train(self, episodes_num=EPISODES_NUM):
		
		# Initialize summary for TensorBoard 						
		summary_writer = tf.summary.FileWriter(self.LOG_DIR)	
		summary = tf.Summary()	
		# Alternatively, you could use animated real-time plots from matplotlib 
		# (https://stackoverflow.com/a/24228275/3284912)
		
		# Initialize the TF session
		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())
		
		############################################################
		# Initialize other variables (like the replay memory)
		############################################################

		# Your code here

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

		for episode in range(episodes_num):
		  
			state = self.env.reset()

			############################################################
			# Episode-specific initializations go here.
			############################################################
			#
			# Your code here
			#
			############################################################

			while True:

				############################################################
				# Pick the next action using epsilon greedy and and execute it
				############################################################

				# Your code here

				############################################################
				# Step in the environment. Something like: 
				# next_state, reward, done, _ = self.env.step(action)
				############################################################

				# Your code here

				############################################################
				# Update the (limited) replay buffer. 
				#
				# Note : when the replay buffer is full, you'll need to 
				# remove an entry to accommodate a new one.
				############################################################

				# Your code here

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

			print("Training: Episode = %d, Length = %d, Global step = %d" % (episode, episode_length, total_steps))
			summary.value.add(tag="episode length", simple_value=episode_length)
			summary_writer.add_summary(summary, episode)


	# Simple function to visually 'test' a policy
	def playPolicy(self):
		
		done = False
		steps = 0
		state = self.env.reset()
		
		# we assume the CartPole task to be solved if the pole remains upright for 200 steps
		while not done and steps < 200: 	
			self.env.render()				
			q_vals = self.session.run(self.Q, feed_dict={self.x: [state]})
			action = q_vals.argmax()
			state, _, done, _ = self.env.step(action)
			steps += 1
		
		return steps


if __name__ == '__main__':

	# Create and initialize the model
	dqn = DQN('CartPole-v0')
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