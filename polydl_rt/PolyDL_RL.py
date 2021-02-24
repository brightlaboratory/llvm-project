import os
import math
import random
from random import sample 
import time

import pandas as pd
import numpy as np

from envs import PolyDL_Env

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt 

# Setting Up environment.
problem_size=[[32,32,32],[1024,1024,1024],[128,2048,4096],[2048,4096,32]]
env = PolyDL_Env(problem_size[0])

# Set up Action and State space
state_shape = (9,)
action_space = list(range(19))
action_space = [6,7,8,15,16,17,18]
action_shape = len(action_space)
env._get_state()
print(action_space)

# DNN model for DQN
def create_model(state_shape,out_shape):
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu', input_shape=state_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(out_shape, activation=None)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.005),
        loss="mean_squared_error")
    return model

# Initializing Variables
num_episodes = 500
max_steps_per_episode = 60

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.005

print_reward = []
memoization_list = []

# Create Initial and target Model
model = create_model(state_shape,action_shape)
target_model = create_model(state_shape,action_shape)

# Execution per episode
for episode in range(num_episodes):
    env._reset()
    state = env._get_state()
    state = np.array(state).reshape(1,state_shape[0])
    # print("State **************************************************************", state)

    # print("State Shape", state)
    print("Model ", model.predict(state))
    # exit

    #Total Reward will be the learning curve
    totalReward = 0

    print_to_csv = []

    with open('your_file.csv', 'a+') as f:
        f.write("Episode count -> %s \n" % episode)

    for step in range(max_steps_per_episode):

        #explore or exploit
        exploration_rate_threshold = random.uniform(0,1)
        if exploration_rate_threshold < exploration_rate:
            action = random.sample(list(range(7)), 1)[0]
            # action = np.argmax(model.predict(state)[0])
        else:
            action = np.argmax(model.predict(state)[0])

        new_state, reward, done= env._step(action)

        new_state = np.array(new_state).reshape(1,state_shape[0])

        totalReward+=reward

        memoization_list.append([state,action,reward,new_state,done])
            
        #Fit into model
        if(len(memoization_list)>10):
            samples = sample(memoization_list,10)
            for eachSample in samples:
                old_state, action, reward, new_state, done = eachSample
                target = target_model.predict(old_state)
                if done:
                    target[0][action] = reward
                else:
                    expected_reward = max(target_model.predict(new_state)[0])
                    target[0][action] = reward + expected_reward * discount_rate
                model.fit(old_state, target, epochs=1, verbose=0)
        
        #Updating weights into target model
        state = new_state

        if not(step%10):
            weights = model.get_weights()
            target_weights = target_model.get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = weights[i] * 0.9 + target_weights[i] * 0.1
            target_model.set_weights(target_weights)

        # print_to_csv.append(env._currentGFlops())
        with open('your_file.csv', 'a+') as f:
            for item in state[0]:
                f.write("%s," % item)
            f.write("%s," %env._currentGFlops())
            f.write("\n")

    print_reward.append(totalReward)
    
    #Decaying the exploration rate after every episode
    exploration_rate = min_exploration_rate + \
    (max_exploration_rate - min_exploration_rate)* np.exp(-exploration_decay_rate*episode)

    # with open('your_file.csv', 'a+') as f:
    #     for item in print_to_csv:
    #         f.write("%s," % item)
    #     f.write("\n")

print(print_reward)