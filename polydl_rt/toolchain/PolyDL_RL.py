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

def split_unrollFactors(row):
    t=row.split('_')
    s1=""
    for val in t[:11]:
        s1+=val+"_"
    s1=s1[:-1]
    return s1

def split_tile_sizes(row):
    t=row.split('_')
    s1=""
    for val in t[5:11]:
        s1+=val+"_"
    s1=s1[:-1]
    return s1

def getProblemSize(fileName):
    train_df = pd.read_csv(fileName,header=None)
    train_df[0] = train_df.apply(lambda x : split_unrollFactors(x[0]),axis=1)
    train_df = train_df.groupby([0])

    train_df = train_df.max()
    train_df[0] = train_df.index
    train_df[0] = train_df.apply(lambda x : split_tile_sizes(x[0]),axis=1)

    df = train_df[0]
    df = df.str.split("_")
    df = df.values.tolist()

    for j in range(len(df)):
        df[j] = [int(i) for i in df[j]]
    
    return df

# Set up Action and State space
state_shape = (6,)
action_space = list(range(7))
# action_space = [6,7,8,15,16,17,18]
action_shape = len(action_space)

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

# Create Initial and target Model
model = create_model(state_shape,action_shape)
target_model = create_model(state_shape,action_shape)

# Setting Up environment.
files = []
import os
for dirname, _, filenames in os.walk('matmul/'):
    for filename in filenames:
#         print(os.path.join(dirname, filename))
        files.append(os.path.join(dirname, filename))

for file in files:
    # Take File Names from Command Line
    problem_size = getProblemSize(file)
    count_Size = len(problem_size)

    print("*********************************",problem_size[0])

    # Initializing Variables
    num_episodes = 200
    max_steps_per_episode = 21

    learning_rate = 0.1
    discount_rate = 0.99

    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.005

    print_reward = []
    memoization_list = []


    # Execution per episode
    for episode in range(num_episodes):
        env = PolyDL_Env(problem_size[episode%count_Size],file)
        env._reset()
        state = env._get_state()
        state = np.array(state).reshape(1,state_shape[0])
        MaxGFlop = env._MaxGFlops()

        # print("State Shape", state)
        print("Model ", model.predict(state))
        # exit

        #Total Reward will be the learning curve
        totalReward = 0

        print_to_csv = []

        with open('your_file.csv', 'a+') as f:
            f.write("Episode count -> %s \n" % episode)
            f.close()
        
        print("Explore Rate =>", exploration_rate)

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

            if MaxGFlop < env._MaxGFlops():
                MaxGFlop = env._MaxGFlops()
                with open('your_gflops.csv', 'a+') as f:
                    f.write("%s," %episode)
                    f.write("%s," %step)
                    f.write("%s," %MaxGFlop)
                    f.write("\n")
                    f.close()

            totalReward+=reward

            memoization_list.append([state,action,reward,new_state,done])

            #Updating weights into target model
            state = new_state

            #Fit into model
            batch_size=7
            if(len(memoization_list)>batch_size):
                # samples = sample(memoization_list,batch_size)
                """ vectorized implementation; 30x speed up compared with for loop """
                minibatch = random.sample(memoization_list, batch_size)
                old_state = np.array([tup[0][0] for tup in minibatch])
                action = np.array([tup[1] for tup in minibatch])
                reward = np.array([tup[2] for tup in minibatch])
                new_state = np.array([tup[3][0] for tup in minibatch])
                done = np.array([tup[4] for tup in minibatch])
                
                # Q(s', a)
                target = reward + discount_rate * np.amax(target_model.predict(new_state), axis=1)
                # end state target is reward itself (no lookahead)
                target[done] = reward[done]

                # Q(s, a)
                target_f = target_model.predict(old_state)
                # make the agent to approximately map the current state to future discounted reward
                target_f[range(batch_size), action] = target

                model.fit(old_state, target_f, epochs=1, verbose=0)
            

            if not(step%7):
                weights = model.get_weights()
                target_weights = target_model.get_weights()
                for i in range(len(target_weights)):
                    target_weights[i] = weights[i] * 0.9 + target_weights[i] * 0.1
                target_model.set_weights(target_weights)

            print_to_csv.append([state[0],env._currentGFlops()])

        with open('your_file.csv', 'a+') as f:
            for val in print_to_csv:
                for item in val[0]:
                    f.write("%s," % item)
                f.write("%s," %val[1])
                f.write("\n")
            f.close()

        print_reward.append(totalReward)
        
        with open('your_totalflops.csv', 'a+') as f:
            f.write("%s," %totalReward)
            f.close()
        
        #Decaying the exploration rate after every episode
        exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate)* np.exp(-exploration_decay_rate*episode)

        # with open('your_file.csv', 'a+') as f:
        #     for item in print_to_csv:
        #         f.write("%s," % item)
        #     f.write("\n")
model.save("model.h5")