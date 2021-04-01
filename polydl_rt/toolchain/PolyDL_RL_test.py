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
from keras.models import load_model

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

def extract_Top5p(file):
    top5_df = pd.read_csv(file,header=None)
    top5_df = top5_df.dropna(axis=1)
    top5_df = top5_df.iloc[1:]
    top5_df = top5_df.sort_values(by=[4])

    top_count = math.ceil(top5_df.shape[0] * 0.05)
    top5_df = top5_df.iloc[:top_count]

    # Split to integers
    df = top5_df[2]
    df = df.apply(lambda x : split_tile_sizes(x))
    df = df.str.split("_")

    df = df.values.tolist()

    for j in range(len(df)):
        df[j] = [int(i) for i in df[j]]

    return df

# Setting Up environment.
problem_size1=[[128,2048,4096],[35,8457,2560]]

files = []
fileNameList = []
import os
for dirname, _, filenames in os.walk('results_matmul/'):
    for filename in filenames:
#         print(os.path.join(dirname, filename))
        files.append(os.path.join(dirname, filename))
        fileNameList.append(filename)

for idx, file in enumerate(files):
    print(extract_Top5p(file))
    problem_size = extract_Top5p(file)
    # exit("Stop")
    # problem_size = []


    # Set up Action and State space
    state_shape = (6,)
    action_space = list(range(7))
    # action_space = [6,7,8,15,16,17,18]
    action_shape = len(action_space)
    print(action_space)


    # Initializing Variables
    num_episodes = len(problem_size)
    max_steps_per_episode = 20

    print_reward = []
    memoization_list = []

    model = load_model('model.h5')

    with open('your_results.csv', 'a+') as f:
        f.write("Episode count -> %s \n" % fileNameList[idx])
        f.close()


    # Execution per episode
    for episode in range(num_episodes):
        env = PolyDL_Env(problem_size[episode],"matmul/"+fileNameList[idx])
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

        for step in range(max_steps_per_episode):

            #explore or exploit
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

            state = new_state

            print_to_csv.append([state[0],env._currentGFlops()])

        with open('your_results.csv', 'a+') as f:
            f.write("Problem ")
            for val in problem_size[episode]:
                f.write("%s_ " % val)
            f.write("GFLops : %s "% env._currentGFlops())
            f.write("\n")
            f.close()
        
        with open('your_results.csv', 'a+') as f:
            for val in state:
                f.write("State %s " % val)
            f.write("\n")
            f.close()
        
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

print(print_reward)