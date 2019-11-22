#!/usr/bin/env python3
import matplotlib.pyplot as plt 
import gym
import time
import numpy as np
import openai_ros2
#from openai_ros2.envs import LobotArmMoveSimpleEnv  
from openai_ros2.envs import LobotArmMoveSimpleRandomGoalEnv
from openai_ros2.robots import LobotArmSim
import rclpy
import random
from gym.spaces import MultiDiscrete
from typing import Type
MAXSTATES = 10**6  #infinity into discrete space. to save the q table data #TODO Q table. 
GAMMA = 0.6  #Q learning mathermatic   Discount 
ALPHA = 0.1  # Learning rateGAMMA = 0.90  #Q learning mathermatic   Discount  ALPHA = 0.01 
JOINT = 3
env: LobotArmMoveSimpleRandomGoalEnv = gym.make('LobotArmMoveSimple-v1')
#env: LobotArmMoveSimpleEnv = gym.make('LobotArmMoveSimple-v0')
action_space: Type[MultiDiscrete] = env.action_space
rclpy.spin_once(env.node)
env.reset()
rewardList = []
length=[]
action = action_space.sample()
observation, reward, done, info = env.step(action)


def create_bins():    #can change the value to suit the training ??
    bins = np.zeros((6,10)) # create the array szie to use #sentdex dicrete value
    bins[0] = np.linspace(-2.4, 2.4, bins.shape[1])  
    bins[1] = np.linspace(-1.6, 2.1, bins.shape[1])
    bins[2] = np.linspace(-1.6, 3.1, bins.shape[1])
    bins[3] = np.linspace(-10, 10, bins.shape[1])	
    bins[4] = np.linspace(-20, 20, bins.shape[1])    
    bins[5] = np.linspace(-10, 15, bins.shape[1])    
    return bins

def assign_bins(observation, bins):  #compare observation state is fall at which bins. and decided the state only.
	state = np.zeros(6)
	for i in range(6):
		state[i] = np.digitize(observation[i], bins[i]) #continuous varialbe to discrete ,.. # return the index f observation in the bins
	return state

def get_all_Q_Table_states_as_string():
	states = []
	for i in range(MAXSTATES): #TODO 1 initialize Q table with how many episode you want 
		states.append(str(i).zfill(5))
	return states   #return the state as 9999 because 10000 ep

def initialize_Q_table():
    Q=np.zeros((999999,JOINT,5))  #[state][joint][angle to move]  np 3 layer
    return Q

def get_state_as_string(state):
	string_state = ''.join(str(int(e)) for e in state) #set 
	return string_state

def max_Q_Joint_atNewState(Qtable,act):   #TODO update Q table flow and call the highest value flow
	max_v = np.full((Qtable.shape),-np.inf)	#Q value for 3 joint
	max_action=np.full((act.shape),-np.inf)
	for joint in range(len(Qtable)):  #define different joint
		if Qtable[joint][int(act[joint])] > max_v[joint][int(act[joint])]:
			max_v[joint][int(act[joint])] = Qtable[joint][int(act[joint])]   # have to move to second one 
			max_action[joint] = max_v[joint].argmax()   #action to take

	return max_action, max_v #either is left of right?      # max action 2<Action.BigPositive: 0.02>


def updateQvalue(Qtable,Alp,Gam,rewa,max_q_s1a1,a1):   #TODO update Q table flow and call the highest value flow
	for joint in range(len(Qtable)):  #define different joint
		Qtable[joint][int(a1[joint])] += Alp*(rewa + Gam*max_q_s1a1[joint][int(a1[joint])] -Qtable[joint][int(a1[joint])])   # have to move to second one 

	return Qtable #either is left of right?      # max action 2<Action.BigPositive: 0.02>

def max_Q_Joint_action(Qtable,act):   #TODO update Q table flow and call the highest value flow
    max_action=np.full((act.shape),-np.inf)
    for joint in range(len(Qtable)):  #define different joint
        if Qtable[joint].max() > -np.inf:
            max_action[joint] = Qtable[joint].argmax()   #action to take
    return max_action 

if __name__ == '__main__':
    bins = create_bins()  
    Q = initialize_Q_table()  
    for n in range(500):#how much iteration you want?
        print("-------------Starting----------------")
        done = False #initiallize to run 
        eps = 1.0 / np.sqrt(n+n*2)   #TODO check adaptive?   1.0 / (np.sqrt(n**2)+1)  
        total_reward =0
        state = get_state_as_string(assign_bins((list(observation.position_data)+list(observation.velocity_data)), bins))
        while not done:#500 state before next iteration	
            a= np.random.uniform()   #TODO reconfirm episolon and random value. should we include relationship?
            if a < eps:  #need random so it can explore more during maybe the sartup time  #TODO to check action for every state  / episode
                action = action_space.sample() # epsilon greedy  # exploration ,random choose action
            else:			 #TODO pandas
                action =max_Q_Joint_action(Q[int(state)],action)
            observation, reward, done, info = env.step(action) #action (0,0,1) wrong		
            total_reward += reward
            state_new = get_state_as_string(assign_bins((list(observation.position_data)+list(observation.velocity_data)), bins))# why the state will change here? new state after based on the env,.step and your action 
            a1, max_q_s1a1 = max_Q_Joint_atNewState(Q[int(state_new)],action)
            Q[int(state)]=updateQvalue(Q[int(state)],ALPHA,GAMMA,reward,max_q_s1a1,a1)
            state= state_new
            current_time = time.time()
            if done: #env._LobotArmMoveSimpleEnv__step_num > 100 or 
                break
        time.sleep(1.0)

        if n % 20 == 0:
            length.append(env._LobotArmMoveSimpleRandomGoalEnv__step_num)
            rewardList.append(env._LobotArmMoveSimpleRandomGoalEnv__cumulated_episode_reward)
            '''length.append(env._LobotArmMoveSimpleEnv__step_num)
            rewardList.append(env._LobotArmMoveSimpleEnv__cumulated_episode_reward)'''
        print("-------------Resetting ",n,"environment---------------")
        env.reset()
        print("-------------Reset finished----------------")
    print("-------------Training done----------------")

N = len(rewardList)
running_avg = np.empty(N)
for t in range(N):
	running_avg[t] = np.mean(rewardList[max(0, t-100):(t+1)])
plt.plot(running_avg)
plt.title("Running Average")
plt.show()


np.save('testingsave',Q)