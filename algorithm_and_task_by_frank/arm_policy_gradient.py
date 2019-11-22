#!/usr/bin/env python3
#https://gist.github.com/onimaru/ea2f88c2156a77ce7262fb5e2f112fe0
import numpy as np
import matplotlib.pyplot as plt 
import gym
import time
import sys
#from sklearn.utils.extmath import softmax   #for a value not from a matrix       #TODO replace one layer NN of softmax 

import openai_ros2
#from openai_ros2.envs import LobotArmMoveSimpleEnv  
from openai_ros2.envs import LobotArmMoveSimpleRandomGoalEnv
from openai_ros2.robots import LobotArmSim
import rclpy
from gym.spaces import MultiDiscrete
from typing import Type
env: LobotArmMoveSimpleRandomGoalEnv = gym.make('LobotArmMoveSimple-v0')

'''
env = gym.make('CartPole-v0')
'''
    
def softmax(state,theta):  #TODO to try out sklearn softmax function , will it get the same?
    a = np.matmul(state,theta)   # [1x6]  [6x3] = 3 value  1*3    #5(Porbabilty of action) x 3 (joint)
    return np.exp(a)/np.sum(np.exp(a))  

def policy(softmax_probs):
    if np.random.rand() < softmax_probs[0]:  #TODO softmax_probs[0]  how to selecte one by one and return 0-4? multidiscrete(multi classification)
        return 0
    else: return 1

#ddpg - continous action space  ,    
'''  need compare one by one? use dictionary here?   

softmax_probs[0]: #joint 0  0-4
softmax_probs[1]: #joint 1
softmax_probs[2]: #joint 2
'''

def grads(action,softmax_probs):  #TODO set action for 3 joint. 
    s = softmax_probs
    if action == 0:
        return np.array([s[1],-s[1]])[None,:]
    else: 
        return np.array([-s[0],s[0]])[None,:]

def get_episode(theta, observation):  #theta,observation
    env.reset() 
    state = list(observation.position_data)+list(observation.velocity_data) #state observation  
    #state=env.reset()
    episode = []
    env.render()
    while True:
        #env.render()    # uncomment this to enable render mode
        s = softmax(state,theta)   #multiply two matrix (1*4 X 4*2) []
        action = policy(s)  #need a multidiscrete value.
        next_state, reward, done, _ = env.step(action)
        
        episode.append((state,reward,action,s))
        state = next_state
        if done: break
    return episode

def cp_play(n_episodes,alpha,y):
    R = []
    action_space: Type[MultiDiscrete] = env.action_space #TODO action space 5 5 5 
    env.reset() 
    #rclpy.spin_once(env.node)
    action = action_space.sample()
    observation, reward, done, info = env.step(action)
    #env.render()
    episode_length = []
    theta = np.random.rand(6,3) #np.random.rand(4,2)  #TODO to confirm the exact value , declare random once only.
    #theta  = [np.random.uniform(-2.4,2.4)],[np.random.uniform(-1.6,2.1)],[np.random.uniform(-1.6,3.1)],[np.random.uniform(-10,10)],[np.random.uniform(-20,20)],[np.random.uniform(-10,15)]
    #theta  += list(theta).append([np.random.uniform(-2.4,2.4)],[np.random.uniform(-1.6,2.1)],[np.random.uniform(-1.6,3.1)],[np.random.uniform(-10,10)],[np.random.uniform(-20,20)],[np.random.uniform(-10,15)])
    
    for i in range(n_episodes):
        episode = get_episode(theta,observation)  #add in batch type episode = get_episode(theta,observation)
        states = [item[0] for item in episode]
        rewards = [item[1] for item in episode]
        actions = [item[2] for item in episode]
        softs = [item[3] for item in episode]
        R.append(sum(rewards))  #to check total Reward for each episodes only
        episode_length.append(len(episode))  #how long each episode is to store it
        grad = [grads(i,s) for i,s in zip(actions,softs)]
        
        for t in range(len(grad)):
            theta += alpha*np.array(np.dot(states[t][None,:].T,grad[t])*sum([r*(y**r) for i,r in enumerate(rewards[t:])])) #theta will update till they all us
        

    #TODO experince replay







        l=100
        if len(R)>=l:
            for j in range(l,len(R)):
                if np.mean(R[j-l:j])>=195:# next 100 epsiode score average of 195
                    print('Solved in:')
                    env.close()
                    sys.exit('Episode {} with average reward in last 100 steps: {}.'.format(j-l, np.mean(R[j-l:j])))
        
        print('Episodes: {}, Average reward: {}'.format(i,sum(rewards)))
    return R

cp_play(1000,0.005,0.99)

env.close()
sys.exit('Not solved in 1000 steps')
