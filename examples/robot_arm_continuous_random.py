#!/usr/bin/env python3
""" NOT A REAL TRAINING SCRIPT
Please check the README.md in located in this same folder
for an explanation of this script"""

import gym
import time
import numpy
import openai_ros2
from openai_ros2.envs import LobotArmMoveRandomConActEnv #LobotArmMoveRandomConActEnv   #LobotArmMoveSimpleEnv
#from openai_ros2.robots import LobotArmConActSim   #LobotArmConActSim    #LobotArmSim 
import rclpy
import random
from gym.spaces import MultiDiscrete
from typing import Type

env: LobotArmMoveRandomConActEnv= gym.make('LobotArmMoveRandom-v1')   #'LobotArmMoveSimple-v0' LobotArmMoveSimpleEnv
action_space: Type[MultiDiscrete] = env.action_space
rclpy.spin_once(env.node)
env.reset()
while True:
    print("-------------Starting----------------")
    for x in range(500):
        # action = numpy.array([1.00, -1.01, 1.01])
        action = action_space.sample()
        observation, reward, done, info = env.step(action)
        # Type hints
        observation: LobotArmMoveRandomConActEnv.ObservationData  #LobotArmMoveSimpleEnv
        reward: float
        done: bool
        info: str
        if done:
            break
    time.sleep(1.0)
    print("-------------Resetting environment---------------")
    env.reset()
    print("-------------Reset finished----------------")
