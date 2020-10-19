import gym
import numpy
import time

import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *
from libs_common.atari_wrapper import *

import models.a2c.src.model            as Model
import models.a2c.src.config           as Config

 
path = "models/a2c/"

#env = gym.make("MsPacmanNoFrameskip-v4")
env = gym.make("QbertNoFrameskip-v4")

env = AtariWrapper(env)
env.reset()


agent = libs_agents.AgentA2C(env, Model, Config)

max_iterations = 10*(10**6) 

trainig = TrainingIterations(env, agent, max_iterations, path, 10000)
trainig.run() 

'''
agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main(True)

    env.render()
    time.sleep(0.01)
'''