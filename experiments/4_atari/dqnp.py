import gym
import numpy
import time

import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *
from libs_common.atari_wrapper import *

import models.dqnp.src.model            as Model
import models.dqnp.src.config           as Config


path = "models/dqnp/"

env = gym.make("MsPacmanNoFrameskip-v4")

env = AtariWrapper(env)
env.reset()


agent = libs_agents.AgentDQN(env, Model, Config)

max_iterations = 10*(10**6) 

trainig = TrainingIterations(env, agent, max_iterations, path, 256)
trainig.run() 

'''
agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main(True)

    env.render()
    time.sleep(0.01)
'''