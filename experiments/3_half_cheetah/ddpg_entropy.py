import gym
import pybulletgym
import numpy
import time

import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *


import models.ddpg_entropy.src.model_critic     as ModelCritic
import models.ddpg_entropy.src.model_actor      as ModelActor
import models.ddpg_entropy.src.model_entropy    as ModelEntropy
import models.ddpg_entropy.src.config           as Config

path = "models/ddpg_entropy/"

env = gym.make("HalfCheetahPyBulletEnv-v0")
#env.render()
 
agent = libs_agents.AgentDDPGEntropy(env, ModelCritic, ModelActor, ModelEntropy, Config)

max_iterations = 10*(10**6)
trainig = TrainingIterations(env, agent, max_iterations, path, 10000)
trainig.run()

'''
agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main()
    env.render()
    time.sleep(0.01)
'''