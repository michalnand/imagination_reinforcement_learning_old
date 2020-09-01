import sys
sys.path.insert(0, '..')
import gym
import pybulletgym
import numpy
import time


import agents


import models.ddpg.src.model_critic     as ModelCritic
import models.ddpg.src.model_actor      as ModelActor
import models.ddpg.src.config           as Config
from common.Training import *

path = "models/ddpg/"

env = gym.make("HalfCheetahPyBulletEnv-v0")
#env.render()

agent = agents.AgentDDPG(env, ModelCritic, ModelActor, Config)

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