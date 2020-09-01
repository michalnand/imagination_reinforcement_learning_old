import sys
sys.path.insert(0, '..')

import agents
import time
import gym
import gym_super_mario_bros
import numpy

import models.dqn_imagination.src.model            as Model
import models.dqn_imagination.src.model_env        as ModelEnv
import models.dqn_imagination.src.config           as Config


from common.Training import *
from common.atari_wrapper import *


path = "models/dqn_imagination/"

env = gym.make("SuperMarioBros-v0")
env = SuperMarioWrapper(env)
env.reset()


agent = agents.AgentDQNImagination(env, Model, ModelEnv, Config)

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