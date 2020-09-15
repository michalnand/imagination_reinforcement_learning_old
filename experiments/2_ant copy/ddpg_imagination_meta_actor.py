import gym
import pybulletgym
import numpy
import time

import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *

import models.ddpg_imagination_meta_actor_critic.src.model_critic     as ModelCritic
import models.ddpg_imagination_meta_actor_critic.src.model_actor      as ModelActor
import models.ddpg_imagination_meta_actor_critic.src.model_env        as ModelEnv
import models.ddpg_imagination_meta_actor_critic.src.config           as Config

path = "models/ddpg_imagination_meta_actor_critic/"

env = gym.make("AntPyBulletEnv-v0")
#env.render()

agent = libs_agents.AgentDDPGImaginationMetaActorCritic(env, ModelCritic, ModelActor, ModelEnv, Config)

max_iterations = 4*(10**6)
trainig = TrainingIterations(env, agent, max_iterations, path, 1000)
trainig.run() 

'''
agent.load(path)
agent.disable_training()
while True:
    agent.main()
    time.sleep(0.01)
'''