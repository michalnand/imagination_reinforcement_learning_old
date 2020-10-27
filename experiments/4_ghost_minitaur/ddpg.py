import gym
import pybulletgym
import numpy
import time

import sys
sys.path.insert(0, '../..')

import libs_agents
from libs_common.Training import *


import models.ddpg.model.src.model_critic     as ModelCritic
import models.ddpg.model.src.model_actor      as ModelActor
import models.ddpg.model.src.config           as Config



from pybullet_envs.bullet import minitaur_gym_env
from pybullet_envs.bullet import minitaur_env_randomizer

randomizer = (minitaur_env_randomizer.MinitaurEnvRandomizer())
env = minitaur_gym_env.MinitaurBulletEnv(   render=False,
                                            leg_model_enabled=False,
                                            motor_velocity_limit=numpy.inf,
                                            pd_control_enabled=True,
                                            accurate_motor_model_enabled=True,
                                            motor_overheat_protection=True,
                                            env_randomizer=randomizer,
                                            hard_reset=False)

class Wrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        action_ = numpy.pi*(0.1*action + 0.5)
        return self.env.step(action_)

env = Wrapper(env)



path = "models/ddpg/model/" 

agent = libs_agents.AgentDDPG(env, ModelCritic, ModelActor, Config)

max_iterations = 4*(10**6)
trainig = TrainingIterations(env, agent, max_iterations, path, 10000)
trainig.run()

'''
agent.load(path)
agent.disable_training()
while True:
    reward, done = agent.main()
    time.sleep(0.01)
'''