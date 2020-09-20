import sys
sys.path.insert(0, '../..')

import libs_agents
import numpy
import time

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

observation = env.reset()

state_shape    = env.observation_space.shape
actions_count  = env.action_space.shape[0]

print("state_shape      = ", state_shape)
print("actions_count    = ", actions_count)

print("state = \n", observation)


agent = libs_agents.AgentRandomContinuous(env)

k = 0.1
fps = 0
while True:
    time_start      = time.time()
    reward, done    = agent.main()
    time_stop       = time.time()
    env.render()

    fps = (1.0-k)*fps + k*1.0/(time_stop - time_start)


    if reward != 0:
        print("reward = ", reward)
    
    if done:
        print("FPS = ", round(fps, 1))
        print("DONE \n\n")
    
    time.sleep(0.01)