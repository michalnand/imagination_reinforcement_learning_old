# imagination_reinforcement_learning
imagination RL experiments



# 0 line follower

![](doc/images/line_follower.gif)

![](experiments/0_line_follower/results/training_score_per_episode.png)

* DDPG : common ddpg
* DDPG + imagination : DDPG imagination (4 rollouts + 4 steps) and bonus reward from imagination
* DDPG + imagination with meta-actor : imagined sttes entropty maximization exploration

# 1 lunar lander

![](doc/images/lunar_lander.gif)

![](experiments/1_lunar_lander/results/training_score_per_episode.png)

* DDPG : common ddpg
* DDPG + imagination : DDPG imagination (4 rollouts + 4 steps) and bonus reward from imagination
* DDPG + imagination with meta-actor : imagined sttes entropty maximization exploration

# 2 pybullet Ant walking

![](doc/images/ant.gif)

![](experiments/2_ant/results/training_score_per_episode.png)

* DDPG : common ddpg
* DDPG + imagination : DDPG imagination (4 rollouts + 4 steps) and bonus reward from imagination
* DDPG + imagination with meta-actor : imagined sttes entropty maximization exploration


# 4 atari pacman


last conv. layer attention visualisation
![](doc/images/pacman.gif)


pacman with curiosity - forward model for state prediction used
![](doc/images/pacman_curiosity.gif)

# dependences
cmake python3 pip3 swig
numpy numpy matplotlib torch pillow opencv-python 

envs : gym pybullet pybulletgym 'gym[atari]' 'gym[box2d]' gym-super-mario-bros gym_2048