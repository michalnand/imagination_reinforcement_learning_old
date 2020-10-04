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

**basic python libs**
pip3 install numpy numpy matplotlib torch pillow opencv-python 

**graph neural networks**
when CPU only :

pip3 install networkx torch_geometric torch_sparse torch_scatter


for CUDA different packages are reuired :
- this is for cuda 10.2
- and pytorch 1.6


pip3 install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip3 install torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip3 install torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip3 install torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip3 install torch-geometric

see https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

**environments**
pip3 install  gym pybullet pybulletgym 'gym[atari]' 'gym[box2d]' gym-super-mario-bros gym_2048