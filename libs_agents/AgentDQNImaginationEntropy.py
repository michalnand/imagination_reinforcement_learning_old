import numpy
import torch
from .ExperienceBuffer import *

import cv2


class AgentDQNImaginationEntropy():
    def __init__(self, env, Model, ModelEnv, Config):
        self.env = env

        config = Config.Config()


        self.gamma                  = config.gamma
        self.update_frequency       = config.update_frequency
        self.tau                    = config.tau

        self.batch_size             = config.batch_size 
        self.bellman_steps          = config.bellman_steps
                 
        self.imagination_rollouts   = config.imagination_rollouts
        self.imagination_steps      = config.imagination_steps

        self.entropy_beta           = config.entropy_beta
        self.curiosity_beta         = config.curiosity_beta

        self.exploration            = config.exploration



        self.state_shape    = self.env.observation_space.shape
        self.actions_count  = self.env.action_space.n


        self.experience_replay = ExperienceBuffer(config.experience_replay_size, self.bellman_steps)

        self.model          = Model.Model(self.state_shape, self.actions_count)
        self.model_target   = Model.Model(self.state_shape, self.actions_count)
        self.optimizer      = torch.optim.Adam(self.model.parameters(), lr= config.learning_rate)

        for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
            target_param.data.copy_(param.data)


        self.model_env          = ModelEnv.Model(self.state_shape, self.actions_count)
        self.optimizer_env      = torch.optim.Adam(self.model_env.parameters(), lr= config.env_learning_rate)

    
        self.state    = env.reset()
        self.iterations     = 0
        self.enable_training()

    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False
    
    def main(self, show_activity = False):
        if self.enabled_training:
            self.exploration.process()
            self.epsilon = self.exploration.get()
        else:
            self.epsilon = self.exploration.get_testing()

        state_t     = torch.from_numpy(self.state).to(self.model.device).unsqueeze(0).float()
        
        q_values    = self.model(state_t)
        q_values    = q_values.squeeze(0).detach().to("cpu").numpy()
 
        action_idx_np, _ = self._sample_action(state_t, self.epsilon)

        self.action = action_idx_np[0]

        state_new, self.reward, done, self.info = self.env.step(self.action)
 
        if self.enabled_training:
            self.experience_replay.add(self.state, self.action, self.reward, done)


        if self.enabled_training and (self.iterations > self.experience_replay.size):
            if self.iterations%self.update_frequency == 0:
                self.train_model()
            
            #soft update target network
            for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_((1.0 - self.tau)*target_param.data + self.tau*param.data)

        if show_activity:
            self._show_entropy_activity(state_t)

        if done:
            self.state = self.env.reset()
        else:
            self.state = state_new.copy()

        
        self.iterations+= 1

        return self.reward, done
    
    def _show_activity(self, state, alpha = 0.6):
        activity_map    = self.model.get_activity_map(state)
        activity_map    = numpy.stack((activity_map,)*3, axis=-1)*[0, 0, 1]

        state_map    = numpy.stack((state[0],)*3, axis=-1)
        image        = alpha*state_map + (1.0 - alpha)*activity_map

        image        = (image - image.min())/(image.max() - image.min())

        image = cv2.resize(image, (400, 400), interpolation = cv2.INTER_AREA)
        cv2.imshow('state activity', image)
        cv2.waitKey(1)

    def _tensor_to_numpy(self, x, spacing = 4):
        x_np    = x.detach().to("cpu").numpy()
        height  = x_np.shape[2]
        width   = x_np.shape[3]

        rows    = 4
        cols    = 4
        
        result  = numpy.zeros((rows*(height+spacing), cols*(width+spacing)))
        for row in range(rows):
            for col in range(cols):
                idx = col + row*cols
                result[row*width + row*spacing:(row+1)*width + row*spacing, col*height + col*spacing:(col+1)*height + col*spacing] = x_np[idx][0]

        k = 1.0/(result.max() - result.min())
        q = 1.0 - k*result.max()   
        result = k*result + q

        return result


    def _show_entropy_activity(self, state_t):
        if self.iterations%10 != 0:
            return

        size                = 400
        states_imagined_t   = self._process_imagination(state_t, self.epsilon).squeeze(0)

        states_dif_t        = state_t.squeeze(0) - states_imagined_t
 
        entropy_t           = torch.std(states_dif_t, dim=0).mean(dim=0)
        

        states_imagined_np  = self._tensor_to_numpy(states_imagined_t)
        states_dif_np       = self._tensor_to_numpy(states_dif_t**2)
        
        entropy_np          = entropy_t.detach().to("cpu").numpy()
        k = 1.0/(entropy_np.max() - entropy_np.min())
        q = 1.0 - k*entropy_np.max()   
        entropy_np = k*entropy_np + q

        cv2.imshow('states_imagined_np', entropy_np)
        cv2.waitKey(1)
       
        print(">>>> ", states_imagined_t.shape, states_imagined_np.shape, entropy_t.shape)
        pass
        '''
        size            = 400

        action_t        = torch.zeros(1,  dtype=int)
        action_t[0]     = action

        state_t              = torch.from_numpy(state).unsqueeze(dim=0).to(self.model_env.device)
        action_one_hot_t     = self._one_hot_encoding(action_t)
        state_prediction     = self.model_env(state_t, action_one_hot_t).squeeze().detach().to("cpu").numpy()

      
        space   = numpy.zeros((state.shape[1], 4)) 
        
        dif     =  (state_next - state_prediction)**2
        dif     =  (dif - dif.min())/(dif.max() - dif.min())


        image   =  numpy.hstack((state[0], space, state_prediction[0], space, dif[0]))

        image = cv2.resize(image, (3*size, size), interpolation = cv2.INTER_AREA)


        font                   = cv2.FONT_HERSHEY_SIMPLEX


        cv2.putText(image,'state', (10 + 0*size, int(size*0.98)), font, 1, (255,255,255), 2)

        cv2.putText(image,'prediction', (10 + 1*size, int(size*0.98)), font, 1, (255,255,255), 2)

        cv2.putText(image,'error', (10 + 2*size, int(size*0.98)), font, 1, (255,255,255), 2)


        cv2.imshow('curiosity activity', image)
        cv2.waitKey(1)
        '''
        
    def train_model(self):
        state_t, action_t, reward_t, state_next_t, done_t = self.experience_replay.sample(self.batch_size, self.model.device)
        
        #environment model state prediction
        action_one_hot_t    = self._one_hot_encoding(action_t)
        state_predicted_t   = self.model_env(state_t, action_one_hot_t)

        #env model loss
        env_loss = (state_next_t - state_predicted_t)**2
        env_loss = env_loss.mean()

        #update env model
        self.optimizer_env.zero_grad()
        env_loss.backward() 
        self.optimizer_env.step()



        im_entropy, im_curiosity = self.intrinsics_motivation(state_t, action_t, state_next_t, state_predicted_t)

        intrinsics_motivation_t = self.entropy_beta*im_entropy + self.curiosity_beta*im_curiosity

        #q values, state now, state next
        q_predicted      = self.model.forward(state_t)
        q_predicted_next = self.model_target.forward(state_next_t)

        #compute target, n-step Q-learning
        q_target         = q_predicted.clone()
        for j in range(self.batch_size):
            gamma_        = self.gamma

            reward_sum = 0.0
            for i in range(self.bellman_steps):
                if done_t[j][i]:
                    gamma_ = 0.0
                reward_sum+= reward_t[j][i]*(gamma_**i)

            action_idx    = action_t[j]
            q_target[j][action_idx]   = reward_sum + (gamma_**self.bellman_steps)*torch.max(q_predicted_next[j]) + intrinsics_motivation_t[j]
 
        #train DQN model
        loss = ((q_target.detach() - q_predicted)**2)
        loss  = loss.mean() 

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-10.0, 10.0)
        self.optimizer.step()

    
    def save(self, save_path):
        self.model.save(save_path)
        self.model_env.save(save_path)

    def load(self, load_path):
        self.model.load(load_path)
        self.model_env.load(load_path)
    

    def _sample_action(self, state_t, epsilon):

        batch_size = state_t.shape[0]

        q_values_t  = self.model(state_t)

        action_idx_t     = torch.zeros(batch_size).to(self.model.device)

        action_one_hot_t = torch.zeros((batch_size, self.actions_count)).to(self.model.device)

        #e-greedy strategy
        for b in range(batch_size):
            action = torch.argmax(q_values_t[b])
            if numpy.random.random() < epsilon:
                action = numpy.random.randint(self.actions_count)

            action_idx_t[b]                 = action
            action_one_hot_t[b][action]     = 1.0
        
        action_idx_np       = action_idx_t.detach().to("cpu").numpy().astype(dtype=int)

        return action_idx_np, action_one_hot_t


    
    def _process_imagination(self, states_t, epsilon):

        self.model.eval()
        self.model_env.eval()

        batch_size  = states_t.shape[0]

        states_imagined_t      = torch.zeros((self.imagination_rollouts, batch_size, ) + self.state_shape ).to(self.model_env.device)

        for r in range(self.imagination_rollouts):
            states_imagined_t[r] = states_t.clone()

        for s in range(self.imagination_steps):
            #imagine rollouts
            for r in range(self.imagination_rollouts):
                _, action_t             = self._sample_action(states_imagined_t[r], epsilon)
                states_imagined_next_t  = self.model_env(states_imagined_t[r], action_t)
                states_imagined_t[r]    = states_imagined_next_t.clone()

        #swap axis (batch first) : batch rollout state
        states_imagined_t = states_imagined_t.transpose(1, 0)


        self.model.train()
        self.model_env.train()

        return states_imagined_t


    def _compute_entropy(self, states_t, states_initial_t):
        batch_size  = states_t.shape[0]
        result      = torch.zeros(batch_size).to(self.model_env.device)

        for b in range(batch_size):
            s_dif       = states_t[b] - states_initial_t[b]
            
            result[b]   = torch.std(s_dif.view(s_dif.size(0), -1), dim=0).mean()

        return result


    def intrinsics_motivation(self, state_t, action_t, state_next_t, state_predicted_t):
        
        #compute imagined states, use state_t as initial state
        states_imagined_t   = self._process_imagination(state_t, self.epsilon)
        
        #compute entropy of imagined states
        im_entropy           = self._compute_entropy(states_imagined_t.detach(), state_t)
       
        #compute curiosity
        im_curiosity         = ((state_next_t - state_predicted_t)**2).mean(dim = 1).detach()
        
        return im_entropy, im_curiosity
