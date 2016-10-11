# -*- coding: utf-8 -*-

import six.moves.cPickle as pickle
import copy
import os
import numpy as np
from chainer import cuda

from cnn_feature_extractor import CnnFeatureExtractor
from q_net import QNet


class CnnDqnAgent(object):
    def __init__(self):
        super(CnnDqnAgent, self).__init__()
        self.policy_frozen = False
        self.epsilon_delta = 1.0 / 10 ** 4.4
        self.min_eps = 0.1
        self.actions = [0, 1, 2]

        self.cnn_feature_extractor = 'alexnet_feature_extractor.pickle'
        self.model = 'bvlc_alexnet.caffemodel'
        self.model_type = 'alexnet'
        self.image_feature_dim = 256 * 6 * 6
        self.image_feature_count = 1

        self.prediction_update_tick = 0
    
    def _observation_to_featurevec(self, observation):
        feature_image = [self.feature_extractor(observation["image"][i]) for i in range(self.image_feature_count)]
        return np.concatenate(feature_image + observation["depth"])

    def agent_init(self, **options):
        self.use_gpu = options['use_gpu']
        self.depth_image_dim = options['depth_image_dim']
        self.q_net_input_dim = self.image_feature_dim * self.image_feature_count + self.depth_image_dim

        if os.path.exists(self.cnn_feature_extractor):
            print("loading... " + self.cnn_feature_extractor)
            with open(self.cnn_feature_extractor, 'rb') as f:
                self.feature_extractor = pickle.load(f)
            print("done")
        else:
            print('there is no chainer alexnet model file ', self.cnn_feature_extractor)
            print('making chainer model from ', self.model)
            print('this process take a tens of minutes.')
            self.feature_extractor = CnnFeatureExtractor(self.use_gpu, self.model, self.model_type, self.image_feature_dim)
            pickle.dump(self.feature_extractor, open(self.cnn_feature_extractor, 'wb'))
            print("pickle.dump finished")

        self.time = 0
        self.epsilon = 1.0  # Initial exploratoin rate
        self.q_net = QNet(self.use_gpu, self.actions, self.q_net_input_dim)

    def agent_start(self, observation):
        # Initialize State
        self.state = np.zeros((self.q_net.hist_size, self.q_net_input_dim), dtype=np.float32)
        
        new_feature_vec = self._observation_to_featurevec(observation)
        self.state[0, :] = new_feature_vec
        
        # Generate an Action e-greedy
        state_ = np.expand_dims(self.state, 0)
        if self.use_gpu >= 0:
            state_ = cuda.to_gpu(state_, device=self.use_gpu)
        action, _, deg_intereset = self.q_net.e_greedy(state_, self.epsilon)
        return_action = action

        # Update for next step
        self.last_action = copy.deepcopy(return_action)
        self.last_state = self.state.copy()
        self.last_observation = new_feature_vec

        return return_action, deg_intereset

    def agent_step(self, reward, observation):
        new_feature_vec = self._observation_to_featurevec(observation)
        past_states = self.state[0:-1, :]
        self.state[0, :] = new_feature_vec
        self.state[1:, :] =  past_states
        
        # Exploration decays along the time sequence
        state_ = np.expand_dims(self.state, 0)
        if self.use_gpu >= 0:
            state_ = cuda.to_gpu(state_, device=self.use_gpu)

        if self.policy_frozen is False:  # Learning ON/OFF
            if self.q_net.initial_exploration < self.time:
                self.epsilon -= self.epsilon_delta
                if self.epsilon < self.min_eps:
                    self.epsilon = self.min_eps
                eps = self.epsilon
            else:  # Initial Exploation Phase
                print("Initial Exploration : %d/%d steps" % (self.time, self.q_net.initial_exploration)),
                eps = 1.0
        else:  # Evaluation
            print("Policy is Frozen")
            eps = 0.05

        # Generate an Action by e-greedy action selection
        action, q_now, deg_intereset = self.q_net.e_greedy(state_, eps)

        return action, eps, q_now, new_feature_vec, deg_intereset
    
    def agent_step_update(self, reward, action, eps, q_now, new_feature_vec, deg_intereset):
        # Learning Phase
        if self.policy_frozen is False:  # Learning ON/OFF
            self.q_net.stock_experience(self.time, self.last_state, self.last_action, reward, self.state, False)
            self.q_net.experience_replay(self.time)
        
        self.prediction_update_tick += 1
        if self.prediction_update_tick >= 10:
            self.prediction_update_tick = 0
            print('prediction update')
            self.q_net.prediction_update()

        # Target model update
        if self.q_net.initial_exploration < self.time and np.mod(self.time, self.q_net.target_model_update_freq) == 0:
            print("Model Updated")
            self.q_net.target_model_update()

        # Simple text based visualization
        if self.use_gpu >= 0:
            q_max = np.max(q_now.get())
        else:
            q_max = np.max(q_now)

        print('Step:%d  Action:%d  Reward:%.1f  Epsilon:%.6f  Q_max:%3f def_interest:%3f' % (
            self.time, self.q_net.action_to_index(action), reward, eps, q_max, deg_intereset))

        # Updates for next step
        self.last_observation = new_feature_vec

        if self.policy_frozen is False:
            self.last_action = copy.deepcopy(action)
            self.last_state = self.state.copy()
            self.time += 1

    def agent_end(self, reward):  # Episode Terminated
        print('episode finished. Reward:%.1f / Epsilon:%.6f' % (reward, self.epsilon))

        # Learning Phase
        if self.policy_frozen is False:  # Learning ON/OFF
            self.q_net.stock_experience(self.time, self.last_state, self.last_action, reward, self.last_state, True)
            self.q_net.experience_replay(self.time)

        # Target model update
        if self.q_net.initial_exploration < self.time and np.mod(self.time, self.q_net.target_model_update_freq) == 0:
            print("Model Updated")
            self.q_net.target_model_update()

        # Time count
        if self.policy_frozen is False:
            self.time += 1
