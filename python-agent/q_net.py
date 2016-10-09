# -*- coding: utf-8 -*-

import copy
import numpy as np
import chainer
from chainer import cuda, Variable, optimizers
import chainer.functions as F
import chainer.links as L


class QCalculationChain(chainer.Chain):
    def __init__(self, input_size, hidden_size, output_size):
        super(QCalculationChain, self).__init__(
                l4 = L.Linear(input_size, hidden_size, wscale=np.sqrt(2)),
                q_value=L.Linear(hidden_size, output_size, initialW=np.zeros((output_size, hidden_size), dtype=np.float32))
            )
            
    def to_cpu(self):
        super(QCalculationChain, self).to_cpu()
        
    def to_gpu(self, device):
        super(QCalculationChain, self).to_gpu(device)

    def __call__(self, x):
        y = self.l4(x)
        y = F.relu(y)
        y = self.q_value(y)
        return y        

class DegInterestChain(chainer.Chain):
    def __init__(self, feature_vec_size, feature_vec_count, q_value_size):
        self.feature_vec_size = feature_vec_size
        self.feature_vec_count = feature_vec_count
        self.q_value_size = q_value_size
        self.input_size = feature_vec_size*feature_vec_count+q_value_size
        super(DegInterestChain, self).__init__(
                lstm1 = L.LSTM(self.input_size, feature_vec_size),
                l1 = L.Linear(feature_vec_size, feature_vec_size),
            )
        self.reset_state()
            
    def reset_state(self):
        self.lstm1.reset_state()
        self.P = None
        self.E = None
    
    def to_cpu(self):
        super(DegInterestChain, self).to_cpu()
        if not self.P is None: self.P.to_cpu()
        if not self.E is None: self.E.to_cpu()
        
    def to_gpu(self, device):
        super(DegInterestChain, self).to_gpu(device)
        if not self.P is None: self.P.to_gpu(device)
        if not self.E is None: self.E.to_gpu(device)
        
    def update_prediction(self, x):
        y = self.lstm1(x)
        y = self.l1(y)
        y = F.relu(y)
        self.P = F.relu(y)
        return self.P

    def calc_deg_interest(self, x):
        print('calc_deg_interest')
        if self.P is None:
            self.P = Variable(x.data, volatile='auto')
        self.E = F.sum(abs(x - self.P))/self.feature_vec_size
        return F.sigmoid(self.E)

class QNet:
    # Hyper-Parameters
    gamma = 0.99  # Discount factor
    initial_exploration = 10**3  # Initial exploratoin. original: 5x10^4
    replay_size = 32  # Replay (batch) size
    target_model_update_freq = 10**4  # Target update frequancy. original: 10^4
    data_size = 10**5  # Data size of history. original: 10^6
    hist_size = 1 #original: 4

    def __init__(self, use_gpu, enable_controller, dim):
        self.use_gpu = use_gpu
        self.num_of_actions = len(enable_controller)
        self.enable_controller = enable_controller
        self.dim = dim
        self.hidden_dim = 256
        self.predictor_error = 0
        
        print("Initializing Q-Network...")      
        self.model = QCalculationChain(self.dim*self.hist_size, self.hidden_dim, self.num_of_actions)
        if self.use_gpu >= 0:
            self.model.to_gpu()
        self.model_target = copy.deepcopy(self.model)

        self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.0001)
        self.optimizer.setup(self.model)
        
        self.Predictor = DegInterestChain(self.dim, self.hist_size, self.num_of_actions)
        if self.use_gpu >= 0:
            self.Predictor.to_gpu()
            
        self.optimizer_pred = optimizers.Adam()
        self.optimizer_pred.setup(self.Predictor)
        
        # History Data :  D=[s, a, r, s_dash, end_episode_flag]
        self.d = [np.zeros((self.data_size, self.hist_size, self.dim), dtype=np.float32),
                  np.zeros(self.data_size, dtype=np.float32),
                  np.zeros((self.data_size, 1), dtype=np.float32),
                  np.zeros((self.data_size, self.hist_size, self.dim), dtype=np.float32),
                  np.zeros((self.data_size, 1), dtype=np.bool)]
    
    def forward(self, state, action, reward, state_dash, episode_end):
        print('forward')
        num_of_batch = state.shape[0]

        q = self.model(Variable(state))  # Get Q-value

        # Generate Target Signals
        tmp = self.model_target(Variable(state_dash))  # Q(s',*)
        if self.use_gpu >= 0:
            tmp = list(map(np.max, tmp.data.get()))  # max_a Q(s',a)
        else:
            tmp = list(map(np.max, tmp.data))  # max_a Q(s',a)
        
        max_q_dash = np.asanyarray(tmp, dtype=np.float32)
        if self.use_gpu >= 0:
            target = np.asanyarray(q.data.get(), dtype=np.float32)
        else:
            # make new array
            target = np.array(q.data, dtype=np.float32)

        for i in xrange(num_of_batch):
            if not episode_end[i][0]:
                tmp_ = reward[i] + self.gamma * max_q_dash[i]
            else:
                tmp_ = reward[i]

            action_index = self.action_to_index(action[i])
            target[i, action_index] = tmp_

        # TD-error clipping
        if self.use_gpu >= 0:
            target = cuda.to_gpu(target)
        td = Variable(target) - q  # TD error
        td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)  # Avoid zero division
        td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)

        zero_val = np.zeros((self.replay_size, self.num_of_actions), dtype=np.float32)
        if self.use_gpu >= 0:
            zero_val = cuda.to_gpu(zero_val)
        zero_val = Variable(zero_val)
        loss = F.mean_squared_error(td_clip, zero_val)
        return loss, q
    
    def stock_experience(self, time,
                        state, action, reward, state_dash,
                        episode_end_flag):
        print('store')
        data_index = time % self.data_size

        if episode_end_flag is True:
            self.d[0][data_index] = state
            self.d[1][data_index] = action
            self.d[2][data_index] = reward
        else:
            self.d[0][data_index] = state
            self.d[1][data_index] = action
            self.d[2][data_index] = reward
            self.d[3][data_index] = state_dash
        self.d[4][data_index] = episode_end_flag
    
    def experience_replay(self, time):
        print('replay')
        if self.initial_exploration < time:
            # Pick up replay_size number of samples from the Data
            if time < self.data_size:  # during the first sweep of the History Data
                replay_index = np.random.randint(0, time, (self.replay_size, 1))
            else:
                replay_index = np.random.randint(0, self.data_size, (self.replay_size, 1))

            s_replay = np.ndarray(shape=(self.replay_size, self.hist_size, self.dim), dtype=np.float32)
            a_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.uint8)
            r_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.float32)
            s_dash_replay = np.ndarray(shape=(self.replay_size, self.hist_size, self.dim), dtype=np.float32)
            episode_end_replay = np.ndarray(shape=(self.replay_size, 1), dtype=np.bool)
            for i in xrange(self.replay_size):
                s_replay[i] = np.asarray(self.d[0][replay_index[i]], dtype=np.float32)
                a_replay[i] = self.d[1][replay_index[i]]
                r_replay[i] = self.d[2][replay_index[i]]
                s_dash_replay[i] = np.array(self.d[3][replay_index[i]], dtype=np.float32)
                episode_end_replay[i] = self.d[4][replay_index[i]]

            if self.use_gpu >= 0:
                s_replay = cuda.to_gpu(s_replay)
                s_dash_replay = cuda.to_gpu(s_dash_replay)

            # Gradient-based update
            self.optimizer.zero_grads()
            loss, _ = self.forward(s_replay, a_replay, r_replay, s_dash_replay, episode_end_replay)
            loss.backward()
            self.optimizer.update()
    
    def e_greedy(self, state, epsilon):
        var_s = Variable(state)
        deg_intereset = self.Predictor.calc_deg_interest(var_s[:, 0, :]).data
        self.predictor_error += self.Predictor.E
        
        print('aaaa')
        
        var_q = self.model(var_s)
        q = var_q.data
        self.Predictor.update_prediction(F.concat([F.flatten(var_s), F.flatten(var_q)]))
        
        print('bbbbb')
        
        if np.random.rand() < epsilon:
            index_action = np.random.randint(0, self.num_of_actions)
            print(" Random"),
        else:
            if self.use_gpu >= 0:
                index_action = np.argmax(q.data.get())
            else:
                index_action = np.argmax(q)
            print("#Greedy"),
        return self.index_to_action(index_action), q, deg_intereset
    
    def target_model_update(self):
        self.Predictor.zerozrads()
        self.predictor_error.backward()
        self.predictor_error.unchain_backward()
        self.predictor_error = 0
        self.optimizer_pred.uopdate()
        self.Predictor.reset_state()
        
        self.model_target = copy.deepcopy(self.model)
    
    def index_to_action(self, index_of_action):
        return self.enable_controller[index_of_action]
    
    def action_to_index(self, action):
        return self.enable_controller.index(action)