import threading
import numpy as np
import torch


class Buffer:
    def __init__(self, args):
        self.size = args.buffer_size
        self.args = args
        # memory management
        self.current_size = 0
        # create the buffer to store info
        self.buffer = dict()
        if torch.cuda.is_available():
            self.device = torch.device(0)
        else:
            self.device = torch.device('cpu')
        # 每一个agent都有o u r o_next， 有固定的size * shape
        for i in range(self.args.n_agents):
            # self.buffer['o_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
            # self.buffer['u_%d' % i] = np.empty([self.size, self.args.action_shape[i]])
            # self.buffer['r_%d' % i] = np.empty([self.size])
            # self.buffer['o_next_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
            self.buffer['o_%d' % i] = torch.empty([self.size, self.args.obs_shape[i]]).to(self.device)
            self.buffer['u_%d' % i] = torch.empty([self.size, self.args.action_shape[i]]).to(self.device)
            self.buffer['r_%d' % i] = torch.empty([self.size]).to(self.device)
            self.buffer['o_next_%d' % i] = torch.empty([self.size, self.args.obs_shape[i]]).to(self.device)
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    # o u r o_next 需要以二维数组的形式存储
    def store_episode(self, o, u, r, o_next, m=None):
        idxs = self._get_storage_idx(inc=1)  # 以transition的形式存，每次只存一条经验
        for i in range(self.args.n_agents):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = torch.from_numpy(o[i]).to(self.device)
                self.buffer['u_%d' % i][idxs] = torch.from_numpy(u[i]).to(self.device)
                self.buffer['r_%d' % i][idxs] = torch.as_tensor(r[i])
                self.buffer['o_next_%d' % i][idxs] = torch.from_numpy(o_next[i]).to(self.device)
    
    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx
