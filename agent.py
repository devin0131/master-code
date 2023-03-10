import numpy as np
import torch
import os
# from maddpg.maddpg import MADDPG
from mdmaddpg.maddpg import MADDPG


class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.low_action = args.low_action[agent_id]
        self.high_action = args.high_action[agent_id]
        self.policy = MADDPG(args, agent_id)

    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(self.low_action, self.high_action, self.args.action_shape[self.agent_id])
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            pi = self.policy.actor_network(inputs).squeeze(0)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            noise = noise_rate * (self.high_action - self.low_action) * np.random.randn(*u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, self.low_action, self.high_action)
        return u.copy()

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

