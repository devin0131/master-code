import torch
import torch.nn as nn
import torch.nn.functional as F


class feature_net(nn.Module):
    def __init__(self, args, agent_id):
        super(feature_net, self).__init__()
        self.fc1 = nn.Linear(args.obs_shape[agent_id], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class action_net(nn.Module):
    def __init__(self, args, agent_id):
        super(action_net, self).__init__()
        # self.max_action = torch.tensor(args.high_action, dtype=torch.float32)
        self.fc1 = nn.Linear(2*args.message_size+64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, args.action_shape[agent_id])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = torch.tanh(self.action_out(x))
        return actions

# define the actor network


class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.feature_net = feature_net(args, agent_id)
        self.action_net = action_net(args, agent_id)
        self.wh = nn.Linear(64, 64)
        self.wk = nn.Linear(args.message_size+64+64, args.message_size)
        self.wc = nn.Linear(64+args.message_size, args.message_size)
        self.wg = nn.Linear(64+args.message_size, args.message_size)
        self.wfg = nn.Linear(64+args.message_size, args.message_size)

    def forward(self, x, m):
        em = self.feature_net(x)
        hm = self.wh(em)
        km = torch.sigmoid(self.wk(torch.concat((em, hm, m), 1)))
        c1 = torch.concat((em, m), 1)
        cm = torch.tanh(self.wc(c1))
        gm = torch.sigmoid(self.wg(c1))
        fgm = torch.sigmoid(self.wfg(c1))
        m_out = (cm * gm) + (fgm * m)

        x = self.action_net(torch.concat((em, km * m, m_out), 1))
        return (x, m_out)


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        # for i in range(len(action)):
        #     # print(i, type(action[i]), type(self.max_action))
        #     action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value
