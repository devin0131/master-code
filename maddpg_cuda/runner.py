from tqdm import tqdm
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from .replay_buffer import Buffer
from .agent import Agent


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.buffer = Buffer(args)
        self.agents = self._init_agents()
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        returns = []
        for time_step in tqdm(range(self.args.time_steps)):
            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.evaluate())
                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.xlabel('episode * ' +
                           str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')

            # 跑一个回合
            self.env.reset()
            r = np.zeros(self.args.n_agents,dtype=np.float32)
            done = np.zeros(self.args.n_agents, dtype=bool)
            s = self.env.state().reshape(self.args.n_agents, -1)
            while (True):
                # reset the environment
                self.env.render_mode = None
                u = []
                # actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(
                            s[agent_id], self.noise, self.epsilon)
                        self.env.step(action.astype(np.float32))
                        # print(self.env.terminations)
                        # print(self.env.truncations)
                        u.append(action)
                        # actions.append(action)
                # for i in range(self.args.n_agents, self.args.n_players):
                #     actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                ###################################################
                # 获取下一个状态、奖励、是否结束、info
                for i in range(self.args.n_agents):
                    r[i] = self.env.rewards[self.env.agents[i]]
                    done[i] = self.env.truncations[self.env.agents[i]]
                # s_next, r, done, info = self.env.step(actions)
                s_next = self.env.state().reshape(self.args.n_agents, -1)
                ###################################################
                self.buffer.store_episode(
                    s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
                s = s_next
                if self.buffer.current_size >= self.args.batch_size:
                    transitions = self.buffer.sample(self.args.batch_size)
                    for agent in self.agents:
                        other_agents = self.agents.copy()
                        other_agents.remove(agent)
                        agent.learn(transitions, other_agents)
                self.noise = max(0.05, self.noise - (self.args.noise_rate - 0.05)/self.args.time_steps-30)
                self.epsilon = max(0.05, self.epsilon - (self.args.noise_rate - 0.05)/ self.args.time_steps-30)
                # np.save(self.save_path + '/returns.pkl', returns)
                if done.any():
                    break

    def evaluate(self):
        returns = []
        r = np.zeros(self.args.n_agents)
        self.env.render_mode = "human"
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            self.env.reset()
            s = self.env.state().reshape(self.args.n_agents, -1)
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        self.env.step(action.astype(np.float32))
                for i in range(self.args.n_agents):
                    r[i] = self.env.rewards[self.env.agents[i]]
                s_next = self.env.state().reshape(self.args.n_agents, -1)
                rewards += r.sum()
                s = s_next
            returns.append(rewards)
            print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes
