import numpy as np
from gymnasium.spaces.box import Box
from . import seeding
from . import simulated_annealing
from . import ball_box
from . import channel
import json
import os
import pygame
import logging

# UAC:
# h 飞行高度
# f 最大处理速度
# theta 服务角
# pos 坐标


class UAV:
    # def __init__(self, h, p, fmax, theta):
    def __init__(self):
        self.id = 0
        self.h = 0
        self.p = 0  # 发送功率
        self.fmax = 0
        self.theta = 0
        self.pos = np.zeros(2)
        self.n_asso = 0
        self.asso_smd = dict()
        self.action = np.zeros(2)
        self.reward = 0
        self.color = None
        self.size = 0.1

    def reset(self):
        self.n_asso = 0
        self.asso_smd = dict()
        self.reward = 0

    def add_asso(self, smdid):
        self.asso_smd[smdid] = 0
        self.n_asso += 1

    def distribute_asso(self):
        if self.n_asso == 0:
            return
        res = self.fmax / self.n_asso
        # print(res, self.fmax, self.n_asso)
        for key in self.asso_smd:
            self.asso_smd[key] = res

    def comp_res(self, smd_id):
        return self.asso_smd[smd_id]


# SMD:
# lamda 任务生成率
# F 任务大小
# f 任务所需计算量
# tmux 任务容忍时间
class SMD:
    # def __init__(self, lamda, F, f, p, fmax, tmax, smd_id):
    def __init__(self):
        self.id = None
        self.lamda = 0
        self.pos = np.zeros(2)
        self.F = 0  # 文件大小
        self.f = 0  # 所需计算能力
        self.p = 0  # 发送功率
        self.fmax = 0  # 本机计算能力
        self.tmax = 0
        self.b = 0
        self.asso = -1  # 选择卸载的无人机 , 如果该值==-1,则本地计算
        self.ulocal = 0
        self.color = None
        self.size = 0.05


class World:
    def __init__(self):
        self.k = 0      # 用户计算CPU消耗能量参数
        self.wt = 0     # 时间效用权重
        self.we = 0     # 能量效用权重
        self.B = 0      # 总通信带宽
        self.n_uav = 0
        self.n_smd = 0
        self.xside = 0
        self.yside = 0
        self.uavs = None
        self.smds = None
        self.ch = channel.Channel()

    @property
    def entities(self):
        return self.uavs + self.smds

    def rate(self, p, b, pos1, pos2, h):
        dist = np.sqrt(np.linalg.norm(pos1-pos2)**2 + h**2)
        return self.ch.rate(p, dist, b)

    def e_comp(self, f, t):
        return self.k * f * t

    @staticmethod
    def e_commu(p, t):
        return p * t

    def step(self):
        for uav in self.uavs:
            uav.pos += uav.action


class Scenario:
    def __init__(self):
        dir = os.path.dirname(__file__)
        with open(dir + "/umec.json") as json_file:
            self.data = json.load(json_file)
        self.optimize = simulated_annealing

    def make_world(self, n_uav, n_smd):
        world = World()
        world.n_uav = n_uav
        world.n_smd = n_smd
        world.uavs = [UAV() for _ in range(n_uav)]
        world.smds = [SMD() for _ in range(n_smd)]
        # 根据配置文件初始化 uav和smd
        data = self.data
        for uavid, uav in enumerate(world.uavs):
            uavdata = data["uavs"][data["uav_index"][uavid]]
            uav.id = uavid
            uav.h = uavdata['h']
            uav.p = uavdata['p']
            uav.fmax = uavdata['fmax']
            uav.theta = uavdata['theta']
            uav.pos = np.array(uavdata['pos'], dtype=np.float32)
        for smdid, smd in enumerate(world.smds):
            smddata = data["smds"][data["smd_index"][smdid]]
            smd.id = smdid
            smd.pos = np.array(smddata['pos'], dtype=np.float32)
            smd.lamda = smddata['lamda']
            smd.F = smddata['F']
            smd.f = smddata['f']
            smd.p = smddata['p']
            smd.fmax = smddata['fmax']
            smd.tmax = smddata['tmax']

        # 根据配置文件配置world
        world.k = data['world']['k']
        world.wt = data['world']['wt']
        world.we = data['world']['we']
        world.B = data['world']['B']
        world.xside = data['world']['xside']
        world.yside = data['world']['yside']
        return world

    # 计算SMD的本地计算效用
    def local_utility(self, world: World, smd: SMD) -> float:
        tlocal = smd.f / smd.fmax
        ulocal = smd.lamda * (- world.we * world.e_comp(smd.fmax, tlocal)) + \
            world.wt * (smd.tmax - tlocal)
        return ulocal

    # 计算SMD的卸载效用
    def off_utility(self, world: World, smd: SMD):
        fuav = world.uavs[smd.asso].comp_res(smd.id)
        tcomp = smd.f / fuav
        tcommu = smd.F / \
            world.rate(smd.p, smd.b, smd.pos,
                       world.uavs[smd.asso].pos, world.uavs[smd.asso].h)
        uoff = smd.lamda * ((world.we * (-world.e_comp(fuav, tcomp) - world.e_commu(
            smd.p, tcommu))) + (world.wt * (smd.tmax - tcommu - tcomp)))
        # logging.debug("smd:{}, uoff:{}, tcomp:{}, tcommu:{}, fuav:{}".format(smd.id, uoff, tcomp, tcommu, fuav))
        return uoff

    # solution 应该包括smd的卸载选择，以及带宽
    # solution = (asso, band)
    # asso = (len(smd),)
    # band = (len(smd),)
    # st.
    # asso[*] = {-1, ..., max(uavs)-1}, 整型
    # band = {b1,b2,b3,...,bn}, sum(band) = b
    def system_utility(self, world: World, solution):
        asso, band = solution
        utility = 0
        for smd in world.smds:
            smd.asso = asso[smd.id]
            world.uavs[smd.asso].add_asso(smd.id)
            smd.b = band[smd.id]
        for uav in world.uavs:
            uav.distribute_asso()
        for smd in world.smds:
            if smd.asso == -1:
                utility += self.local_utility(world, smd)
            else:
                utility += self.off_utility(world, smd)
        for uav in world.uavs:
            uav.reset()
        return utility

    def solution_result(self, world: World, solution):
        asso, band = solution
        for smd in world.smds:
            smd.asso = asso[smd.id]
            world.uavs[smd.asso].add_asso(smd.id)
            smd.b = band[smd.id]
        for uav in world.uavs:
            uav.distribute_asso()
        for smd in world.smds:
            if smd.asso == -1:
                logging.info("本地效用:{}".format_map(
                    self.local_utility(world, smd)))
            else:
                fuav = world.uavs[smd.asso].comp_res(smd.id)
                tcomp = smd.f / fuav
                tcommu = smd.F / \
                    world.rate(smd.p, smd.b, smd.pos,
                               world.uavs[smd.asso].pos, world.uavs[smd.asso].h)
                uoff = smd.lamda * ((world.we * (-world.e_comp(fuav, tcomp) - world.e_commu(
                    smd.p, tcommu))) + (world.wt * (smd.tmax - tcommu - tcomp)))
                logging.info("卸载效用:{}, fs:{}, t计算:{}, t通信:{}, t效用:{}, e效用:{}".format(uoff, fuav, tcomp, tcommu, smd.tmax - tcommu - tcomp, -world.e_comp(fuav, tcomp) - world.e_commu(
                    smd.p, tcommu)))
        for uav in world.uavs:
            uav.reset()

    # 系统花费 = -系统效用

    def system_cost(self, world: World, solution):
        return -self.system_utility(world, solution)

    def get_avalible_solution(self, world: World, disable_loca: bool = False):
        if disable_loca is True:
            asso = np.random.randint(0, world.n_uav, world.n_smd)
        else:
            asso = np.random.randint(-1, world.n_uav, world.n_smd)
        band = np.zeros(world.n_smd, dtype=int)
        n_box = asso[asso != -1].size
        if n_box == 0:
            return (asso, band)
        else:
            result = ball_box.random_distribution(world.B, n_box)
            if result is None:
                raise Exception("Band not avaliable")
                return None
            else:
                index = 0
                for smdid in range(world.n_smd):
                    if asso[smdid] != -1:
                        band[smdid] = result[index]
                        index += 1
                    else:
                        band[smdid] = 0
                return (asso, band)

    # 重置坐标
    # world 重置world中的坐标
    # 重置时使用的种子
    # 是否使用配置文件中的坐标reset

    def reset_world(self, world: World, np_random, config_pos=False):

        for _, agent in enumerate(world.uavs):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for _, landmark in enumerate(world.smds):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # 无人机只会初始化在一开始的位置
        for i, uav in enumerate(world.uavs):
            uav.pos = np.array(self.data['uavs'][i]['pos'], dtype=np.float32)
            uav.reset()
        if config_pos is True:
            for i, smd in enumerate(world.smds):
                smd.pos = np.array(
                    self.data['smds'][i]['pos'], dtype=np.float32)
        else:
            for i, smd in enumerate(world.smds):
                smd.pos = np_random.uniform(-world.xside/2, world.yside/2, 2)

        ##
        # print("reset :")
        # for uav in world.uavs:
        #     print("uav:{}, pos:{}".format(uav.id, uav.pos))
        # for smd in world.smds:
        #     print("smd:{}, pos:{}".format(smd.id, smd.pos))

    def local_reward(self, agent, world):
        rew = 0
        # for smdid in agent.asso_smd:
        #     rew += self.off_utility(world, world.smds[smdid])
        # dist = np.linalg.norm(agent.pos - world.smds[agent.id].pos)
        # rew = -dist

        def d(pos1, pos2, h):
            return np.sqrt(np.linalg.norm(pos1-pos2)**2 + h**2)
        dist = [world.ch.rate(smd.p, d(smd.pos, agent.pos, agent.h), 1)
                for smd in world.smds]
        rew = np.sum(np.array(dist))
        return rew

    def global_reward(self, world: World):
        # def cost_function(solution):
        #     return self.system_cost(world, solution)
        #
        # def random_solution():
        #     return self.get_avalible_solution(world, True)
        # rew = 0
        # best_solution, best_cost = self.optimize.simulated_annealing(
        #     cost_function, random_solution, 100, 0.95, 10)
        # # for smd in world.smds:
        # #     if smd.asso == 100:
        # #         rew += self.cal_utility(world, smd)
        # # logging.info("solution:{}, cost:{}".format(best_solution, best_cost))
        # self.solution_result(world, best_solution)
        # rew = - best_cost
        rew = 0

        for smd in world.smds:
            # dist = [np.linalg.norm(smd.pos - uav.pos) for uav in world.uavs]
            def d(pos1, pos2, h):
                return np.sqrt(np.linalg.norm(pos1-pos2)**2 + h**2)
            rate = [world.ch.rate(smd.p, d(smd.pos, uav.pos, uav.h), 1)
                    for uav in world.uavs]
            rew += np.sum(np.array(rate))
        return rew

    def observation(self, uav: UAV, world: World):
        # 每个用户的 lamda，任务大小，所需计算量，时间约束， 最大计算能力， 坐标 1 + 1 + 1 + 1 + 1 + 2 = 7
        # 无人机自己的坐标 高度，最大计算资源 2 + 1 + 1 = 4
        ob = np.zeros(world.n_smd * 7)
        for i, smd in enumerate(world.smds):
            ob[i*7:(i+1)*7] = np.concatenate((np.array([smd.lamda,
                                                        smd.F, smd.f, smd.tmax, smd.fmax]), smd.pos))
        ob = np.concatenate((ob, uav.pos, np.array([uav.h, uav.fmax])))

        return ob


class env:
    def __init__(self, n_uav=3, n_smd=3, local_ratio=None, max_cycles=100):
        self.uav_selection = 0
        self.renderOn = True
        self.steps = 0

        self.max_cycles = max_cycles
        self.scenario = Scenario()
        self.seed()
        self.world = self.scenario.make_world(n_uav, n_smd)
        self.scenario.reset_world(self.world, self.np_random)

        self.width = 700
        self.height = 700
        pygame.init()
        self.screen = pygame.Surface([self.width, self.height])
        self.screen = pygame.display.set_mode(self.screen.get_size())
        self.rewards = {uavid: 0.0 for uavid in self.world.uavs}
        self.uavs = [uav.id for uav in self.world.uavs]
        self.agents = self.uavs
        self.truncations = {uavid: False for uavid in self.uavs}
        self.local_ratio = local_ratio
        self.max_num_agents = n_uav
        self.num_agents = n_uav
        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0
        for uav in self.world.uavs:
            space_dim = 2

            obs_dim = len(self.scenario.observation(uav, self.world))
            state_dim += obs_dim
            self.action_spaces[uav.id] = Box(
                low=-8, high=8, shape=(space_dim,)
            )
            self.observation_spaces[uav.id] = Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(obs_dim,),
                dtype=np.float32,
            )

        self.state_space = Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(state_dim,),
            dtype=np.float32,
        )
        self.current_actions = [None] * self.num_agents

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        return self.scenario.observation(
            self.world.uavs[agent], self.world
        ).astype(np.float32)

    def state(self):
        states = tuple(
            self.scenario.observation(
                uav, self.world
            ).astype(np.float32)
            for uav in self.world.uavs
        )
        return np.concatenate(states, axis=None)

    def reset(self):
        self.scenario.reset_world(self.world, self.np_random, False)
        self.truncations = {uavid: False for uavid in self.agents}
        self.rewards = {name: 0.0 for name in self.agents}
        self.uav_selection = 0
        self.steps = 0

        self.current_actions = [None] * self.num_agents

    # 在这个函数运行世界一步的操作
    # calculate reward
    def _execute_world_step(self):
        for uav in self.world.uavs:
            # 可以的话对action进行必要的检查
            self._set_action(self.current_actions[uav.id], uav)
        self.world.step()

        global_reward = 0.0
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))

        for uav in self.world.uavs:
            uav_reward = float(self.scenario.local_reward(uav, self.world))
            if self.local_ratio is not None:
                reward = (
                    global_reward * (1 - self.local_ratio)
                    + uav_reward * self.local_ratio
                )
            else:
                reward = uav_reward

            self.rewards[uav.id] = reward
        # for uav in self.world.uavs:
        #     self.rewards[uav.id] = 1

    # 为agent添加运行action的动作

    def _set_action(self, action, uav):
        uav.action = action.copy()

    # action: 包含每个无人机的动作，但是按照env的写法，这个action是某一个无人机的action

    def step(self, action):
        self.current_actions[self.uav_selection] = action
        self.uav_selection = (self.uav_selection + 1) % self.world.n_uav
        if self.uav_selection == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True
        self.render()

    def render(self):
        if self.renderOn:
            self.draw()
            pygame.display.flip()

    def draw(self):
        # clear screen
        self.screen.fill((255, 255, 255))

        # update bounds to center around agent
        all_poses = [entity.pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses)))

        # update geometry and text positions
        # text_line = 0
        for e, entity in enumerate(self.world.entities):
            # geometry
            x, y = entity.pos
            y *= (
                -1
            )  # this makes the display mimic the old pyglet setup (ie. flips image)
            x = (
                (x / cam_range) * self.width // 2 * 0.9
            )  # the .9 is just to keep entities from appearing "too" out-of-bounds
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2
            pygame.draw.circle(
                self.screen, entity.color * 200, (x, y), entity.size * 350
            )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
            pygame.draw.circle(
                self.screen, (0, 0, 0), (x, y), entity.size * 350, 1
            )  # borders
            assert (
                0 < x < self.width and 0 < y < self.height
            ), f"Coordinates {(x, y)} are out of bounds."

            # text
            # if isinstance(entity, Agent):
            #     if entity.silent:
            #         continue
            #     if np.all(entity.state.c == 0):
            #         word = "_"
            #     elif self.continuous_actions:
            #         word = (
            #             "[" +
            #             ",".join(
            #                 [f"{comm:.2f}" for comm in entity.state.c]) + "]"
            #         )
            #     else:
            #         word = alphabet[np.argmax(entity.state.c)]
            #
            #     message = entity.name + " sends " + word + "   "
            #     message_x_pos = self.width * 0.05
            #     message_y_pos = self.height * 0.95 - \
            #         (self.height * 0.05 * text_line)
            #     self.game_font.render_to(
            #         self.screen, (message_x_pos,
            #                       message_y_pos), message, (0, 0, 0)
            #     )
            #     text_line += 1

    def close(self):
        if self.renderOn:
            pygame.event.pump()
            pygame.display.quit()
            self.renderOn = False
