import numpy as np
from gymnasium.spaces.box import Box
import seeding
import json

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

    def add_asso(self, smdid):
        self.asso_smd[smdid] = 0
        self.n_asso += 1

    def distribute_asso(self):
        self.f = dict()
        res = self.fmax / self.n_asso
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
        self.id = 0
        self.lamda = 0
        self.pos = np.zeros(2)
        self.F = 0  # 文件大小
        self.f = 0  # 所需计算能力
        self.p = 0  # 发送功率
        self.fmax = 0  # 本机计算能力
        self.tmax = 0
        self.b = 0
        self.asso = 100  # 选择卸载的无人机 , 如果该值==100,则本地计算
        self.ulocal = 0


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

    @staticmethod
    def dist(pos1, pos2):
        return np.linalg.norm(pos1-pos2)

    def sinr(self, pos1, pos2):
        pass

    def rate(self, b, pos1, pos2, h):
        pass

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
        with open("umec.json") as json_file:
            self.data = json.load(json_file)

        

    def make_world(self, n_uav, n_smd):
        world = World()
        world.n_uav = n_uav
        world.n_smd = n_smd
        world.uavs = [UAV() for _ in range(n_uav)]
        world.smds = [SMD() for _ in range(n_smd)]
        ## 根据配置文件初始化 uav和smd
        data = self.data
        for uavid, uav in enumerate(world.uavs):
            uav.id = uavid
            uav.h = data["uavs"][uavid]['h']
            uav.p = data["uavs"][uavid]['p']
            uav.fmax = data["uavs"][uavid]['fmax']
            uav.theta = data["uavs"][uavid]['theta']
            uav.pos = np.array(data["uavs"][uavid]['pos']).copy()
        for smdid, smd in enumerate(world.smds):
            smd.id = smdid
            smd.pos = data["smds"][smdid]['pos']
            smd.lamda = data["smds"][smdid]['lamda']
            smd.F = data["smds"][smdid]['F']
            smd.f = data["smds"][smdid]['f']
            smd.p = data["smds"][smdid]['p']
            smd.fmax = data["smds"][smdid]['fmax']
            smd.tmax = data["smds"][smdid]['tmax']

        ## 根据配置文件配置world
        world.k = data['world']['k']
        world.wt = data['world']['wt']
        world.we = data['world']['we']
        world.B = data['world']['B']
        world.xside = data['world']['xside']
        world.yside = data['world']['yside']
        return world

    def local_utility(self, world: World, smd: SMD) -> float:
        tlocal = smd.f / smd.fmax
        ulocal = (- world.we * world.ecomp(smd.fmax, tlocal)) + \
            world.wt * (smd.tmax - tlocal)
        return ulocal

    def off_utility(self, world: World, smd: SMD):
        fuav = world.uavs[smd.asso].comp_res[smd.id]
        tcomp = smd.f / fuav
        tcommu = smd.F / world.rate(smd.b, smd.pos, world.uavs[smd.asso].pos, world.uavs[smd.asso].h)
        uoff = (world.we * (-world.ecomp(fuav, tcomp) - world.e_commu(smd.p,
                tcommu))) + (world.wt * (smd.tmax - tcommu - tcomp))
        return uoff

    ## 重置坐标
    # world 重置world中的坐标
    # 重置时使用的种子
    # 是否使用配置文件中的坐标reset
    def reset_world(self, world: World, np_random, config_pos=False):
        ## 无人机只会初始化在一开始的位置
        for i, uav in enumerate(world.uavs):
            uav.pos = np.array(self.data['uavs'][i]['pos'])
        if config_pos is True:
            for i, smd in enumerate(world.smds):
                smd.pos = np.array(self.data['smds'][i]['pos'])
        else:
            for i, smd in enumerate(world.smds):
                smd.pos = np_random.uniform(-world.xside/2, world.yside/2, 2)

            
            


    def local_reward(self, agent, world):
        rew = 0
        # for smdid in agent.asso_smd:
        #     rew += self.off_utility(world, world.smds[smdid])
        dist = np.linalg.norm(agent.pos - world.smds[agent.id].pos)
        rew = -dist
        return rew

    def global_reward(self, world: World):
        rew = 0
        # for smd in world.smds:
        #     if smd.asso == 100:
        #         rew += self.cal_utility(world, smd)
        return rew

    def observation(self, uav: UAV, world: World):
        ## 每个用户的 lamda，任务大小，所需计算量，时间约束， 最大计算能力， 坐标 1 + 1 + 1 + 1 + 1 + 2 = 7
        ## 无人机自己的坐标 高度，最大计算资源 2 + 1 + 1 = 4
        ob = np.zeros(world.n_smd * 7)
        for i, smd in enumerate(world.smds):
            ob[i*7:(i+1)*7] = np.concatenate((np.array([smd.lamda, smd.F, smd.f, smd.tmax, smd.fmax]), smd.pos))
        ob = np.concatenate((ob,uav.pos, np.array([uav.h, uav.fmax])))
        
        return ob


class env:
    def __init__(self, n_uav = 3, n_smd = 3, local_ratio = None, max_cycles = 25):
        self.uav_selection = 0

        self.steps = 0

        self.max_cycles = max_cycles
        self.scenario = Scenario()
        self.seed()
        self.world = self.scenario.make_world(n_uav, n_smd)
        self.scenario.reset_world(self.world, self.np_random)

        self.rewards = {uavid : 0.0 for uavid in self.world.uavs}
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
                low=-1, high=1, shape=(space_dim,)
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
            ## 可以的话对action进行必要的检查
            self._set_action(self.current_actions[uav.id],uav)
        self.world.step()

        global_reward = 0.0
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_utility(self.world))
        
        for uav in self.world.uavs:
            uav_reward = float(self.scenario.local_utility(uav, self.world))
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
        
