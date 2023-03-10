import pettingzoo.mpe as mpe
from types import MethodType
import numpy as np

def new_local_reward(self, agent, world):
    idx = int(agent.name[-1])
    rew = -np.linalg.norm(agent.state.p_pos - world.landmarks[idx].state.p_pos)
    # if agent.collide:
    #     for a in world.agents:
    #         if self.is_collision(a, agent):
    #             rew -= 1
    return rew
        
def make_env(args):
    origin_mpe = mpe.simple_spread_v2.env(max_cycles=args.evaluate_episode_len, continuous_actions=True, render_mode='human', local_ratio=1)
    for agent in origin_mpe.unwrapped.world.agents:
        agent.collide = True
                

    origin_mpe.unwrapped.scenario.reward = MethodType(new_local_reward, origin_mpe.unwrapped.scenario)
    origin_mpe.reset()
    return origin_mpe
origin_mpe = make_env
