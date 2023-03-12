import math
import random


def simulated_annealing(cost_function, initial_solution, temperature, cooling_rate, stopping_temperature):
    """
    模拟退火算法函数

    参数：
    cost_function: 代价函数，它将计算给定解决方案的代价。
    initial_solution: 初始解决方案。
    temperature: 初始温度。
    cooling_rate: 降温速率。
    stopping_temperature: 终止温度。

    返回：
    最优解决方案和对应的代价。
    """
    current_solution = initial_solution
    best_solution = current_solution
    current_cost = cost_function(current_solution)
    best_cost = current_cost

    while temperature > stopping_temperature:
        # 生成随机解决方案
        candidate_solution = current_solution + random.uniform(-1, 1)
        candidate_cost = cost_function(candidate_solution)

        # 计算接受概率
        delta_cost = candidate_cost - current_cost
        acceptance_probability = math.exp(-delta_cost / temperature)

        # 决定是否接受新的解决方案
        if acceptance_probability > random.random():
            current_solution = candidate_solution
            current_cost = candidate_cost

        # 更新最优解决方案
        if current_cost < best_cost:
            best_solution = current_solution
            best_cost = current_cost

        # 降温
        temperature *= cooling_rate

    return best_solution, best_cost
