import random


def distribute_balls(m, n, result, path):
    if m == 0 and all([p > 0 for p in path]):
        result.append(path)
        return
    if n == 0 or m == 0:
        return
    distribute_balls(m, n-1, result, path + [0])
    for i in range(1, m-n+2):
        distribute_balls(m-i, n-1, result, path + [i])
    return


def random_distribution(m, n):
    if m < n:
        return None
    result = []
    distribute_balls(m, n, result, [])
    if result:
        return random.choice(result)
    else:
        return None


if __name__ == "__main__":

    m = 100
    n = 3

    print("所有可能情况：")
    result = []
    distribute_balls(m, n, result, [])
    for r in result:
        print(r)

    print("随机输出一种情况：", random_distribution(m, n))
