'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-05-28 14:48:47
Version: v1
File: 
Brief: 
'''
"""
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
        
        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
"""
# coding: utf8
'''
https://blog.csdn.net/qq_32892383/article/details/89576003
 为了能够有效控制倒立摆首先应建立一个控制模型。明显的，这个控制模型的输入应该是当前倒立摆的状态（observation）而输出为对当前状态做出的决策动作（action）。从前面的知识我们了解到决定倒立摆状态的observation是一个四维向量，包含小车位置、杆子夹角、小车速度及角变化率，如果对这个向量求它的加权和，那么就可以根据加权和值的符号来决定采取的动作（action），用sigmoid函数将这个问题转化为二分类问题，从而可以建立一个简单的控制模型。其模型如下图所示
 上图的实际功能与神经网络有几分相似，但比神经网络要简单得多。通过加入四个权值，我们可以通过改变权重值来改变决策（policy），即有加权和H s u m = ω 1 x + ω 2 θ + ω 3 x ˙ + ω 4 θ ˙ + b 
 的符号为正判定输出为1，否则为0。为了得到一组较好的权值从而有效控制倒立摆，我们可以采用爬山算法（hill climbing algorithm）进行学习优化。爬山算法是一种启发式方法，是对深度优先搜索的一种改进，它利用反馈信息帮助生成解的决策。
爬山算法的基本思路是每次迭代时给当前取得的最优权重加上一组随机值，如果加上这组值使得有效控制倒立摆的持续时间变长了那么就更新它为最优权重，如果没有得到改善就保持原来的值不变，直到迭代结束。在迭代过程中，模型的参数不断得到优化，最终得到一组最优的权值作为控制模型的解。其代码如下：
'''
import numpy as np
import gym
import time

def get_action(weights, observation):# 根据权值对当前状态做出决策
    wxb = np.dot(weights[:4], observation) + weights[4] # 计算加权和
    if wxb >= 0:# 加权和大于0时选取动作1，否则选取0
        return 1
    else:
        return 0

def get_sum_reward_by_weights(env, weights):
# 测试不同权值的控制模型有效控制的持续时间（或奖励）
    observation = env.reset() # 重置初始状态
    observation = observation[0]
    sum_reward = 0 # 记录总的奖励
    for t in range(1000):
        # time.sleep(0.01)
        # env.render()
        action = get_action(weights, observation) # 获取当前权值下的决策动作
        observation, reward, done, _, _ = env.step(action)# 执行动作并获取这一动作下的下一时间步长状态
        sum_reward += reward
        print(f'===')
        print(f'action: {action}')
        print(f'observation: {observation}')
        print(f'reward: {reward}')
        print(f'done: {done}')
        print(f'sum_reward: {sum_reward}')
        if done:# 如若游戏结束，返回
            break
    return sum_reward


def get_weights_by_random_guess():
# 选取随机猜测的5个随机权值
    return np.random.rand(5)

def get_weights_by_hill_climbing(best_weights):
# 通过爬山算法选取权值（在当前最好权值上加入随机值）
    return best_weights + np.random.normal(0, 0.1, 5)

def get_best_result(algo="random_guess"):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    np.random.seed(10)
    best_reward = 0 # 初始最佳奖励
    best_weights = np.random.rand(5) # 初始权值为随机取值

    for iter in range(10000):# 迭代10000次
        cur_weights = None

        if algo == "hill_climbing": # 选取动作决策的算法 
            # print(best_weights)
            cur_weights = get_weights_by_hill_climbing(best_weights)
        else: # 若为随机猜测算法，则选取随机权值
            cur_weights = get_weights_by_random_guess()
        print(f'cur_weights: {cur_weights}')
		# 获取当前权值的模型控制的奖励和
        cur_sum_reward = get_sum_reward_by_weights(env, cur_weights)

        # print(cur_sum_reward, cur_weights)
		# 更新当前最优权值
        if cur_sum_reward > best_reward:
            best_reward = cur_sum_reward
            best_weights = cur_weights
		# 达到最佳奖励阈值后结束
        if best_reward >= 200:
            break
    print(f'==== the final result:\n')
    print(iter, best_reward, best_weights)
    return best_reward, best_weights

# 程序从这里开始执行
print(get_best_result("hill_climbing")) # 调用爬山算法寻优并输出结果 

# env = gym.make("CartPole-v0")
# get_sum_reward_by_weights(env, [0.22479665, 0.19806286, 0.76053071, 0.16911084, 0.08833981])
