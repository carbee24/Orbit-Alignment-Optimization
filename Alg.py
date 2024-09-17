import math
import numpy as np
from random import random
import matplotlib.pyplot as plt
import pickle
import copy
import ray
import torch
from torch.utils.data import DataLoader

def func(x, y):  # 函数优化问题
    res = 4 * x ** 2 - 2.1 * x ** 4 + x ** 6 / 3 + x * y - 4 * y ** 2 + 4 * y ** 4
    return res


# x为公式里的x1,y为公式里面的x2
class SA:
    def __init__(self, env, iter=100, T0=0.01, Tf=1e-6, alpha=0.5):
        self.env = env
        self.iter = iter  # 内循环迭代次数,即为L =100
        self.alpha = alpha  # 降温系数，alpha=0.99
        self.T0 = T0  # 初始温度T0为100
        self.Tf = Tf  # 温度终值Tf为0.01
        self.T = T0  # 当前温度
        self.x = [random() * 11 - 5 for i in range(iter)]  # 随机生成100个x的值
        self.y = [random() * 11 - 5 for i in range(iter)]  # 随机生成100个y的值
        self.most_best = []
        """
        random()这个函数取0到1之间的小数
        如果你要取0-10之间的整数（包括0和10）就写成 (int)random()*11就可以了，11乘以零点多的数最大是10点多，最小是0点多
        该实例中x1和x2的绝对值不超过5（包含整数5和-5），（random() * 11 -5）的结果是-6到6之间的任意值（不包括-6和6）
        （random() * 10 -5）的结果是-5到5之间的任意值（不包括-5和5），所有先乘以11，取-6到6之间的值，产生新解过程中，用一个if条件语句把-5到5之间（包括整数5和-5）的筛选出来。
        """
        self.history = {'f': [], 'T': []}

    def generate_new(self, x, y):  # 扰动产生新解的过程
        while True:
            x_new = x + self.T * (random() - random())
            y_new = y + self.T * (random() - random())
            if (-5 <= x_new <= 5) & (-5 <= y_new <= 5):
                break  # 重复得到新解，直到产生的新解满足约束条件
        return x_new, y_new

    def generate_new_actions(self, action, ratio):
        # 扰动产生新解的过程
        while True:
            # # x_new = x + self.T * (random() - random())
            # # y_new = y + self.T * (random() - random())
            # if (-5 <= x_new <= 5) & (-5 <= y_new <= 5):
            #     break  # 重复得到新解，直到产生的新解满足约束条件
            # ran = np.random.rand(3, 3)
            # ran2 = np.random.rand(3, 3)
            new_action = action + self.env.action_bound * (2 * np.random.rand(6, 3) - 1) * ratio
            new_action_abs = abs(new_action)
            if np.all(new_action_abs <= self.env.action_bound * ratio):
                break
        return new_action

    def Metrospolis(self, f, f_new):  # Metropolis准则
        if f_new <= f:
            return 1
        else:
            # print(f - f_new)
            p = math.exp((f - f_new) / self.T)
            if random() < p:
                return 1
            else:
                return 0

    # def best(self):  # 获取最优目标函数值
    #     f_list = []  # f_list数组保存每次迭代之后的值
    #     for i in range(self.iter):
    #         f = self.func(self.x[i], self.y[i])
    #         f_list.append(f)
    #     f_best = min(f_list)
    #
    #     idx = f_list.index(f_best)
    #     return f_best, idx  # f_best,idx分别为在该温度下，迭代L次之后目标函数的最优解和最优解的下标

    def run(self):
        # best_body_state = None
        # f_best = 0
        init_body_state, init_D = self.env.reset()
        best_body_state = init_body_state
        f_best = init_D
        if not self.env.terminal_yes_or_not(.01):
            init_T = self.T
            terminal = False
            while True:
                # count = 0

                if terminal:
                    break
                # f_best = 0
                # count = 0
                # 外循环迭代，当前温度小于终止温度的阈值
                self.T = init_T
                SA_done = False
                while self.T > self.Tf:
                    SA_done = self.env.terminal_yes_or_not(.1)
                    print("First SA_done: ", SA_done)
                    terminal = SA_done
                    if SA_done:
                        break
                    action = random() * self.env.action_bound
                    # 内循环迭代100次
                    for i in range(self.iter):
                        action = self.generate_new_actions(action, 1) # 产生新解
                        body_state, D_ = self.env.step(action)  # 产生新值
                        if self.Metrospolis(f_best, D_):  # 判断是否接受新值
                            best_body_state = body_state
                            f_best = D_

                        self.env.go_back(action)
                    # 迭代L次记录在该温度下最优解
                    # ft, _ = self.best()
                    # self.history['f'].append(fbest)
                    # self.history['T'].append(self.T)
                    # 温度按照一定的比例下降（冷却）
                    self.env.update_stands_state(best_body_state)
                    self.T = self.T * self.alpha
                    # count += 1
                    # print("Results of ep{}".format(count))
                    print("SA First Round")
                    # print("Now Body State: ", best_body_state)
                    print("Now Fitness: ", f_best)
                    # 得到最优解
                # f_best, idx = self.best()
                # count += 1
                # print("Results of ep{}".format(count))
                # print("Now Body State: ", best_body_state)
                # print("Now Fitness: ", f_best)
            # print(f"F={f_best}, x={best_solution['x']}, y={best_solution['x']}")
            #     self.env.update_stands_state(best_body_state)
        # if not self.env.terminal_yes_or_not(.01):
        print("SA end!!!!")
        if not self.env.terminal_yes_or_not(.01):
            # self.alpha = 0.8
            self.iter = 50
            terminal = False
            while True:
                # count = 0

                if terminal:
                    break
                # f_best = 0
                # count = 0
                # 外循环迭代，当前温度小于终止温度的阈值
                # self.T = init_T
                SA_done = False
                while self.T > self.Tf:
                    # SA_done = False
                    terminal = SA_done
                    if SA_done:
                        break

                    action = np.random.rand(6, 3) * self.env.action_bound
                    # 内循环迭代100次
                    for i in range(self.iter):
                        action = self.generate_new_actions(action, 1)  # 产生新解
                        body_state, D_ = self.env.step(action)  # 产生新值
                        if self.Metrospolis(f_best, D_):  # 判断是否接受新值
                            best_body_state = body_state
                            f_best = D_
                        SA_done = self.env.terminal_yes_or_not(.01)
                        if SA_done:
                            break
                        self.env.go_back(action)
                    # 迭代L次记录在该温度下最优解
                    # ft, _ = self.best()
                    # self.history['f'].append(fbest)
                    # self.history['T'].append(self.T)
                    # 温度按照一定的比例下降（冷却）
                    # self.env.update_stands_state(best_body_state)
                    self.T = self.T * self.alpha
                    # count += 1
                    # print("Results of ep{}".format(count))
                    print("SA Second Round")
                    # print("Now Body State: ", best_body_state)
                    print("Now Fitness: ", f_best)
                    # 得到最优解
                # f_best, idx = self.best()
                # count += 1
                # print("Results of ep{}".format(count))
                # print("Now Body State: ", best_body_state)
                # print("Now Fitness: ", f_best)
            # print(f"F={f_best}, x={best_solution['x']}, y={best_solution['x']}")
                self.env.update_stands_state(best_body_state)
        print(self.env.terminal_yes_or_not(.01))
        print("Init Body State: ", init_body_state)
        print("Init Fitness: ", init_D)
        print("Best Body State: ", self.env.stand_body_state)
        print("Best Fitness: ", f_best)
        with open("env_2_round.pickle", "wb") as file:
            pickle.dump(self.env, file)

class SA_1_Round:
    def __init__(self, env, iter=96, T0=0.01, Tf=1e-6, alpha=0.5):
        self.env = env
        self.iter = iter  # 内循环迭代次数,即为L =100
        self.alpha = alpha  # 降温系数，alpha=0.99
        self.T0 = T0  # 初始温度T0为100
        self.Tf = Tf  # 温度终值Tf为0.01
        self.T = T0  # 当前温度
        self.x = [random() * 11 - 5 for i in range(iter)]  # 随机生成100个x的值
        self.y = [random() * 11 - 5 for i in range(iter)]  # 随机生成100个y的值
        self.most_best = []
        """
        random()这个函数取0到1之间的小数
        如果你要取0-10之间的整数（包括0和10）就写成 (int)random()*11就可以了，11乘以零点多的数最大是10点多，最小是0点多
        该实例中x1和x2的绝对值不超过5（包含整数5和-5），（random() * 11 -5）的结果是-6到6之间的任意值（不包括-6和6）
        （random() * 10 -5）的结果是-5到5之间的任意值（不包括-5和5），所有先乘以11，取-6到6之间的值，产生新解过程中，用一个if条件语句把-5到5之间（包括整数5和-5）的筛选出来。
        """
        self.history = {'f': [], 'T': []}
        self.terminal_tensor = torch.tensor(np.zeros((1, env.d_n, env.magnet_points_n, 2)))
        self.terminal_tensor[:, :, :30, :] = torch.tensor(np.array([0.0116, 0.0104]))
        self.terminal_tensor[:, :, 30:, :] = torch.tensor(np.array([0.010, 0.010]))

    def generate_new(self, x, y):  # 扰动产生新解的过程
        while True:
            x_new = x + self.T * (random() - random())
            y_new = y + self.T * (random() - random())
            if (-5 <= x_new <= 5) & (-5 <= y_new <= 5):
                break  # 重复得到新解，直到产生的新解满足约束条件
        return x_new, y_new

    def generate_new_actions(self, ratio):
        # 扰动产生新解的过程
        while True:
            # # x_new = x + self.T * (random() - random())
            # # y_new = y + self.T * (random() - random())
            # if (-5 <= x_new <= 5) & (-5 <= y_new <= 5):
            #     break  # 重复得到新解，直到产生的新解满足约束条件
            # ran = np.random.rand(3, 3)
            # ran2 = np.random.rand(3, 3)
            new_action = self.env.action_bound[:4] * (2 * np.random.rand(len(self.env.agents) - 1, 3) - 1) * ratio
            new_action_abs = abs(new_action)
            if np.all(new_action_abs <= self.env.action_bound[:4] * ratio):
                break
        return new_action

    def Metrospolis(self, f, f_new):  # Metropolis准则
        if f_new <= f:
            return 1
        else:
            # print(f - f_new)
            p = math.exp((f - f_new) / self.T)
            if random() < p:
                return 1
            else:
                return 0

    # def best(self):  # 获取最优目标函数值
    #     f_list = []  # f_list数组保存每次迭代之后的值
    #     for i in range(self.iter):
    #         f = self.func(self.x[i], self.y[i])
    #         f_list.append(f)
    #     f_best = min(f_list)
    #
    #     idx = f_list.index(f_best)
    #     return f_best, idx  # f_best,idx分别为在该温度下，迭代L次之后目标函数的最优解和最优解的下标

    @ray.remote
    def generate_multi_env(self, env):
        return copy.deepcopy(env)

    @ray.remote
    def multi_terminal(self, env):
        return env.terminal()

    @ray.remote
    def multi_env_step(self, env, ratio):
        # action = random() * env.action_bound * ratio
        action = self.generate_new_actions(ratio)
        env.step(action)
        # for i , agent in enumerate(env.agents):
        #     body_state = agent.body_state
        #     action_a = action[i]
        #     agent.continuous_body_move(action[i])
        #     body_state = agent.body_state
        # env.space.step(.0001)
        # # d_matrix = self.generate_d_matrix()
        # d_diff_matrix = env.magnet_matrix() - env.target_points_d_matrix
        # env.terminal_list = env.generate_terminal_list_np(d_diff_matrix)
        # env.D = env.d_list_np(d_diff_matrix).min()
        return env

    @ray.remote(num_gpus=0.2)
    def multi_env_step_cuda(self, env, ratio):
        # action = random() * env.action_bound * ratio
        action = self.generate_new_actions(ratio)
        env.step_ray_cuda(action)
        # for i , agent in enumerate(env.agents):
        #     body_state = agent.body_state
        #     action_a = action[i]
        #     agent.continuous_body_move(action[i])
        #     body_state = agent.body_state
        # env.space.step(.0001)
        # # d_matrix = self.generate_d_matrix()
        d_diff_matrix = env.magnet_matrix - env.target_points_d_matrix
        d_diff_matrix_tensor = torch.tensor(d_diff_matrix, device='cuda')
        env.terminal_list = torch.all(torch.sqrt(torch.sum(torch.square(d_diff_matrix_tensor), dim=2)) < 0.01, dim=1).cpu().numpy()
        env.D = torch.sqrt(torch.sum(torch.sum(torch.square(d_diff_matrix_tensor), dim=2), dim=1)).cpu().numpy().min()
        return env

    @ray.remote
    def multi_env_just_step(self, env, ratio):
        action = self.generate_new_actions(ratio)
        env.step_ray_cuda(action)
        return env

    @ray.remote
    def get_multi_env_diff_matrix(self, env):
        return env.magnet_matrix - env.target_points_d_matrix

    @ray.remote
    def update_env_state(self, env):
        d_diff_matrix = env.magnet_matrix() - env.target_points_d_matrix
        env.terminal_list = env.generate_terminal_list_np(d_diff_matrix)
        env.body_state = env.stand_body_state()
        env.magnet_state = env.stand_magnet_state()
        env.D = env.d_list_np(d_diff_matrix).min()
        return env

    @ray.remote
    def env_space_step(self, env):
        env.space.step(0.0001)
        return env

    @ray.remote
    def get_distance_list(self, env):
        return env.D

    @ray.remote
    def multi_env_update_state(self, env, best_body_state):
        # env.best_body_state = best_body_state
        env.update_stands_state(best_body_state)
        return env

    # def best(self):  # 获取最优目标函数值
    #     f_list = []  # f_list数组保存每次迭代之后的值
    #     for i in range(self.iter):
    #         f = self.func(self.x[i], self.y[i])
    #         f_list.append(f)
    #     f_best = min(f_list)
    #
    #     idx = f_list.index(f_best)
    #     return f_best, idx  # f_best,idx分别为在该温度下，迭代L次之后目标函数的最优解和最优解的下标

    def run(self):
        # best_body_state = None
        # f_best = 0
        distances = []
        init_body_state, init_D = self.env.reset()
        best_body_state = init_body_state
        f_best = init_D
        print(init_body_state)
        distances.append(init_D)
        if not self.env.terminal_yes_or_not(.01):
            init_T = self.T
            terminal = False
            while True:
                # count = 0

                if terminal:
                    break
                # f_best = 0
                # count = 0
                # 外循环迭代，当前温度小于终止温度的阈值
                self.T = init_T
                SA_done = False
                while self.T > self.Tf:
                    SA_done = self.env.terminal_yes_or_not(.01)
                    print("First SA_done: ", SA_done)
                    terminal = SA_done
                    if SA_done:
                        break
                    action = random() * self.env.action_bound
                    # 内循环迭代100次
                    for i in range(self.iter):
                        action = self.generate_new_actions(action, 1) # 产生新解
                        body_state, D_ = self.env.step(action)  # 产生新值
                        if self.Metrospolis(f_best, D_):  # 判断是否接受新值
                            best_body_state = body_state
                            f_best = D_

                        self.env.go_back(action)
                    # 迭代L次记录在该温度下最优解
                    # ft, _ = self.best()
                    # self.history['f'].append(fbest)
                    # self.history['T'].append(self.T)
                    # 温度按照一定的比例下降（冷却）
                    self.env.update_stands_state(best_body_state)
                    self.T = self.T * self.alpha
                    # count += 1
                    # print("Results of ep{}".format(count))
                    # print("SA First Round")
                    # print("Now Body State: ", best_body_state)
                    print("Now Fitness: ", f_best)
                    distances.append(f_best)
                    # 得到最优解
                # f_best, idx = self.best()
                # count += 1
                # print("Results of ep{}".format(count))
                # print("Now Body State: ", best_body_state)
                # print("Now Fitness: ", f_best)
            # print(f"F={f_best}, x={best_solution['x']}, y={best_solution['x']}")
            #     self.env.update_stands_state(best_body_state)
        # if not self.env.terminal_yes_or_not(.01):
        print("SA end!!!!")
        print(self.env.terminal_yes_or_not(.01))
        print("Init Body State: ", init_body_state)
        print("Init Fitness: ", init_D)
        print("Best Body State: ", self.env.stand_body_state)
        print("Best Fitness: ", f_best)
        with open("env_1_round_5_workstation.pickle", "wb") as file:
            pickle.dump(self.env, file)
        distances.append(f_best)
        return distances

    # def ray_torch_multi_run(self, logger):
    #     distances = []
    #     self.env.reset()
    #     init_body_state = self.env.stand_body_state
    #     init_D = self.env.D
    #     best_body_state = init_body_state
    #     f_best = init_D
    #     distances.append(f_best)
    #     # print(best_body_state)
    #     logger.info(best_body_state)
    #     # print(f_best)
    #     logger.info(f_best)
    #     multi_env_ref = [self.generate_multi_env.remote(self, self.env) for _ in range(self.iter)]
    #     # multi_env = ray.get(multi_env_ref)
    #     terminal_list_ref = [self.multi_terminal.remote(self, env) for env in multi_env_ref]
    #     # print(ray.get(terminal_list_ref))
    #     best_env = None
    #     if not np.any(ray.get(terminal_list_ref)):
    #         init_T = self.T
    #         terminal = False
    #         while True:
    #             if terminal:
    #                 break
    #             self.T = init_T
    #             # self.T = init_T
    #             SA_done = False
    #             while self.T > self.Tf:
    #                 terminal = SA_done
    #                 if SA_done:
    #                     break
    #                 if f_best < .2:
    #                     multi_env_ref = [self.multi_env_just_step.remote(self, env, 0.2) for env in multi_env_ref]
    #                 else:
    #                     multi_env_ref = [self.multi_env_just_step.remote(self, env, 1) for env in multi_env_ref]
    #                 # multi_env = ray.get(multi_env_ref)
    #                 diff_matrix_group_ref = [self.get_multi_env_diff_matrix.remote(self, env) for env in multi_env_ref]
    #                 diff_matrix_group = ray.get(diff_matrix_group_ref)
    #                 diff_matrix_group_tensor = torch.tensor(np.array(diff_matrix_group), device='cuda')
    #                 diff_matrix_group_tensor_batches = DataLoader(diff_matrix_group_tensor)
    #                 terminal_list = []
    #                 d_list = []
    #                 for batch in diff_matrix_group_tensor_batches:
    #                     d = torch.sqrt(torch.sum(torch.sum(torch.square(batch), dim=3), dim=2)).min().cpu().numpy()
    #                     terminals = torch.all(torch.sqrt(torch.sum(torch.square(batch), dim=3)) < 0.005, dim=2).cpu().numpy()[0]
    #                     terminal = np.any(terminals)
    #                     d_list.append(d)
    #                     terminal_list.append(terminal)
    #                 # terminal_list_ref = [self.multi_terminal.remote(self, env) for env in multi_env_ref]
    #                 # terminal_list = ray.get(terminal_list_ref)
    #                 SA_done = np.any(terminal_list)
    #                 if SA_done:
    #                     index = terminal_list.index(True)
    #                     multi_env = ray.get(multi_env_ref)
    #                     best_env = multi_env[index]
    #                     best_body_state = best_env.stand_body_state
    #                     f_best = best_env.D
    #                     distances.append(f_best)
    #                 # d_list_ref = [self.get_distance_list.remote(self, env) for env in multi_env_ref]
    #                 # d_list = ray.get(d_list_ref)
    #                 elif self.Metrospolis(f_best, min(d_list)):
    #                     f_best = min(d_list)
    #                     distances.append(f_best)
    #                     index = d_list.index(min(d_list))
    #                     # print(multi_env[index].best_body_state)
    #                     multi_env = ray.get(multi_env_ref)
    #                     best_env = multi_env[index]
    #                     # multi_env_ref = [self.generate_multi_env.remote(self, env) for _ in range(self.iter)]
    #                     # multi_env = ray.get(multi_env_ref)
    #                     best_body_state = best_env.stand_body_state
    #                     multi_env_ref = [self.multi_env_update_state.remote(self, env, best_body_state) for env in
    #                                      multi_env_ref]
    #                 else:
    #                     multi_env_ref = [self.multi_env_update_state.remote(self, env, best_body_state) for env in
    #                                      multi_env_ref]
    #                     # multi_env_ref = [self.generate_multi_env.remote(self, best_env) for _ in range(self.iter)]
    #                     # del best_body_state_ref
    #                     # def list_ref
    #                 # else:
    #                 #     distances.append(f_best)
    #                 #     multi_env_ref = [self.multi_env_update_state.remote(self, env, best_body_state) for env in multi_env_ref]
    #                 # del step_results_ref
    #                 # print("Now Fitness: ", f_best)
    #                 logger.info("Now Fitness: {}".format(f_best))
    #         # env = ray.get(multi_env_ref)[0]
    #     # print("Climbing end!!!!")
    #     logger.info("SA end!!!!")
    #     # print(self.env.terminal_yes_or_not(.01))
    #     # print("Init Body State: ", init_body_state)
    #     logger.info("Init Body State: {}".format(init_body_state))
    #     # print("Init Fitness: ", init_D)
    #     logger.info("Init Fitness: {}".format(init_D))
    #     # print("Best Body State: ", best_body_state)
    #     logger.info("Best Body State: {}".format(best_body_state))
    #     # print("Best Fitness: ", f_best)
    #     logger.info("Best Fitness: {}".format(f_best))
    #     logger.info("Best env magnet state: {}".format(best_env.stand_magnet_state))
    #     logger.info("Best env target state: {}".format(best_env.stand_target_state))
    #     # print("Terminal List", self.env.terminal_list)
    #     with open("env_SA_5_workstation_allD_ray_5_48_torch_add_state_ter_5um20240720.pickle", "wb") as file:
    #         pickle.dump(best_env, file)
    #     distances.append(f_best)
    #     return distances

    def ray_torch_multi_run(self, logger):
        distances = []
        # self.env.reset()
        init_body_state = self.env.stand_body_state
        init_D = self.env.D
        best_body_state = init_body_state
        f_best = init_D
        distances.append(f_best)
        # print(best_body_state)
        logger.info(best_body_state)
        # print(f_best)
        logger.info(f_best)
        multi_env_ref = [self.generate_multi_env.remote(self, self.env) for _ in range(self.iter)]
        # multi_env = ray.get(multi_env_ref)
        best_env = self.env
        diff_matrix_group_ref = [self.get_multi_env_diff_matrix.remote(self, env) for env in multi_env_ref]
        diff_matrix_group = ray.get(diff_matrix_group_ref)
        diff_matrix_group_tensor = torch.tensor(np.array(diff_matrix_group))
        diff_matrix_group_tensor_batches = DataLoader(diff_matrix_group_tensor)
        d_list = []
        terminal_list = []
        for batch in diff_matrix_group_tensor_batches:
            # print(batch.shape)
            # d = torch.sqrt(torch.sum(torch.sum(torch.square(batch), dim=3), dim=2)).min().cpu().numpy()
            # terminals = torch.all(torch.sqrt(torch.sum(torch.square(batch[:, :, :30, :]), dim=3)) < 0.005, dim=2).cpu().numpy()[0]
            terminal = torch.any(torch.all(torch.all(torch.abs(batch) < self.terminal_tensor, dim=3), dim=2), dim=1)
            # terminal = np.any(terminals)
            # d_list.append(d)
            terminal_list.append(terminal)
        # print(min(d_list))
        if not np.any(terminal_list):
            init_T = self.T
            terminal = False
            while True:
                if terminal:
                    break
                self.T = init_T
                # self.T = init_T
                SA_done = False
                while self.T > self.Tf:
                    terminal = SA_done
                    if SA_done:
                        break
                    if f_best < .2:
                        multi_env_ref = [self.multi_env_just_step.remote(self, env, 0.2) for env in multi_env_ref]
                    else:
                        multi_env_ref = [self.multi_env_just_step.remote(self, env, 1) for env in multi_env_ref]
                    # multi_env = ray.get(multi_env_ref)
                    diff_matrix_group_ref = [self.get_multi_env_diff_matrix.remote(self, env) for env in multi_env_ref]
                    diff_matrix_group = ray.get(diff_matrix_group_ref)
                    diff_matrix_group_tensor = torch.tensor(np.array(diff_matrix_group))
                    diff_matrix_group_tensor_batches = DataLoader(diff_matrix_group_tensor)

                    terminal_list = []
                    d_list = []
                    for batch in diff_matrix_group_tensor_batches:
                        # print(batch.shape)
                        d = torch.sqrt(torch.sum(torch.sum(torch.square(batch), dim=3), dim=2)).min().numpy()
                        # terminals = torch.all(torch.sqrt(torch.sum(torch.square(batch[:, :, :30, :]), dim=3)) < 0.005, dim=2).cpu().numpy()[0]
                        # terminal = np.any(terminals)
                        # terminal = torch.any(torch.all(torch.abs(batch) < self.terminal_tensor, dim=1) == True).numpy()
                        terminal = torch.any(
                            torch.all(torch.all(torch.abs(batch) < self.terminal_tensor, dim=3), dim=2), dim=1)
                        d_list.append(d)
                        terminal_list.append(terminal)
                    SA_done = np.any(terminal_list)
                    if SA_done:
                        index = terminal_list.index(True)
                        multi_env = ray.get(multi_env_ref)
                        best_env = multi_env[index]
                        best_body_state = best_env.stand_body_state
                        f_best = best_env.D
                        distances.append(f_best)
                    # d_list_ref = [self.get_distance_list.remote(self, env) for env in multi_env_ref]
                    # d_list = ray.get(d_list_ref)
                    elif self.Metrospolis(f_best, min(d_list)):
                        f_best = min(d_list)
                        distances.append(f_best)
                        index = d_list.index(min(d_list))
                        # print(multi_env[index].best_body_state)
                        multi_env = ray.get(multi_env_ref)
                        best_env = multi_env[index]
                        # multi_env_ref = [self.generate_multi_env.remote(self, env) for _ in range(self.iter)]
                        # multi_env = ray.get(multi_env_ref)
                        best_body_state = best_env.stand_body_state
                        multi_env_ref = [self.multi_env_update_state.remote(self, env, best_body_state) for env in
                                         multi_env_ref]
                    else:
                        multi_env_ref = [self.multi_env_update_state.remote(self, env, best_body_state) for env in
                                         multi_env_ref]
                        # multi_env_ref = [self.generate_multi_env.remote(self, best_env) for _ in range(self.iter)]
                        # del best_body_state_ref
                        # def list_ref
                    # else:
                    #     distances.append(f_best)
                    #     multi_env_ref = [self.multi_env_update_state.remote(self, env, best_body_state) for env in multi_env_ref]
                    # del step_results_ref
                    # print("Now Fitness: ", f_best)
                    self.T = self.T * self.alpha
                    logger.info("Now Fitness: {}".format(f_best))
            # env = ray.get(multi_env_ref)[0]
        # print("Climbing end!!!!")
        logger.info("SA end!!!!")
        # print(self.env.terminal_yes_or_not(.01))
        # print("Init Body State: ", init_body_state)
        logger.info("Init Body State: {}".format(init_body_state))
        # print("Init Fitness: ", init_D)
        logger.info("Init Fitness: {}".format(init_D))
        # print("Best Body State: ", best_body_state)
        logger.info("Best Body State: {}".format(best_body_state))
        # print("Best Fitness: ", f_best)
        logger.info("Best Fitness: {}".format(f_best))
        logger.info("Best env magnet state: {}".format(best_env.stand_magnet_state))
        logger.info("Best env target state: {}".format(best_env.stand_target_state))
        # print("Terminal List", self.env.terminal_list)
        with open("env_sa_10_10_10p4_11p6_20240826.pickle", "wb") as file:
            pickle.dump(best_env, file)
        distances.append(f_best)
        return distances, best_env

class Climbing:
    def __init__(self, env, iter=96):
        self.env = env
        self.iter = iter
        self.terminal_tensor = torch.tensor(np.zeros((1, env.d_n, env.magnet_points_n, 2)))
        self.terminal_tensor[:, :, :30, :] = torch.tensor(np.array([0.015, 0.015]))
        self.terminal_tensor[:, :, 30:, :] = torch.tensor(np.array([0.015, 0.010]))

    def generate_new_actions(self, ratio):
        # 扰动产生新解的过程
        while True:
            # # x_new = x + self.T * (random() - random())
            # # y_new = y + self.T * (random() - random())
            # if (-5 <= x_new <= 5) & (-5 <= y_new <= 5):
            #     break  # 重复得到新解，直到产生的新解满足约束条件
            # ran = np.random.rand(3, 3)
            # ran2 = np.random.rand(3, 3)
            new_action = self.env.action_bound[:4] * (2 * np.random.rand(len(self.env.agents) - 1, 3) - 1) * ratio
            new_action_abs = abs(new_action)
            if np.all(new_action_abs <= self.env.action_bound[:4] * ratio):
                break
        return new_action

    def Metrospolis(self, f, f_new):  # Metropolis准则
        if f_new <= f:
            return 1
        else:
            # print(f - f_new)
            p = math.exp((f - f_new) / self.T)
            if random() < p:
                return 1
            else:
                return 0

    # def best(self):  # 获取最优目标函数值
    #     f_list = []  # f_list数组保存每次迭代之后的值
    #     for i in range(self.iter):
    #         f = self.func(self.x[i], self.y[i])
    #         f_list.append(f)
    #     f_best = min(f_list)
    #
    #     idx = f_list.index(f_best)
    #     return f_best, idx  # f_best,idx分别为在该温度下，迭代L次之后目标函数的最优解和最优解的下标

    def run(self, logger):
        # best_body_state = None
        # f_best = 0
        distances = []
        self.env.reset()
        init_body_state = self.env.stand_body_state
        init_D = self.env.D
        best_body_state = init_body_state
        f_best = init_D
        distances.append(f_best)
        print(best_body_state)
        logger.info(best_body_state)
        print(f_best)
        logger.info(f_best)

        if not self.env.terminal():
            terminal = False
            while True:
                if terminal:
                    break


                # print("First SA_done: ", SA_done)
                # terminal = SA_done
                # if SA_done:
                #     break

                # 内循环迭代100次
                for i in range(self.iter):
                    if f_best < 0.25:
                        # action = random() * self.env.action_bound * 0.2
                        action = self.generate_new_actions(0.2) # 产生新解
                    else:
                        # action = random() * self.env.action_bound
                        action = self.generate_new_actions(1)
                    self.env.step(action)
                    inner_terminal = self.env.terminal()
                    if inner_terminal:
                        terminal = True
                        best_body_state = self.env.stand_body_state
                        f_best = self.env.D
                        break
                    if self.env.D < f_best:  # 判断是否接受新值
                        best_body_state = self.env.stand_body_state
                        f_best = self.env.D
                    self.env.go_back(action)
                # 迭代L次记录在该温度下最优解
                # ft, _ = self.best()
                # self.history['f'].append(fbest)
                # self.history['T'].append(self.T)
                # 温度按照一定的比例下降（冷却）
                self.env.update_stands_state(best_body_state)
                print("Now Fitness: ", f_best)
                logger.info("Now Fitness: {}".format(f_best))
                distances.append(f_best)
                # terminal = self.env.terminal()
                # 得到最优解
                # f_best, idx = self.best()
                # count += 1
                # print("Results of ep{}".format(count))
                # print("Now Body State: ", best_body_state)
                # print("Now Fitness: ", f_best)
            # print(f"F={f_best}, x={best_solution['x']}, y={best_solution['x']}")
            #     self.env.update_stands_state(best_body_state)
        # if not self.env.terminal_yes_or_not(.01):
        print("Climbing end!!!!")
        logger.info("Climbing end!!!!")
        # print(self.env.terminal_yes_or_not(.01))
        print("Init Body State: ", init_body_state)
        logger.info("Init Body State: {}".format(init_body_state))
        print("Init Fitness: ", init_D)
        logger.info("Init Fitness: {}".format(init_D))
        print("Best Body State: ", best_body_state)
        logger.info("Best Body State: {}".format(best_body_state))
        print("Best Fitness: ", f_best)
        logger.info("Best Fitness: {}".format(f_best))
        # print("Terminal List", self.env.terminal_list)
        with open("env_climbing_5_workstation_allD_np_5_50.pickle", "wb") as file:
            pickle.dump(self.env, file)
        distances.append(f_best)
        return distances

    @ray.remote
    def generate_multi_env(self, env):
        return copy.deepcopy(env)
    @ray.remote
    def multi_terminal(self, env):
        return env.terminal()

    @ray.remote
    def multi_env_step(self, env, ratio):
        # action = random() * env.action_bound * ratio
        action = self.generate_new_actions(ratio)
        env.step(action)
        # for i , agent in enumerate(env.agents):
        #     body_state = agent.body_state
        #     action_a = action[i]
        #     agent.continuous_body_move(action[i])
        #     body_state = agent.body_state
        # env.space.step(.0001)
        # # d_matrix = self.generate_d_matrix()
        # d_diff_matrix = env.magnet_matrix() - env.target_points_d_matrix
        # env.terminal_list = env.generate_terminal_list_np(d_diff_matrix)
        # env.D = env.d_list_np(d_diff_matrix).min()
        return env

    @ray.remote(num_gpus=0.2)
    def multi_env_step_cuda(self, env, ratio):
        # action = random() * env.action_bound * ratio
        action = self.generate_new_actions(ratio)
        env.step_ray_cuda(action)
        # for i , agent in enumerate(env.agents):
        #     body_state = agent.body_state
        #     action_a = action[i]
        #     agent.continuous_body_move(action[i])
        #     body_state = agent.body_state
        # env.space.step(.0001)
        # # d_matrix = self.generate_d_matrix()
        d_diff_matrix = env.magnet_matrix - env.target_points_d_matrix
        d_diff_matrix_tensor = torch.tensor(d_diff_matrix, device='cuda')
        env.terminal_list = torch.all(torch.sqrt(torch.sum(torch.square(d_diff_matrix_tensor), dim=2)) < 0.01, dim=1).cpu().numpy()
        env.D = torch.sqrt(torch.sum(torch.sum(torch.square(d_diff_matrix_tensor), dim=2), dim=1)).cpu().numpy().min()
        return env

    @ray.remote
    def multi_env_just_step(self, env, ratio):
        action = self.generate_new_actions(ratio)
        env.step_ray_cuda(action)
        return env

    @ray.remote
    def get_multi_env_diff_matrix(self, env):
        return env.magnet_matrix - env.target_points_d_matrix

    @ray.remote
    def update_env_state(self, env):
        d_diff_matrix = env.magnet_matrix() - env.target_points_d_matrix
        env.terminal_list = env.generate_terminal_list_np(d_diff_matrix)
        env.body_state = env.stand_body_state()
        env.magnet_state = env.stand_magnet_state()
        env.D = env.d_list_np(d_diff_matrix).min()
        return env

    @ray.remote
    def env_space_step(self, env):
        env.space.step(0.0001)
        return env

    @ray.remote
    def get_distance_list(self, env):
        return env.D

    @ray.remote
    def multi_env_update_state(self, env, best_body_state):
        # env.best_body_state = best_body_state
        env.update_stands_state(best_body_state)
        return env


    def multi_run(self, logger):
        distances = []
        # self.env.reset()
        init_body_state = self.env.stand_body_state
        init_D = self.env.D
        best_body_state = init_body_state
        f_best = init_D
        distances.append(f_best)
        # print(best_body_state)
        logger.info(best_body_state)
        # print(f_best)
        logger.info(f_best)
        multi_env_ref = [self.generate_multi_env.remote(self, self.env) for _ in range(self.iter)]
        # multi_env = ray.get(multi_env_ref)
        terminal_list_ref = [self.multi_terminal.remote(self, env) for env in multi_env_ref]
        # print(ray.get(terminal_list_ref))
        best_env = self.env
        terminal = False
        if not np.any(ray.get(terminal_list_ref)):

            while True:
                if terminal:
                    break
                if f_best < 0.2:
                    multi_env_ref = [self.multi_env_step.remote(self, env, 0.2) for env in multi_env_ref]
                else:
                    multi_env_ref = [self.multi_env_step.remote(self, env, 1) for env in multi_env_ref]

                terminal_list_ref = [self.multi_terminal.remote(self, env) for env in multi_env_ref]
                terminal_list = ray.get(terminal_list_ref)
                terminal = np.any(terminal_list)
                d_list_ref = [self.get_distance_list.remote(self, env) for env in multi_env_ref]
                d_list = ray.get(d_list_ref)
                if terminal:
                    index = terminal_list.index(True)
                    multi_env = ray.get(multi_env_ref)
                    best_env = multi_env[index]
                    print("find it !!!")
                    abs_diff = abs(best_env.stand_magnet_state - best_env.target_magnet_points)
                    print(np.all(abs_diff < 0.015))
                    best_body_state = best_env.stand_body_state
                    f_best = best_env.D
                    distances.append(f_best)
                    # multi_env_ref = [self.multi_env_update_state.remote(self, env, best_body_state) for env in multi_env_ref]

                elif min(d_list) < f_best:
                    f_best = min(d_list)
                    distances.append(f_best)
                    index = d_list.index(min(d_list))
                    # print(multi_env[index].best_body_state)
                    multi_env = ray.get(multi_env_ref)
                    best_env = multi_env[index]
                    print(index)
                    # multi_env_ref = [self.generate_multi_env.remote(self, env) for _ in range(self.iter)]
                    # multi_env = ray.get(multi_env_ref)
                    best_body_state = best_env.stand_body_state
                    multi_env_ref = [self.multi_env_update_state.remote(self, env, best_body_state) for env in multi_env_ref]
                    # multi_env_ref = [self.generate_multi_env.remote(self, best_env) for _ in range(self.iter)]
                    # del best_body_state_ref
                    # def list_ref
                else:
                    distances.append(f_best)
                    multi_env_ref = [self.multi_env_update_state.remote(self, env, best_body_state) for env in multi_env_ref]
                # del step_results_ref
                # print("Now Fitness: ", f_best)
                logger.info("Now Fitness: {}".format(f_best))
            # env = ray.get(multi_env_ref)[0]
        # print("Climbing end!!!!")
        logger.info("Climbing end!!!!")
        # print(self.env.terminal_yes_or_not(.01))
        # print("Init Body State: ", init_body_state)
        logger.info("Init Body State: {}".format(init_body_state))
        # print("Init Fitness: ", init_D)
        logger.info("Init Fitness: {}".format(init_D))
        # print("Best Body State: ", best_body_state)
        logger.info("Best Body State: {}".format(best_body_state))
        # print("Best Fitness: ", f_best)
        logger.info("Best Fitness: {}".format(f_best))
        # print("Terminal List", self.env.terminal_list)


        # logger.info("Best Fitness: {}".format(f_best))
        logger.info("Best env magnet state: {}".format(best_env.stand_magnet_state))
        logger.info("Best env target state: {}".format(best_env.stand_target_state))
        with open("env_climbing_13_14_20240726.pickle", "wb") as file:
            pickle.dump(best_env, file)
        distances.append(f_best)
        return distances, best_env


    def multi_run_test(self):
        distances = []
        self.env.reset()
        init_body_state = self.env.stand_body_state()
        f_best = self.env.D
        init_D = f_best
        distances.append(f_best)
        # f_best_ref = ray.put(f_best)
        print(init_body_state)
        print(f_best)
        multi_env_ref = [self.generate_multi_env.remote(self, self.env) for _ in range(self.iter)]
        multi_env = ray.get(multi_env_ref)
        while True:
            multi_env_ref = [self.multi_env_step.remote(self, env, 1) for env in multi_env_ref]
            # multi_env_ref = [self.env_space_step.remote(self, env) for env in multi_env_ref]
            # multi_env_ref = [self.update_env_state.remote(self, env) for env in multi_env_ref]
            multi_env = (ray.get(multi_env_ref))
            print(multi_env[0].agents[0].line_state())
            print(multi_env[0].agents[0].fix_point_list)
            print(multi_env[0].agents[0].fix_point_state())

    def ray_gpu_test(self):
        self.env.reset()
        init_D = self.env.D
        print(init_D)
        init_body_state = self.env.stand_body_state
        print(init_body_state)
        d_diff_matrix = self.env.magnet_matrix - self.env.target_points_d_matrix
        d_diff_matrix_tensor = torch.tensor(d_diff_matrix, device='cuda')
        self.env.terminal_list = torch.all(torch.sqrt(torch.sum(torch.square(d_diff_matrix_tensor), dim=2)) < 0.01, dim=1).cpu().numpy()
        self.env.D = torch.sqrt(torch.sum(torch.sum(torch.square(d_diff_matrix_tensor), dim=2), dim=1)).cpu().numpy().min()
        print(self.env.D)
        print(self.env.terminal_list)

    def ray_gpu_multi_run(self, logger):
        distances = []
        self.env.reset()
        init_body_state = self.env.stand_body_state
        init_D = self.env.D
        best_body_state = init_body_state
        f_best = init_D
        distances.append(f_best)
        # print(best_body_state)
        logger.info(best_body_state)
        # print(f_best)
        logger.info(f_best)
        multi_env_ref = [self.generate_multi_env.remote(self, self.env) for _ in range(self.iter)]
        # multi_env = ray.get(multi_env_ref)
        terminal_list_ref = [self.multi_terminal.remote(self, env) for env in multi_env_ref]
        # print(ray.get(terminal_list_ref))
        best_env = None
        if not np.any(ray.get(terminal_list_ref)):
            terminal = False
            while True:
                if terminal:
                    break
                if f_best < .2:
                    multi_env_ref = [self.multi_env_step_cuda.remote(self, env, 0.2) for env in multi_env_ref]
                else:
                    multi_env_ref = [self.multi_env_step_cuda.remote(self, env, 1) for env in multi_env_ref]
                # multi_env = ray.get(multi_env_ref)
                terminal_list_ref = [self.multi_terminal.remote(self, env) for env in multi_env_ref]
                terminal_list = ray.get(terminal_list_ref)
                terminal = np.any(terminal_list)
                if terminal:
                    index = terminal_list.index(True)
                    multi_env = ray.get(multi_env_ref)
                    best_env = multi_env[index]
                    best_body_state = best_env.stand_body_state
                    f_best = best_env.D
                    distances.append(f_best)
                else:

                    d_list_ref = [self.get_distance_list.remote(self, env) for env in multi_env_ref]
                    d_list = ray.get(d_list_ref)
                    if min(d_list) < f_best:
                        f_best = min(d_list)
                        distances.append(f_best)
                        index = d_list.index(min(d_list))
                        # print(multi_env[index].best_body_state)
                        multi_env = ray.get(multi_env_ref)
                        best_env = multi_env[index]
                        # multi_env_ref = [self.generate_multi_env.remote(self, env) for _ in range(self.iter)]
                        # multi_env = ray.get(multi_env_ref)
                        best_body_state = best_env.stand_body_state
                        multi_env_ref = [self.multi_env_update_state.remote(self, env, best_body_state) for env in
                                         multi_env_ref]
                    # multi_env_ref = [self.generate_multi_env.remote(self, best_env) for _ in range(self.iter)]
                    # del best_body_state_ref
                    # def list_ref
                # else:
                #     distances.append(f_best)
                #     multi_env_ref = [self.multi_env_update_state.remote(self, env, best_body_state) for env in multi_env_ref]
                # del step_results_ref
                # print("Now Fitness: ", f_best)
                logger.info("Now Fitness: {}".format(f_best))
            # env = ray.get(multi_env_ref)[0]
        # print("Climbing end!!!!")
        logger.info("Climbing end!!!!")
        # print(self.env.terminal_yes_or_not(.01))
        # print("Init Body State: ", init_body_state)
        logger.info("Init Body State: {}".format(init_body_state))
        # print("Init Fitness: ", init_D)
        logger.info("Init Fitness: {}".format(init_D))
        # print("Best Body State: ", best_body_state)
        logger.info("Best Body State: {}".format(best_body_state))
        # print("Best Fitness: ", f_best)
        logger.info("Best Fitness: {}".format(f_best))
        # print("Terminal List", self.env.terminal_list)
        with open("env_climbing_5_workstation_allD_ray_5_20_cuda5.pickle", "wb") as file:
            pickle.dump(best_env, file)
        distances.append(f_best)
        return distances


    def ray_torch_multi_run(self, logger):
        distances = []
        # self.env.reset()
        init_body_state = self.env.stand_body_state
        init_D = self.env.D
        # print(init_D)
        best_body_state = init_body_state
        f_best = init_D
        distances.append(f_best)
        # print(best_body_state)
        logger.info(best_body_state)
        # print(f_best)
        logger.info(f_best)
        multi_env_ref = [self.generate_multi_env.remote(self, self.env) for _ in range(self.iter)]
        # multi_env = ray.get(multi_env_ref)
        # terminal_list_ref = [self.multi_terminal.remote(self, env) for env in multi_env_ref]
        # print(ray.get(terminal_list_ref))
        best_env = self.env
        diff_matrix_group_ref = [self.get_multi_env_diff_matrix.remote(self, env) for env in multi_env_ref]
        diff_matrix_group = ray.get(diff_matrix_group_ref)
        diff_matrix_group_tensor = torch.tensor(np.array(diff_matrix_group))
        diff_matrix_group_tensor_batches = DataLoader(diff_matrix_group_tensor)
        d_list = []
        terminal_list = []
        for batch in diff_matrix_group_tensor_batches:
            # print(batch.shape)
            # d = torch.sqrt(torch.sum(torch.sum(torch.square(batch), dim=3), dim=2)).min().cpu().numpy()
            # terminals = torch.all(torch.sqrt(torch.sum(torch.square(batch[:, :, :30, :]), dim=3)) < 0.005, dim=2).cpu().numpy()[0]
            terminal = torch.any(torch.all(torch.all(torch.abs(batch) < self.terminal_tensor, dim=3), dim=2), dim=1)
            # terminal = np.any(terminals)
            # d_list.append(d)
            terminal_list.append(terminal)
        # print(min(d_list))
        if not np.any(terminal_list):
            terminal = False
            while True:
                if terminal:
                    break
                if f_best < .2:
                    multi_env_ref = [self.multi_env_just_step.remote(self, env, 0.2) for env in multi_env_ref]
                else:
                    multi_env_ref = [self.multi_env_just_step.remote(self, env, 1) for env in multi_env_ref]
                # multi_env = ray.get(multi_env_ref)
                diff_matrix_group_ref = [self.get_multi_env_diff_matrix.remote(self, env) for env in multi_env_ref]
                diff_matrix_group = ray.get(diff_matrix_group_ref)
                diff_matrix_group_tensor = torch.tensor(np.array(diff_matrix_group))
                diff_matrix_group_tensor_batches = DataLoader(diff_matrix_group_tensor)

                terminal_list = []
                d_list = []
                for batch in diff_matrix_group_tensor_batches:
                    # print(batch.shape)
                    d = torch.sqrt(torch.sum(torch.sum(torch.square(batch), dim=3), dim=2)).min().numpy()
                    # terminals = torch.all(torch.sqrt(torch.sum(torch.square(batch[:, :, :30, :]), dim=3)) < 0.005, dim=2).cpu().numpy()[0]
                    # terminal = np.any(terminals)
                    # terminal = torch.any(torch.all(torch.abs(batch) < self.terminal_tensor, dim=1) == True).numpy()
                    terminal = torch.any(torch.all(torch.all(torch.abs(batch) < self.terminal_tensor, dim=3), dim=2), dim=1)
                    d_list.append(d)
                    terminal_list.append(terminal)

                terminal = np.any(terminal_list)
                if terminal:
                    index = terminal_list.index(True)
                    multi_env = ray.get(multi_env_ref)
                    best_env = multi_env[index]
                    best_body_state = best_env.stand_body_state
                    f_best = best_env.D
                    distances.append(f_best)
                # d_list_ref = [self.get_distance_list.remote(self, env) for env in multi_env_ref]
                # d_list = ray.get(d_list_ref)
                elif min(d_list) < f_best:
                    f_best = min(d_list)
                    distances.append(f_best)
                    index = d_list.index(min(d_list))
                    # print(multi_env[index].best_body_state)
                    multi_env = ray.get(multi_env_ref)
                    print(index)
                    best_env = multi_env[index]
                    # multi_env_ref = [self.generate_multi_env.remote(self, env) for _ in range(self.iter)]
                    # multi_env = ray.get(multi_env_ref)
                    best_body_state = best_env.stand_body_state
                    multi_env_ref = [self.multi_env_update_state.remote(self, env, best_body_state) for env in
                                     multi_env_ref]
                    # multi_env_ref = [self.generate_multi_env.remote(self, best_env) for _ in range(self.iter)]
                    # del best_body_state_ref
                    # def list_ref
                else:
                    multi_env_ref = [self.multi_env_update_state.remote(self, env, best_body_state) for env in
                                     multi_env_ref]
                # else:
                #     distances.append(f_best)
                #     multi_env_ref = [self.multi_env_update_state.remote(self, env, best_body_state) for env in multi_env_ref]
                # del step_results_ref
                # print("Now Fitness: ", f_best)
                logger.info("Now Fitness: {}".format(f_best))
            # env = ray.get(multi_env_ref)[0]
        # print("Climbing end!!!!")
        logger.info("Climbing end!!!!")
        # print(self.env.terminal_yes_or_not(.01))
        # print("Init Body State: ", init_body_state)
        logger.info("Init Body State: {}".format(init_body_state))
        # print("Init Fitness: ", init_D)
        logger.info("Init Fitness: {}".format(init_D))
        # print("Best Body State: ", best_body_state)
        logger.info("Best Body State: {}".format(best_body_state))
        # print("Best Fitness: ", f_best)
        logger.info("Best Fitness: {}".format(f_best))
        logger.info("Best env magnet state: {}".format(best_env.stand_magnet_state))
        logger.info("Best env target state: {}".format(best_env.stand_target_state))
        # print("Terminal List", self.env.terminal_list)
        with open("env_climbing_opt_10_15_15_15_20240826.pickle", "wb") as file:
            pickle.dump(best_env, file)
        distances.append(f_best)
        return distances, best_env




# sa = SA(func)
# sa.run(
#
# plt.plot(sa.history['T'], sa.history['f'])
# plt.title('SA')
# plt.xlabel('T')
# plt.ylabel('f')
# plt.gca().invert_xaxis()
# plt.show()

