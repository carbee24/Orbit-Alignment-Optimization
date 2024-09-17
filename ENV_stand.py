import numpy as np
# from Stand import Stand6
from utils import rotate2X, rotate2X_index, rotate2X_index_cuda
import torch

TARGET = np.array([
        [220., 310.],
        [260., 290.],
        [300., 300.],
        [340., 310.],
        [380., 290.]
    ])

TARGET1 = np.array([
        [70., 310.],
        [110., 290.],
        [150., 300.],
        [190., 310.],
        [230., 290.]
    ])

TARGET2 = np.array([
        [370., 310.],
        [410., 290.],
        [450., 300.],
        [490., 310.],
        [530., 290.]
    ])


class Action_Space():
    def __init__(self, action_bound):
        self.high = action_bound
        self.low = -1 *  action_bound
        self.shape = (len(action_bound.flatten()), 0)

class Env():

    def __init__(self, space, stand_list, action_bound):
        self.space = space
        self.stand_list = stand_list
        self.action_bound = action_bound
        self.action_space = Action_Space(action_bound)


    def reset(self):
        # self.stand_list.position = 300, 300
        for i in range(10):

            action = np.random.choice(self.stand_list.ACTIONS)
            self.stand_list.body_move(action, (.0001, .0001))
            self.space.step(1)
        return self.stand_list.state

    def step(self, action):
        terminal = False
        D = np.sqrt(np.sum(np.square(self.stand_list.manget_state - TARGET)))
        self.stand_list.continuous_body_move(action)
        self.space.step(1)
        next_state = self.stand_list.state
        D_ = np.sqrt(np.sum(np.square(self.stand_list.manget_state - TARGET)))
        D_terminal = np.sqrt(np.sum(np.square(self.stand_list.manget_state - TARGET), axis=1))
        if np.all(D_terminal < 1e-4 / 2):
            R = 100
            terminal = True

        elif D_ < D:
            R = 1
        elif D_ == D:
            R = 0
        else:
            R = -1
        return next_state, R, terminal

class MA_ENV():
    def __init__(self, space, stands, action_bound):
        self.space = space
        self.agents = stands
        self.action_bound = action_bound
        self.action_space = Action_Space(action_bound)

    def reset(self):
        for i in range(10):
            for agent in self.agents:
                action = np.random.choice(agent.ACTIONS)
                agent.body_move(action, (.0001, .0001))
                self.space.step(1)

        return np.array([agent.state for agent in self.agents]).flatten()


    def step(self, action):
        action = action.reshape(-1, 3)
        terminal = False
        COM_TARGET = np.vstack([TARGET1, TARGET2])
        com_manget_state = np.vstack([agent.manget_state() for agent in self.agents])
        D = np.sqrt(np.sum(np.square(com_manget_state - COM_TARGET)))
        for i , agent in enumerate(self.agents):
            agent.continuous_body_move(action[i])
            self.space.step(1)
        next_state = np.array([agent.state for agent in self.agents])
        com_manget_state = np.vstack([agent.manget_state() for agent in self.agents])
        D_ = np.sqrt(np.sum(np.square(com_manget_state - COM_TARGET)))
        D_terminal = np.sqrt(np.sum(np.square(com_manget_state - COM_TARGET), axis=1))
        if np.all(D_terminal < 1e-4 / 2):
            R = 100
            terminal = True

        elif D_ < D:
            R = 1
        elif D_ == D:
            R = 0
        else:
            R = -1
        return next_state.flatten(), R, terminal



class MS_ENV():
    def __init__(self, space, stands, action_bound):
        self.space = space
        self.agents = stands
        self.action_bound = action_bound
        self.action_space = Action_Space(action_bound)
        self.target_points = np.vstack([agent.target_point_group for agent in self.agents])
        self.target_points_vector = self.target_points - self.target_points[0]
        self.target_vector = rotate2X(self.target_points_vector)

    def reset(self):
        manget_state = self.generate_manget_points_vector()
        for i in range(10):
            for agent in self.agents:
                action = np.random.choice(agent.ACTIONS)
                agent.body_move(action, (.1, .1))
                self.space.step(1)
            manget_state = self.generate_manget_points_vector()
        state = self.generate_env_state()
        return state


    def step(self, action):
        action = action.reshape(-1, 3)
        terminal = False
        # COM_TARGET = np.vstack([TARGET1, TARGET2])
        # com_manget_state = np.vstack([agent.manget_state() for agent in self.agents])
        # manget_state_vector = self.generate_manget_points_vector()
        manget_state_vector = self.generate_manget_vector()
        D = np.sqrt(np.sum(np.square(manget_state_vector - self.target_vector)))
        for i , agent in enumerate(self.agents):
            body_state = agent.body_state
            action_a = action[i]
            agent.continuous_body_move(action[i])
            body_state = agent.body_state
            self.space.step(1)
        next_state = self.generate_env_state()
        manget_state_vector = self.generate_manget_vector()
        D_ = np.sqrt(np.sum(np.square(manget_state_vector - self.target_vector)))
        D_terminal = np.sqrt(np.sum(np.square(manget_state_vector - self.target_vector), axis=1))
        if np.all(D_terminal < 0.01):
            R = 100
            terminal = True

        elif D_ < D:
            R = 1
        else:
            R = -1
        return next_state, R, terminal

    def generate_env_state(self):
        # manget_state = np.vstack([agent.manget_state() for agent in self.agents])
        manget_state_vector = self.generate_manget_points_vector()
        Mean = np.mean(manget_state_vector, axis=0)
        Std = np.std(manget_state_vector, axis=0)
        normal_manget_state_vector = (manget_state_vector - Mean) / Std
        return normal_manget_state_vector.flatten()

    def generate_manget_vector(self):
        manget_state_vector = self.generate_manget_points_vector()
        return rotate2X(manget_state_vector)

    def generate_manget_points_vector(self):
        manget_state = np.vstack([agent.manget_state() for agent in self.agents])
        manget_state_vector = manget_state - manget_state[0]
        return manget_state_vector


    # def distance_calc(self):


class MS_ENV_NO_Standardization():
    def __init__(self, space, stands, action_bound):
        self.space = space
        self.agents = stands
        self.action_bound = action_bound
        self.action_space = Action_Space(action_bound)
        self.target_points = np.vstack([agent.target_point_group for agent in self.agents])
        self.target_points_vector = self.target_points - self.target_points[0]
        self.target_vector = rotate2X(self.target_points_vector)

    def reset(self):
        manget_state = self.generate_manget_points_vector()
        for i in range(10):
            for agent in self.agents:
                action = np.random.choice(agent.ACTIONS)
                agent.body_move(action, (.1, .1))

            manget_state = self.generate_manget_points_vector()
        self.space.step(.001)
        state = self.generate_env_state()
        return state


    def step(self, action):
        action = action.reshape(-1, 3)
        terminal = False
        # COM_TARGET = np.vstack([TARGET1, TARGET2])
        # com_manget_state = np.vstack([agent.manget_state() for agent in self.agents])
        # manget_state_vector = self.generate_manget_points_vector()
        manget_state_vector = self.generate_manget_vector()
        D = np.sqrt(np.sum(np.square(manget_state_vector - self.target_vector)))
        for i , agent in enumerate(self.agents):
            body_state = agent.body_state
            action_a = action[i]
            agent.continuous_body_move(action[i])
            body_state = agent.body_state
        self.space.step(.0001)
        next_state = self.generate_env_state()
        manget_state_vector = self.generate_manget_vector()
        D_ = np.sqrt(np.sum(np.square(manget_state_vector - self.target_vector)))
        D_terminal = np.sqrt(np.sum(np.square(manget_state_vector - self.target_vector), axis=1))
        if np.all(D_terminal < 0.01):
            R = 100
            terminal = True

        elif D_ < D:
            R = 1
        else:
            R = -1
        return next_state, R, terminal

    def generate_env_state(self):
        # manget_state = np.vstack([agent.manget_state() for agent in self.agents])
        manget_state_vector = self.generate_manget_points_vector()
        # Mean = np.mean(manget_state_vector, axis=0)
        # Std = np.std(manget_state_vector, axis=0)
        # normal_manget_state_vector = (manget_state_vector - Mean) / Std
        return manget_state_vector.flatten() / 1000

    def generate_manget_vector(self):
        manget_state_vector = self.generate_manget_points_vector()
        return rotate2X(manget_state_vector)

    def generate_manget_points_vector(self):
        manget_state = np.vstack([agent.manget_state() for agent in self.agents])
        manget_state_vector = manget_state - manget_state[0]
        return manget_state_vector


    # def distance_calc(self):


class MS_ENV_NO_Standardization_1D_Motion():
    def __init__(self, space, stands, action_bound):
        self.space = space
        self.agents = stands
        self.action_bound = action_bound
        self.action_space = Action_Space(action_bound)
        self.target_points = np.vstack([agent.target_point_group for agent in self.agents])
        self.target_points_vector = self.target_points - self.target_points[0]
        self.target_vector = rotate2X(self.target_points_vector)

    def reset(self):
        manget_state = self.generate_manget_points_vector()
        for i in range(10):
            for agent in self.agents:
                action = np.random.choice(2)
                agent.body_move(action, (.1, .1))

            manget_state = self.generate_manget_points_vector()
        self.space.step(.001)
        state = self.generate_env_state()
        return state


    def step(self, action):
        action = action.reshape(-1, 1)
        terminal = False
        # COM_TARGET = np.vstack([TARGET1, TARGET2])
        # com_manget_state = np.vstack([agent.manget_state() for agent in self.agents])
        # manget_state_vector = self.generate_manget_points_vector()
        manget_state_vector = self.generate_manget_vector()
        D = np.sqrt(np.sum(np.square(manget_state_vector - self.target_vector)))
        for i , agent in enumerate(self.agents):
            body_state = agent.body_state
            action_a = action[i]
            agent.continuous_body_move(action[i])
            body_state = agent.body_state
        self.space.step(.0001)
        next_state = self.generate_env_state()
        manget_state_vector = self.generate_manget_vector()
        D_ = np.sqrt(np.sum(np.square(manget_state_vector - self.target_vector)))
        D_terminal = np.sqrt(np.sum(np.square(manget_state_vector - self.target_vector), axis=1))
        if np.all(D_terminal < 0.01):
            R = 100
            terminal = True

        elif D_ < D:
            R = 1
        else:
            R = -1
        return next_state, R, terminal

    def generate_env_state(self):
        # manget_state = np.vstack([agent.manget_state() for agent in self.agents])
        manget_state_vector = self.generate_manget_points_vector()
        # Mean = np.mean(manget_state_vector, axis=0)
        # Std = np.std(manget_state_vector, axis=0)
        # normal_manget_state_vector = (manget_state_vector - Mean) / Std
        return manget_state_vector.flatten() / 1000

    def generate_manget_vector(self):
        manget_state_vector = self.generate_manget_points_vector()
        return rotate2X(manget_state_vector)

    def generate_manget_points_vector(self):
        manget_state = np.vstack([agent.manget_state() for agent in self.agents])
        manget_state_vector = manget_state - manget_state[0]
        return manget_state_vector


    # def distance_calc(self):



class SA_MS_ENV_NO_Standardization_1D_Motion():
    def __init__(self, space, stands, action_bound):
        self.space = space
        self.agents = stands
        self.action_bound = action_bound
        self.action_space = Action_Space(action_bound)
        self.target_points = np.vstack([agent.target_point_group for agent in self.agents])
        self.magnet_points_n = self.target_points.shape[0]
        self.d_n = int(self.magnet_points_n * (self.magnet_points_n - 1) / 2)
        # self.target_points_d_matrix = np.zeros((self.d_n, self.magnet_points_n, 2))
        self.sum_d_list = []
        sum = 0
        for i in range(self.magnet_points_n - 1, 0, -1):
            sum += i
            self.sum_d_list.append(sum)
        self.sum_d_list = np.array(self.sum_d_list)
        self.target_points_d_matrix = self.magnet_matrix
        self.target_points_vector = self.target_points - self.target_points[0]
        self.target_vector = rotate2X(self.target_points_vector)
        self.terminal_list = np.zeros(self.d_n)

    def reset(self):
        # manget_state = self.generate_manget_points_vector()
        # for i in range(20):
        #     for agent in self.agents:
        #         action = np.random.choice(6)
        #         agent.body_move(action, (.001, .0001))
        self.agents[0].continuous_body_move([-5,  0, 0])
        self.agents[1].continuous_body_move([-5, -5, 0])
        self.agents[2].continuous_body_move([ 0, -5, 0])
        self.agents[3].continuous_body_move([ 5, -5, 0])
        self.agents[4].continuous_body_move([ 5,  0, 0])
        self.agents[5].continuous_body_move([ 5,  5, 0])
        # manget_state = self.generate_manget_points_vector()
        self.space.step(.001)
        # state = self.generate_env_state()
        # stands_body_states = np.vstack([agent.body_state for agent in self.agents])
        # D = self.state_distance()
        # d_matrix = self.generate_d_matrix()
        # self.generate_terminal_list(d_matrix)
        return self.stand_body_state, self.d_list_np().min()


    def step(self, action):
        # action = action.reshape(-1, 1)
        # terminal = False

        # com_manget_state = np.vstack([agent.manget_state() for agent in self.agents])
        # manget_state_vector = self.generate_manget_points_vector()
        # stands_body_states = np.vstack([agent.body_state for agent in self.agents])
        # D = self.state_distance()
        for i , agent in enumerate(self.agents):
            body_state = agent.body_state
            action_a = action[i]
            agent.continuous_body_move(action[i])
            body_state = agent.body_state
        self.space.step(.0001)
        # d_matrix = self.generate_d_matrix()
        self.generate_terminal_list_np()
        return self.stand_body_state, self.d_list_np().min()

    def go_back(self, action):
        action = -1 * action
        for i , agent in enumerate(self.agents):
            body_state = agent.body_state
            action_a = action[i]
            agent.continuous_body_move(action[i])
            body_state = agent.body_state
        self.space.step(.0001)

    @property
    def stand_body_state(self):
        stands_body_states = np.vstack([agent.body_state for agent in self.agents])
        return stands_body_states

    @property
    def stand_magnet_state(self):
        stand_magnet_state = np.vstack([agent.manget_state() for agent in self.agents])
        return stand_magnet_state

    def update_stands_state(self, body_state):
        for i , agent in enumerate(self.agents):
            state = list(body_state[i])
            agent.body_1.position = state[:2]
            agent.body_1.angle = state[2]
        self.space.step(.0001)


    def generate_env_state(self):
        # manget_state = np.vstack([agent.manget_state() for agent in self.agents])
        manget_state_vector = self.generate_manget_points_vector()
        # Mean = np.mean(manget_state_vector, axis=0)
        # Std = np.std(manget_state_vector, axis=0)
        # normal_manget_state_vector = (manget_state_vector - Mean) / Std
        return manget_state_vector.flatten() / 1000

    def generate_manget_vector(self):
        manget_state_vector = self.generate_manget_points_vector(0)
        return rotate2X(manget_state_vector)

    def generate_manget_points_vector(self, i):
        manget_state = np.vstack([agent.manget_state() for agent in self.agents])
        manget_state_vector = manget_state - manget_state[i]
        return manget_state_vector

    def state_distance(self):
        manget_state_vector = self.generate_manget_vector()
        dis = np.sqrt(np.sum(np.square(manget_state_vector - self.target_vector)))
        return dis

    @property
    def magnet_matrix(self):
        d_matrix = np.zeros((self.d_n, self.magnet_points_n, 2))
        for i in range(self.d_n):
            judge_pos_arr = self.sum_d_list - i - 1
            index = np.argwhere(judge_pos_arr >= 0)[0][0]
            offset = (self.magnet_points_n - 1 - judge_pos_arr[index])
            magnet_state_vector = self.generate_manget_points_vector(index)
            d_matrix[i] = rotate2X_index(magnet_state_vector, offset)
        return d_matrix

    def d_list(self, d_matrix):
        d_list = np.zeros(self.d_n)
        # d_matrix = self.generate_d_matrix()
        for i in range(self.d_n):
            d = np.sqrt(np.sum(np.square(d_matrix[i] - self.target_points_d_matrix[i])))
            d_list[i] = d
        return d_list

    def d_list_np(self):
        # magnet_matrix = self.generate_d_matrix()
        d_matrix = self.magnet_matrix - self.target_points_d_matrix
        return np.sqrt(np.sum(np.sum(np.square(d_matrix), axis=2), axis=1))


    def generate_terminal_list(self, d_matrix, single_point_distance=0.01):
        # terminal_list = np.zeros(self.d_n)
        # d_matrix = self.generate_d_matrix()
        for i in range(self.d_n):
            terminal = np.sqrt(np.sum(np.square(d_matrix[i] - self.target_points_d_matrix[i]), axis=1))
            if np.all(terminal < single_point_distance):
                # R = 100
                self.terminal_list[i] = True
            else:
                self.terminal_list[i] = False

        # return terminal_list
    def generate_terminal_list_np(self, single_point_distance=0.01):
        d_matrix = self.magnet_matrix - self.target_points_d_matrix
        return np.all(np.sqrt(np.sum(np.square(d_matrix), axis=2)) < single_point_distance, axis=1)


    def terminal(self):
        terminal = False
        if np.any(self.terminal_list):
            terminal = True
        return terminal








    def terminal_yes_or_not(self, single_point_distance):
        terminal = False
        manget_state_vector = self.generate_manget_vector()
        D_terminal = np.sqrt(np.sum(np.square(manget_state_vector - self.target_vector), axis=1))
        if np.all(D_terminal < single_point_distance):
            # R = 100
            terminal = True
        return terminal

    # def distance_calc(self):



class SA_MS_ENV():
    def __init__(self, space, stands, action_bound):
        self.space = space
        self.agents = stands
        self.action_bound = action_bound
        self.action_space = Action_Space(action_bound)
        self.target_points = np.vstack([agent.target_point_group for agent in self.agents])
        self.magnet_points_n = self.target_points.shape[0]
        self.d_n = int(self.magnet_points_n * (self.magnet_points_n - 1) / 2)
        # self.target_points_d_matrix = np.zeros((self.d_n, self.magnet_points_n, 2))
        self.sum_d_list = []
        sum = 0
        for i in range(self.magnet_points_n - 1, 0, -1):
            sum += i
            self.sum_d_list.append(sum)
        self.sum_d_list = np.array(self.sum_d_list)
        self.target_points_d_matrix = self.magnet_matrix
        self.target_points_vector = self.target_points - self.target_points[0]
        self.target_vector = rotate2X(self.target_points_vector)
        self.terminal_list = None
        # self.best_body_state = None
        # self.body_state = self.stand_body_state()
        # self.magnet_state = self.stand_magnet_state()
        self.D = 0
        self.space.step(.0001)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def reset(self):
        # manget_state = self.generate_manget_points_vector()
        # for i in range(20):
        #     for agent in self.agents:
        #         action = np.random.choice(6)
        #         agent.body_move(action, (.001, .0001))
        self.agents[0].continuous_body_move([-5,  0, 0])
        self.agents[1].continuous_body_move([-5, -5, 0])
        self.agents[2].continuous_body_move([ 0, -5, 0])
        self.agents[3].continuous_body_move([ 5, -5, 0])
        self.agents[4].continuous_body_move([ 5,  0, 0])
        # self.agents[5].continuous_body_move([ 5,  5, 0])
        # self.agents[0].continuous_body_move([-0.02, 0, 0])
        # self.agents[1].continuous_body_move([-0.02, -0.02, 0])
        # self.agents[2].continuous_body_move([0, -0.02, 0])
        # self.agents[3].continuous_body_move([0.02, -0.02, 0])
        # self.agents[4].continuous_body_move([0.02, 0, 0])
        # self.agents[5].continuous_body_move([0.02, 0.02, 0])
        # manget_state = self.generate_manget_points_vector()
        # self.space.step(.001)
        # state = self.generate_env_state()
        # stands_body_states = np.vstack([agent.body_state for agent in self.agents])
        # D = self.state_distance()
        # d_matrix = self.generate_d_matrix()
        # self.generate_terminal_list(d_matrix)
        # init_body_state = self.stand_body_state
        # self.best_body_state = init_body_state
        d_diff_matrix = self.magnet_matrix - self.target_points_d_matrix
        self.terminal_list = self.generate_terminal_list_np(d_diff_matrix)
        # self.best_body_state = self.stand_body_state
        # self.body_state = self.stand_body_state()
        # self.magnet_state = self.stand_magnet_state()
        self.D = self.d_list_np(d_diff_matrix).min()
        # return self.D


    def step(self, action):
        # action = action.reshape(-1, 1)
        # terminal = False

        # com_manget_state = np.vstack([agent.manget_state() for agent in self.agents])
        # manget_state_vector = self.generate_manget_points_vector()
        # stands_body_states = np.vstack([agent.body_state for agent in self.agents])
        # D = self.state_distance()
        for i , agent in enumerate(self.agents):
            # body_state = agent.body_state
            # action_a = action[i]
            agent.continuous_body_move(action[i])
            # body_state = agent.body_state
        ### self.space.step(.0001)
        # d_matrix = self.generate_d_matrix()
        d_diff_matrix = self.magnet_matrix - self.target_points_d_matrix
        self.terminal_list = self.generate_terminal_list_np(d_diff_matrix)
        # self.body_state = self.stand_body_state()
        # self.magnet_state = self.stand_magnet_state()
        self.D = self.d_list_np(d_diff_matrix).min()

    def step_ray_cuda(self, action):
        for i , agent in enumerate(self.agents):
            # body_state = agent.body_state
            # action_a = action[i]
            agent.continuous_body_move(action[i])

    def go_back(self, action):
        action = -1 * action
        for i , agent in enumerate(self.agents):
            body_state = agent.body_state
            action_a = action[i]
            agent.continuous_body_move(action[i])
            body_state = agent.body_state
        # self.space.step(.0001)

    @property
    def stand_body_state(self):
        stands_body_states = np.vstack([agent.body_state() for agent in self.agents])
        return stands_body_states

    @property
    def stand_magnet_state(self):
        stand_magnet_state = np.vstack([agent.manget_state() for agent in self.agents])
        return stand_magnet_state

    @property
    def stand_target_state(self):
        stand_target_state = np.vstack([agent.target_state() for agent in self.agents])
        return stand_target_state

    def update_stands_state(self, body_state):
        for i , agent in enumerate(self.agents):
            state = list(body_state[i])
            agent.body_1.position = state[:2]
            agent.body_1.angle = state[2]
        # self.space.step(.0001)
        d_diff_matrix = self.magnet_matrix - self.target_points_d_matrix
        # self.terminal_list = self.generate_terminal_list_np(d_diff_matrix)
        self.D = self.d_list_np(d_diff_matrix).min()


    def generate_env_state(self):
        # manget_state = np.vstack([agent.manget_state() for agent in self.agents])
        manget_state_vector = self.generate_manget_points_vector()
        # Mean = np.mean(manget_state_vector, axis=0)
        # Std = np.std(manget_state_vector, axis=0)
        # normal_manget_state_vector = (manget_state_vector - Mean) / Std
        return manget_state_vector.flatten() / 1000

    def generate_manget_vector(self):
        manget_state_vector = self.generate_manget_points_vector(0)
        return rotate2X(manget_state_vector)

    def generate_manget_points_vector(self, i):
        manget_state = np.vstack([agent.manget_state() for agent in self.agents])
        manget_state_vector = manget_state - manget_state[i]
        return manget_state_vector

    def state_distance(self):
        manget_state_vector = self.generate_manget_vector()
        dis = np.sqrt(np.sum(np.square(manget_state_vector - self.target_vector)))
        return dis

    @property
    def magnet_matrix(self):
        d_matrix = np.zeros((self.d_n, self.magnet_points_n, 2))
        for i in range(self.d_n):
            judge_pos_arr = self.sum_d_list - i - 1
            index = np.argwhere(judge_pos_arr >= 0)[0][0]
            offset = (self.magnet_points_n - 1 - judge_pos_arr[index])
            magnet_state_vector = self.generate_manget_points_vector(index)
            d_matrix[i] = rotate2X_index(magnet_state_vector, offset)
        return d_matrix

    def d_list(self, d_matrix):
        d_list = np.zeros(self.d_n)
        # d_matrix = self.generate_d_matrix()
        for i in range(self.d_n):
            d = np.sqrt(np.sum(np.square(d_matrix[i] - self.target_points_d_matrix[i])))
            d_list[i] = d
        return d_list

    def d_list_np(self, d_matrix):
        # magnet_matrix = self.generate_d_matrix()
        # d_matrix = self.magnet_matrix - self.target_points_d_matrix
        return np.sqrt(np.sum(np.sum(np.square(d_matrix), axis=2), axis=1))


    def generate_terminal_list(self, d_matrix, single_point_distance=0.01):
        # terminal_list = np.zeros(self.d_n)
        # d_matrix = self.generate_d_matrix()
        for i in range(self.d_n):
            terminal = np.sqrt(np.sum(np.square(d_matrix[i] - self.target_points_d_matrix[i]), axis=1))
            if np.all(terminal < single_point_distance):
                # R = 100
                self.terminal_list[i] = True
            else:
                self.terminal_list[i] = False

        # return terminal_list
    def generate_terminal_list_np(self, d_diff_matrix, single_point_distance=0.005):
        # d_matrix = self.magnet_matrix - self.target_points_d_matrix
        # d_diff_matrix = self.magnet_matrix - self.target_points_d_matrix
        # self.terminal_list = self.generate_terminal_list_np(d_diff_matrix, single_point_distance)
        return np.all(np.sqrt(np.sum(np.square(d_diff_matrix), axis=2)) < single_point_distance, axis=1)


    def terminal(self):
        terminal = False
        if np.any(self.terminal_list):
            terminal = True
        return terminal

    def terminal_yes_or_not(self, single_point_distance):
        terminal = False
        manget_state_vector = self.generate_manget_vector()
        D_terminal = np.sqrt(np.sum(np.square(manget_state_vector - self.target_vector), axis=1))
        if np.all(D_terminal < single_point_distance):
            # R = 100
            terminal = True
        return terminal

    # def distance_calc(self):


class DQN_MS_ENV():
    def __init__(self, space, stands, target_magnet_points, step_length=(0.01, 0.000002)):
        self.space = space
        self.agents = stands
        # self.action_bound = action_bound
        # self.action_space = Action_Space(action_bound)
        self.target_magnet_points = target_magnet_points
        self.target_points = np.vstack([agent.target_point_group for agent in self.agents])
        self.magnet_points_n = self.target_points.shape[0]
        self.d_n = int(self.magnet_points_n * (self.magnet_points_n - 1) / 2)
        # self.target_points_d_matrix = np.zeros((self.d_n, self.magnet_points_n, 2))
        self.sum_d_list = []
        sum = 0
        for i in range(self.magnet_points_n - 1, 0, -1):
            sum += i
            self.sum_d_list.append(sum)
        self.sum_d_list = np.array(self.sum_d_list)
        self.target_points_d_matrix = self.get_target_magnet_matrix()
        self.target_points_vector = self.target_points - self.target_points[0]
        self.target_vector = rotate2X(self.target_points_vector)
        self.terminal_list = None
        # self.best_body_state = None
        # self.body_state = self.stand_body_state()
        # self.magnet_state = self.stand_magnet_state()
        self.terminal_matrix = np.zeros((self.d_n, self.magnet_points_n, 2))
        self.terminal_matrix[:, :30, :] = np.array([0.0110, 0.0108])
        self.terminal_matrix[:, 30:, :] = np.array([0.010, 0.010])
        d_diff_matrix = self.magnet_matrix - self.target_points_d_matrix
        self.terminal_list = np.all(np.all(abs(d_diff_matrix) < self.terminal_matrix, axis=1), axis=1)
        D = self.d_list_np(d_diff_matrix).min()
        self.D = D


        # self.target_points = np.vstack([agent.target_point_group for agent in self.agents])
        # self.target_points_vector = self.target_points - self.target_points[0]
        # self.target_vector = rotate2X(self.target_points_vector)
        self.init_stand_body_state = np.vstack([agent.body_state for agent in self.agents])
        self.step_length = step_length
        self.space.step(.00001)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def reset(self):
        # self.set_stand_body_state()
        # manget_state = self.generate_manget_points_vector()
        self.agents[0].continuous_body_move([-0.02, 0, 0])
        self.agents[1].continuous_body_move([-0.02, -0.02, 0])
        self.agents[2].continuous_body_move([0, -0.02, 0])
        self.agents[3].continuous_body_move([0.02, -0.02, 0])
        self.agents[4].continuous_body_move([0.02, 0, 0])
        # self.agents[5].continuous_body_move([5, 5, 0])
        # manget_state = self.generate_manget_points_vector()
        # self.space.step(.001)
        # self.space.step(.001)
        state = self.generate_env_state()
        d_diff_matrix = self.magnet_matrix - self.target_points_d_matrix
        # self.terminal_list = self.generate_terminal_list_np(d_diff_matrix)
        # self.body_state = self.stand_body_state()
        # self.magnet_state = self.stand_magnet_state()
        D = self.d_list_np(d_diff_matrix).min()
        self.D = D
        return state


    def step(self, action):
        # action = action
        terminal = False
        agent_n = action // 6
        action_n = action % 6

        # COM_TARGET = np.vstack([TARGET1, TARGET2])
        # com_manget_state = np.vstack([agent.manget_state for agent in self.agents])
        # manget_state_vector = self.generate_manget_points_vector()
        # manget_state_vector = self.generate_manget_vector()
        # D = np.sqrt(np.sum(np.square(manget_state_vector - self.target_vector)))
        d_diff_matrix = self.magnet_matrix - self.target_points_d_matrix
        # self.terminal_list = self.generate_terminal_list_np(d_diff_matrix)
        # self.body_state = self.stand_body_state()
        # self.magnet_state = self.stand_magnet_state()
        D = self.d_list_np(d_diff_matrix).min()
        # for i , agent in enumerate(self.agents):
        #     body_state = agent.body_state
        #     action_a = action[i]
        #     agent.continuous_body_move(action[i])
        #     body_state = agent.body_state
        self.agents[agent_n].single_step_move(action_n, self.step_length)
        # self.space.step(.0001)
        # print(self.stand_body_state)
        # print(self.state_distance())
        next_state = self.generate_env_state()
        d_diff_matrix = self.magnet_matrix - self.target_points_d_matrix
        # manget_state_vector = self.generate_manget_vector()
        D_ = self.d_list_np(d_diff_matrix).min()
        # self.terminal_list = self.generate_terminal_list_np(d_diff_matrix)
        # D_terminal = np.sqrt(np.sum(np.square(manget_state_vector - self.target_vector), axis=1))
        self.terminal_list = np.all(np.all(abs(d_diff_matrix) < self.terminal_matrix, axis=1), axis=1)
        self.D = D_
        if self.terminal():
            R = 100
            terminal = True

        elif D_ < D:
            R = 1
        else:
            R = -1
        return next_state, R, terminal

    # def set_stand_body_state(self):
    #     for i in range(len(self.agents)):
    #         self.agents[i].set_body_state(self.init_stand_body_state[i])

    @property
    def stand_body_state(self):
        stands_body_states = np.vstack([agent.body_state() for agent in self.agents])
        return stands_body_states

    @property
    def stand_magnet_state(self):
        stand_magnet_state = np.vstack([agent.manget_state() for agent in self.agents])
        return stand_magnet_state

    @property
    def stand_target_state(self):
        stand_target_state = np.vstack([agent.target_state() for agent in self.agents])
        return stand_target_state

    def update_stands_state(self, body_state):
        for i, agent in enumerate(self.agents):
            state = list(body_state[i])
            agent.body_1.position = state[:2]
            agent.body_1.angle = state[2]
        # self.space.step(.0001)
        d_diff_matrix = self.magnet_matrix - self.target_points_d_matrix
        # self.terminal_list = self.generate_terminal_list_np(d_diff_matrix)
        self.D = self.d_list_np(d_diff_matrix).min()

    def generate_env_state(self):
        # manget_state = np.vstack([agent.manget_state for agent in self.agents])
        # manget_state_vector = self.generate_manget_points_vector()
        return self.stand_magnet_state.flatten()

    def generate_manget_vector(self):
        manget_state_vector = self.generate_manget_points_vector()
        return rotate2X(manget_state_vector)

    def generate_manget_points_vector(self, i):
        # manget_state = np.vstack([agent.manget_state() for agent in self.agents])
        manget_state_vector = self.stand_magnet_state - self.stand_magnet_state[i]
        return manget_state_vector

    def state_distance(self):
        manget_state_vector = self.generate_manget_vector()
        dis = np.sqrt(np.sum(np.square(manget_state_vector - self.target_vector)))
        return dis

    @property
    def magnet_matrix(self):
        d_matrix = np.zeros((self.d_n, self.magnet_points_n, 2))
        for i in range(self.d_n):
            judge_pos_arr = self.sum_d_list - i - 1
            index = np.argwhere(judge_pos_arr >= 0)[0][0]
            offset = (self.magnet_points_n - 1 - judge_pos_arr[index])
            magnet_state_vector = self.generate_manget_points_vector(index)
            d_matrix[i] = rotate2X_index(magnet_state_vector, offset)
        return d_matrix

    def get_target_magnet_matrix(self):
        d_matrix = np.zeros((self.d_n, self.magnet_points_n, 2))
        for i in range(self.d_n):
            judge_pos_arr = self.sum_d_list - i - 1
            index = np.argwhere(judge_pos_arr >= 0)[0][0]
            offset = (self.magnet_points_n - 1 - judge_pos_arr[index])
            magnet_state_vector = self.target_magnet_points - self.target_magnet_points[index]
            d_matrix[i] = rotate2X_index(magnet_state_vector, offset)
        return d_matrix

    def d_list(self, d_matrix):
        d_list = np.zeros(self.d_n)
        # d_matrix = self.generate_d_matrix()
        for i in range(self.d_n):
            d = np.sqrt(np.sum(np.square(d_matrix[i] - self.target_points_d_matrix[i])))
            d_list[i] = d
        return d_list

    def d_list_np(self, d_matrix):
        # magnet_matrix = self.generate_d_matrix()
        # d_matrix = self.magnet_matrix - self.target_points_d_matrix
        return np.sqrt(np.sum(np.sum(np.square(d_matrix), axis=2), axis=1))

    def generate_terminal_list(self, d_matrix, single_point_distance=0.005):
        # terminal_list = np.zeros(self.d_n)
        # d_matrix = self.generate_d_matrix()
        for i in range(self.d_n):
            terminal = np.sqrt(np.sum(np.square(d_matrix[i] - self.target_points_d_matrix[i]), axis=1))
            if np.all(terminal < single_point_distance):
                # R = 100
                self.terminal_list[i] = True
            else:
                self.terminal_list[i] = False

        # return terminal_list

    def generate_terminal_list_np(self, d_diff_matrix, single_point_distance=0.005):
        # d_matrix = self.magnet_matrix - self.target_points_d_matrix
        # d_diff_matrix = self.magnet_matrix - self.target_points_d_matrix
        # self.terminal_list = self.generate_terminal_list_np(d_diff_matrix, single_point_distance)
        return np.all(np.sqrt(np.sum(np.square(d_diff_matrix), axis=2)) < single_point_distance, axis=1)

    def terminal(self):
        terminal = False
        if np.any(self.terminal_list):
            terminal = True
        return terminal

    def terminal_yes_or_not(self, single_point_distance):
        terminal = False
        manget_state_vector = self.generate_manget_vector()
        D_terminal = np.sqrt(np.sum(np.square(manget_state_vector - self.target_vector), axis=1))
        if np.all(D_terminal < single_point_distance):
            # R = 100
            terminal = True
        return terminal

    # def distance_calc(self):





