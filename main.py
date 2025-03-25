from config import *
from train_test import *
from Agent import DQN
from ReplayBuffer import ReplayBuffer
from Model import MLP
import pymunk
from ENV_stand import DQN_MS_ENV
from Stand_motion import newStand6_Template, newStand8_1_Template, newStand8_2_Template
from stand_action_bound import  *
import time
import logging
import pickle
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
# 创建一个文件处理器
file_handler = logging.FileHandler('dqn_opt.txt', 'w')
# 创建一个日志格式化器
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
# 将文件处理器添加到Logger中
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 获取参数
cfg = get_args()
# 训练

# import time
space = pymunk.Space()  # 2
stand1 = newStand6_Template(space, adjust_point_group=new_adjust_points1, manget_point_group=mes_magnet_points1, target_points=mes_target_points1, distance=10)
stand2 = newStand8_1_Template(space, adjust_point_group=new_adjust_points2, manget_point_group=mes_magnet_points2, target_points=mes_target_points2, distance=10)
stand3 = newStand8_2_Template(space, adjust_point_group=new_adjust_points3, manget_point_group=mes_magnet_points3, target_points=mes_target_points3, distance=10)
stand4 = newStand8_1_Template(space, adjust_point_group=new_adjust_points4, manget_point_group=mes_magnet_points4, target_points=mes_target_points4, distance=10)
stand5 = newStand8_2_Template(space, adjust_point_group=new_adjust_points5, manget_point_group=mes_magnet_points5, target_points=mes_target_points5, distance=10)
stands = [stand1, stand2, stand3, stand4, stand5]
target_magnet_points = magnet_points
# target_magnet_points = np.vstack([new_magnet_points1, new_magnet_points2, new_magnet_points3, new_magnet_points4, new_magnet_points5])
env = DQN_MS_ENV(space, stands, target_magnet_points)
print(len(env.terminal_list))

if cfg['seed'] != 0:
    all_seed(env, seed=cfg['seed'])
n_states = len(env.target_points) * 2
n_actions = (len(env.agents) - 1) * 6
logger.info(f"状态空间维度：{n_states}，动作空间维度：{n_actions}")
cfg.update({"n_states": n_states, "n_actions": n_actions})  # 更新n_states和n_actions到cfg参数中
model = MLP(n_states, n_actions, hidden_dim=cfg['hidden_dim'])  # 创建模型
memory = ReplayBuffer(cfg['memory_capacity'])
agent = DQN(model, memory, cfg)
start_time = time.time()
dis_dic = train(cfg, env, agent, logger)
end_time = time.time()
# print("Training cost {} minutes".format((end_time - start_time)/60))
logger.info("Training cost {} minutes".format((end_time - start_time)/60))
with open("dqn_env_opt_10_10_10p8_11_20240826.pickle", "wb") as file:
    pickle.dump(env, file)
np.save("dqn_distance_opt_10_10_10p8_11_20240826.npy", dis_dic['distance'])
# plot_rewards(dis_dic['distance'], cfg, tag="train")
# 测试
# res_dic = test(cfg, env, agent)
# plot_rewards(res_dic['rewards'], cfg, tag="test")  # 画出结果
magnet_points_name_df = pd.DataFrame(magnet_points_name)
target_points_name_df = pd.DataFrame(target_points_name)
target_points_z_df = pd.DataFrame(target_points_z)
best_magnet_state_df = pd.DataFrame(env.stand_magnet_state)
best_target_state_df = pd.DataFrame(env.stand_target_state)
magnet_sheet = pd.concat([magnet_points_name_df, best_magnet_state_df], axis=1)
target_sheet = pd.concat([target_points_name_df, best_target_state_df, target_points_z_df], axis=1)
with pd.ExcelWriter("./dqn_opt_result.xlsx", engine='openpyxl') as writer:
    target_sheet.to_excel(writer, sheet_name='Sheet1', index=False, float_format='%.4f')
    magnet_sheet.to_excel(writer, sheet_name='Sheet2', index=False, float_format='%.4f')