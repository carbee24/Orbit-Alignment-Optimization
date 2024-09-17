from SA4StandEnv import Climbing
from ENV_stand import SA_MS_ENV
from Stand_1d_motion import newStand6_Template, newStand8_1_Template, newStand8_2_Template
from stand_action_bound import  *


if __name__ == "__main__":
    import pymunk

    import time

    import logging

    import pandas as pd

    # import os

    # import ray

    # ray.init()

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # import ray

    # ray.init(num_gpus=1)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建一个文件处理器
    file_handler = logging.FileHandler('opt_climbing_log.txt', 'w')

    # 创建一个日志格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 将文件处理器添加到Logger中
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    space = pymunk.Space()  # 2
    
    ### 生成stand对象输入实测坐标值
    stand1 = newStand6_Template(space, adjust_point_group=new_adjust_points1, manget_point_group=mes_magnet_points1,target_points=mes_target_points1, distance=10)
    stand2 = newStand8_1_Template(space, adjust_point_group=new_adjust_points2, manget_point_group=mes_magnet_points2,target_points=mes_target_points2, distance=10)
    stand3 = newStand8_2_Template(space, adjust_point_group=new_adjust_points3, manget_point_group=mes_magnet_points3,target_points=mes_target_points3, distance=10)
    stand4 = newStand8_1_Template(space, adjust_point_group=new_adjust_points4, manget_point_group=mes_magnet_points4,target_points=mes_target_points4, distance=10)
    stand5 = newStand8_2_Template(space, adjust_point_group=new_adjust_points5, manget_point_group=mes_magnet_points5,target_points=mes_target_points5, distance=10)
    action_bound = np.array([[.01, .01, 0.0000002],
                             [.01, .01, 0.0000002],
                             [.01, .01, 0.0000002],
                             [.01, .01, 0.0000002],
                             [.01, .01, 0.0000002],
                             # [.01, .01, 0.0000002],
                             ])
    
    stands = [stand1, stand2, stand3, stand4, stand5]
    ### 目标磁铁坐标为理论值，用于对比
    # target_magnet_points = np.vstack([input_magnet_points1, input_magnet_points2, input_magnet_points3, input_magnet_points4, input_magnet_points5])
    # target_magnet_points = magnet_points + [-1.099125713856000846e-03, -3.300000666386601264e-03]
    target_magnet_points = magnet_points
    
    env = SA_MS_ENV(space, stands, action_bound, target_magnet_points)
    
    sa = Climbing(env)
    
    start_time = time.time()
   
    distances, best_env = sa.ray_torch_multi_run(logger)
    end_time = time.time()
    # print("Climbing Cost {} mins..".format((end_time - start_time)/60))
    logger.info("Climbing Cost {} mins..".format((end_time - start_time)/60))
    
    magnet_points_name_df = pd.DataFrame(magnet_points_name)
    target_points_name_df = pd.DataFrame(target_points_name)
    target_points_z_df = pd.DataFrame(target_points_z)
    best_magnet_state_df = pd.DataFrame(best_env.stand_magnet_state)
    best_target_state_df = pd.DataFrame(best_env.stand_target_state)
    magnet_sheet = pd.concat([magnet_points_name_df, best_magnet_state_df], axis=1)
    target_sheet = pd.concat([target_points_name_df, best_target_state_df, target_points_z_df], axis=1)
    with pd.ExcelWriter("./climbing_opt_result.xlsx", engine='openpyxl') as writer:
        target_sheet.to_excel(writer, sheet_name='Sheet1', index=False, float_format='%.4f')
        magnet_sheet.to_excel(writer, sheet_name='Sheet2', index=False, float_format='%.4f')
    np.save("climbing_distances_opt.npy", distances)
