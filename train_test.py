def train(cfg, env, agent, logger):
    ''' 训练
    '''
    # print("开始训练！")
    logger.info("开始优化！")
    distance = []  # 记录所有回合的奖励
    # init_stand_body_state = env.stand_body_state
    # init_stand_distance = env.D
    # print(env.stand_body_state)
    # print(env.state_distance())
    # distance.append(env.state_distance())
    # state = env.reset()
    state = env.generate_env_state()
    init_stand_state = env.stand_body_state
    init_D = env.D
    logger.info(env.D)
    distance.append(env.D)
    # steps = []
    # i_ep = 0
    # ep_step = 0
    # ep_terminal = False
    reach_distance_2 = False
    reach_distance_dot2 = False
    if not env.terminal():
        terminal = False
        while True:
            if terminal:
                break
            if env.D <= 2 and not reach_distance_2:
                env.step_length = (0.001, 0.000002)
                reach_distance_2 = True
            if env.D <= .15 and not reach_distance_dot2:
                env.step_length = (0.0001, 0.0000002)
                reach_distance_dot2 = True
            for i in range(100):


                action = agent.sample_action(state)  # 选择动作
                # print(action)
                next_state, reward, done = env.step(action)  # 更新环境，返回transition
                agent.memory.push((state, action, reward, next_state, done))  # 保存transition
                state = next_state  # 更新下一个状态
                agent.update()  # 更新智能体
                # ep_reward += reward  # 累加奖励
                terminal = done
                if done:
                    break
                if (i + 1) % 50 == 0:  # 智能体目标网络更新
                    agent.target_net.load_state_dict(agent.policy_net.state_dict())
            logger.info(env.D)
            distance.append(env.D)
            # steps.append(ep_step)
            # rewards.append(ep_reward)
            # if (i_ep + 1) % 10 == 0:
            #     print(f"回合：{i_ep+1}/{cfg['train_eps']}，奖励：{ep_reward:.2f}，Epislon：{agent.epsilon:.3f}")

    # print("完成训练！")
    logger.info("完成训练！")
    # print(init_stand_body_state)
    # print(init_stand_distance)
    # print(env.stand_body_state)
    # print(env.state_distance())
    logger.info("Init Body State: {}".format(init_stand_state))
    # print("Init Fitness: ", init_D)
    logger.info("Init Fitness: {}".format(init_D))
    # print("Best Body State: ", best_body_state)
    logger.info("Best Body State: {}".format(env.stand_body_state))
    # print("Best Fitness: ", f_best)
    logger.info("Best Fitness: {}".format(env.D))
    logger.info("Best env magnet state: {}".format(env.stand_magnet_state))
    logger.info("Best env target state: {}".format(env.stand_target_state))
    # env.close()
    distance.append(env.D)
    return {'distance': distance}

def one_ep_train(cfg, env, agent):
    print("开始训练！")
    distance = []  # 记录所有回合的奖励
    ep_step = 0
    state = env.reset()
    init_stand_body_state = env.stand_body_state
    init_stand_distance = env.state_distance()
    print(env.stand_body_state)
    print(env.state_distance())
    distance.append(env.state_distance())
    while True:
        ep_step += 1
        action = agent.sample_action(state)  # 选择动作
        # print(action)
        next_state, reward, done = env.step(action)  # 更新环境，返回transition
        agent.memory.push((state, action, reward, next_state, done))  # 保存transition
        state = next_state  # 更新下一个状态
        if env.state_distance() <= 1:
            print("Reach state 1")
            env.step_length = (0.0001, 0.00000002)
        agent.update()
        if ep_step % cfg["ep_max_steps"] == 0:
            # agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print(env.stand_body_state)
            print(env.state_distance())
            print("epsilon: ", agent.epsilon)
            distance.append(env.state_distance())
        if ep_step % (cfg["ep_max_steps"] * 10) == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        if done:
            break
    print("完成训练！")
    print(init_stand_body_state)
    print(init_stand_distance)
    print(env.stand_body_state)
    print(env.state_distance())
    distance.append(env.state_distance())
    # env.close()
    return {'distance': distance}



def test(cfg, env, agent):
    print("开始测试！")
    rewards = []  # 记录所有回合的奖励
    steps = []
    for i_ep in range(cfg['test_eps']):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        for _ in range(cfg['ep_max_steps']):
            ep_step+=1
            action = agent.predict_action(state)  # 选择动作
            next_state, reward, done = env.step(action)  # 更新环境，返回transition
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            if done:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        print(f"回合：{i_ep+1}/{cfg['test_eps']}，奖励：{ep_reward:.2f}")
    print("完成测试")
    # env.close()
    return {'rewards':rewards}