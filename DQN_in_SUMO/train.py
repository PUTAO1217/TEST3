import gymnasium as gym
import traci
import os
import sys
import time as tm
import pandas as pd
import torch
# from torch.utils.tensorboard import SummaryWriter
from dqn import Agent
import sumolib
import save_and_load as sl
import environment as env
from environment import get_reward
from environment import get_custom_state
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)
from sumolib import checkBinary
import optparse

# 设置结果保存路径(在output_model文件夹中以当前时间为文件名生成一个文件夹)
saved_directory = sl.assign_train_directory("output_model")
# 保存的每个episode的平均奖励
reward_data = pd.DataFrame(columns=["Episode", "Average_Reward"])
# 设置记录器，可以实时查看reward曲线 (具体做法：运行该程序后，在终端命令行输入"tensorboard --logdir=runs")
# writer = SummaryWriter()


# 定义训练的总轮数和每轮运行的仿真步数
NUM_EPISODES = 200
NUM_STEPS = 500

# 连接SUMO仿真
def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

if __name__ == "__main__":
    options = get_options()
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    traci.start([sumoBinary, "-c", r"G:\PycharmProjects\DQN_sim\DQN_in_SUMO\intersection\test2.sumocfg"])


class SumoEnv(gym.Env):
    def __init__(self, config_file):
        # 初始化仿真环境，加载SUMO配置文件
        self.sumoBinary = sumolib.checkBinary('sumo-gui')
        self.sumoConfig = config_file
        self.sumoCmd = [self.sumoBinary, "-c", self.sumoConfig]
        traci.start(self.sumoCmd)
# 配置仿真环境，获取状态和动作维数
# ********************************************************
env = SumoEnv( r"G:\PycharmProjects\DQN_sim\DQN_in_SUMO\intersection\test2.sumocfg")
# 重置仿真环境并获取初始状态
state, info = env.reset()
state = get_custom_state()
s_dim = len(get_custom_state)
a_dim = env.action_space
reward = get_reward()
# ********************************************************

# 初始化agent (所有超参数都在这里修改, 具体含义见"dqn.py")
agent = Agent(s_dim, a_dim, gamma=0.99, memory_capacity=20000, batch_size=256, learning_rate=0.001,
              tau=0.005, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=1000)


start = tm.time()  # 记录程序开始时间
print("开始训练 ...")

for i_episode in range(NUM_EPISODES):
    total_reward = 0  # 统计当前episode的总奖励
    state_array, _ = env.reset()  # 在每一轮训练开始时初始化环境，并获取本轮的初始state (该state必须是一维的numpy.array)
    state = torch.tensor(state_array, dtype=torch.float32, device=agent.device).unsqueeze(0)  # 将状态变量转为torch.tensor
    old_action = None
    step = 1
    for t in range(NUM_STEPS):
        traci.simulationStep()
        action = agent.choose_action(state)  # 该action为torch.tensor类型
        if action != old_action:
            if old_action is not None:
                # 先将旧相位的信号灯设置为黄灯
                traci.trafficlight.setPhase("JO", old_action + 2)
                tm.sleep(3)  # 等待3秒钟，使黄灯显示完毕
            # 根据动作设置新的相位
            if action == 1:  # 南北方向绿灯，东西方向红灯
                traci.trafficlight.setPhase("JO", 0)
            else:  # 东西方向绿灯，南北方向红灯
                traci.trafficlight.setPhase("JO", 2)
            old_action = action
        reward = get_reward()

        # 此处需要替换成自己的代码，包括将动作值输入进仿真，获取奖励值，并判断该episode是否完成
        # *********************************************************************************************************
        observation, reward, terminated, truncated, _ = env.step(action.item())

        total_reward += reward

        # action.item()将torch.tensor转为一个int，将该数据输入环境
        reward = torch.tensor([reward], device=agent.device)  # 将获得的奖励转为torch.tensor
        done = terminated or truncated  # 当前episode是否结束
        # *********************************************************************************************************


        if terminated:
            next_state = None
        else:
            # 如果该episode未结束，需要在仿真更新一步后获取最新的状态
            # *************************************************************************************************
            traci.simulationStep()
            # 如果当前episode未结束，将next_state转为torch.tensor
            next_state = torch.tensor(observation, dtype=torch.float32, device=agent.device).unsqueeze(0)
            # *************************************************************************************************

        # 存储SARS，前三个量都是torch.tensor，next_state可能是torch.tensor，也可能是None
        agent.store_transition(state, action, reward, next_state)

        # 将下一状态转为当前状态，以备循环下一个step时使用
        state = next_state
        # agent训练一次
        agent.learn()
        # 如果episode结束，就开始下一次episode
        step += 1
        if done:
            break
    average_reward = total_reward / step
    # writer.add_scalar("Average Reward", average_reward, i_episode + 1)
    print(f"第{i_episode + 1}轮, 平均奖励: {average_reward: .2f}, "
          f"经验池[{len(agent.memory)}/{agent.memory_capacity}]")
    new_reward_data = pd.DataFrame([[i_episode + 1, average_reward]], columns=["Episode", "Average_Reward"])
    reward_data = pd.concat([reward_data, new_reward_data])

# 保存模型和结果
sl.save_model(agent, reward_data, saved_directory)
# writer.close()


print("\n训练完成，", end="")

end = tm.time()  # 记录程序结束时间
# 统计程序运行时间
duration = round(end - start)
if duration < 60:
    print(f"总运行时间: {duration:.0f}s")
elif duration < 3600:
    print(f"总运行时间: {duration // 60}m {duration % 60:.0f}s")
else:
    print(f"总运行时间: {duration // 3600}h {(duration % 3600) // 60}m {duration % 60:.0f}s")