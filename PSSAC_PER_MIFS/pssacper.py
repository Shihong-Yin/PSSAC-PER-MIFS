#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" parameter-sharing soft actor-critic algorithm with prioritized experience replay """

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import rl_utils
import env_utils
from environment import Environment  # Environment类
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Arrow
import matplotlib.cm as cm
import time


class SumTree:
    """ SumTree 数据结构，用于高效地实现优先经验回放 """
    def __init__(self, capacity):
        self.capacity = capacity  # 样本容量
        self.tree = np.zeros(2 * capacity - 1)  # 存储优先级的树
        self.data = np.zeros(capacity, dtype=object)  # 存储实际经验
        self.write = 0  # 写指针
        self.n_entries = 0  # 当前存储的经验数量

    def _propagate(self, idx, change):
        """更新父节点的值"""
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, priority):
        """更新某个叶子节点的优先级"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority, data):
        """添加新的经验"""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def _retrieve(self, idx, s):
        """根据累积概率检索样本"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def get(self, s):
        """获取样本及其索引和优先级"""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    """ 优先经验回放缓冲区 """
    def __init__(self, capacity, alpha=0.6):
        """
        参数:
            capacity (int): 缓冲区容量
            alpha (float): 优先级参数，0 表示完全随机，1 表示完全基于优先级
        """
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.capacity = capacity
        self.epsilon = 1e-6  # 避免优先级为0

    def add(self, transition, priority=None):
        """添加经验，并设置优先级"""
        if priority is None:
            priority = self.tree.tree[-self.tree.capacity:].max() if self.tree.n_entries > 0 else 1.0
        priority = (priority + self.epsilon) ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size, beta=0.4):
        """根据优先级采样经验，并计算重要性采样权重"""
        batch = []
        idxs = []
        segment = self.tree.total_priority / batch_size
        priorities = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)  # # #
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        sampling_probabilities = priorities / self.tree.total_priority
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update_priorities(self, idxs, priorities):
        """更新样本的优先级"""
        for idx, priority in zip(idxs, priorities):
            priority = (priority + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


# 定义Actor和Critic网络
class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # 动作的取值范围
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # 使用重参数化技巧
        y_t = torch.tanh(x_t)
        action = y_t * self.action_bound
        # 计算log_prob
        log_prob = normal.log_prob(x_t)
        # 计算动作的熵调整
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        cat = torch.cat([state, action], dim=-1)  # 将状态和动作拼接
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        q_value = self.fc_out(x)
        return q_value


# 定义PSSAC Agent
class PSSACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, action_bound, actor_lr, critic_lr, tau, gamma,
                 buffer_size, minimal_size, batch_size, device, max_grad_norm=1.0,
                 alpha_lr=3e-4, target_entropy=None,
                 alpha=0.6, beta_start=0.4, beta_frames=100000):
        # 每个智能体的策略网络和价值网络都是独立的
        self.actor = Actor(state_dim, hidden_dim, action_dim, action_bound).to(device)  # 实例化策略网络
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)  # 实例化价值网络
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)  # 实例化价值网络
        self.target_critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)

        # 初始化目标网络
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        # self.action_dim = action_dim
        # 温度参数α及其优化器
        if target_entropy is None:
            self.target_entropy = -action_dim  # 默认目标熵
        else:
            self.target_entropy = target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.alpha = self.log_alpha.exp()

        self.gamma = gamma  # 折扣系数
        # self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_bound = action_bound
        self.max_grad_norm = max_grad_norm  # 设置最大梯度范数
        self.device = device

        # 经验回放缓冲区
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size, alpha=alpha)
        self.minimal_size = minimal_size
        self.batch_size = batch_size

        # PER 相关
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # 用于线性增加beta

        # 用于存储损失值的列表
        self.actor_losses = []
        self.critic1_losses = []
        self.critic2_losses = []
        self.alpha_losses = []

    def get_beta(self):
        """线性增加beta值"""
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        return beta

    # 动作选择
    def take_action(self, state, deterministic=False):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        self.actor.eval()  # 设置为评估模式
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor.forward(state)
                action = torch.tanh(mean) * self.action_bound
                log_prob = None
            else:
                action, log_prob = self.actor.sample(state)
        self.actor.train()  # 设置回训练模式
        action = action.cpu().numpy()[0]
        if log_prob is not None:
            log_prob = log_prob.cpu().numpy()[0]
        return action, log_prob

    # 软更新目标网络
    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # 训练
    def update(self):
        if len(self.replay_buffer) < self.minimal_size:
            return  # 当经验不足时，不进行训练

        # 获取当前 beta 值
        beta = self.get_beta()

        # 从经验回放缓冲区中采样
        transitions, idxs, is_weights = self.replay_buffer.sample(self.batch_size, beta)

        # 转换为张量
        states = torch.tensor(np.stack([t[0] for t in transitions]), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.stack([t[1] for t in transitions]), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array([t[2] for t in transitions]), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.stack([t[3] for t in transitions]), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array([t[4] for t in transitions]), dtype=torch.float).view(-1, 1).to(self.device)
        is_weights = torch.tensor(is_weights, dtype=torch.float).to(self.device).unsqueeze(1)

        # 计算目标 Q 值
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.target_critic1(next_states, next_actions)
            q2_next = self.target_critic2(next_states, next_actions)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_targets = rewards + self.gamma * (1 - dones) * min_q_next

        # 计算当前 Q 值和 Critic 损失
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        td_error1 = current_q1 - q_targets
        td_error2 = current_q2 - q_targets
        critic1_loss = (F.mse_loss(current_q1, q_targets, reduction='none') * is_weights).mean()
        critic2_loss = (F.mse_loss(current_q2, q_targets, reduction='none') * is_weights).mean()

        # 更新Critic1网络
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.max_grad_norm)
        self.critic1_optimizer.step()

        # 更新Critic2网络
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.max_grad_norm)
        self.critic2_optimizer.step()

        self.critic1_losses.append(critic1_loss.item())
        self.critic2_losses.append(critic2_loss.item())

        # 计算Actor损失
        actions_new, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, actions_new)
        q2_new = self.critic2(states, actions_new)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = ((self.alpha * log_probs) - min_q_new).mean()

        # 更新Actor网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.actor_losses.append(actor_loss.item())

        # 计算α的损失
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        # 更新α
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        self.alpha_losses.append(alpha_loss.item())

        # 软更新目标网络
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)

        # 计算 TD 误差并更新优先级
        td_errors = np.abs(td_error1.detach().cpu().numpy()) + np.abs(td_error2.detach().cpu().numpy())
        td_errors = td_errors.flatten()
        self.replay_buffer.update_priorities(idxs, td_errors)


def plot_positions(env, positions_history, episode, algorithm_name):
    """ 渲染历史状态 """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, env.area_size)
    ax.set_ylim(0, env.area_size)
    # Generate a colormap
    colormap = cm.get_cmap('hsv', env.num_agents+1)
    for step, (agent_positions, agent_directions, goal_positions, obstacle_positions) in enumerate(positions_history):
        # Plot agents
        for idx, (pos, direction) in enumerate(zip(agent_positions, agent_directions)):
            color = colormap(idx)  # Assign color to each agent
            # 绘制观测区域
            # observation_angle = env.observation_angle
            # start_angle = np.degrees(direction - observation_angle / 2)
            # end_angle = np.degrees(direction + observation_angle / 2)
            # wedge = patches.Wedge(pos, env.observation_radius, start_angle, end_angle, color='orange', alpha=0.01, label='Observation')
            # ax.add_patch(wedge)
            # 绘制智能体位置和方向
            agent_circle = Circle(pos, radius=env.agent_radius, edgecolor=color, fill=False, alpha=0.5, linestyle='dashed')  # 使用圆形表示USV边界
            ax.add_patch(agent_circle)
            agent_ellipse = Ellipse(pos, width=env.agent_radius * 2, height=env.agent_radius,
                                    angle=np.degrees(direction), color=color, fill=True, alpha=0.3, linewidth=1.5, label='Agent')
            ax.add_patch(agent_ellipse)
            # 绘制方向箭头
            arrow_dx = np.cos(direction) * env.agent_radius
            arrow_dy = np.sin(direction) * env.agent_radius
            agent_arrow = Arrow(pos[0], pos[1], arrow_dx, arrow_dy, width=0.01 * env.area_size, color=color)
            ax.add_patch(agent_arrow)
            if step == 0:
                ax.text(pos[0], pos[1], str(idx + 1), color=color, fontsize=12, ha='center', va='center')
        if step == 0:
            # Plot goals
            for idx, pos in enumerate(goal_positions):
                color = colormap(idx)  # Assign color to each agent
                goal_circle = Circle(pos, env.goal_radius, color=color, fill=True, alpha=0.3, label='Goal')
                ax.add_patch(goal_circle)
                ax.text(pos[0], pos[1], str(idx + 1), color=color, fontsize=12, ha='center', va='center')
            # Plot obstacles
            for idx, (pos, radius) in enumerate(zip(obstacle_positions, env.obstacle_radii)):
                obstacle_circle = Circle(pos, radius, color='red', fill=True, alpha=0.3, linewidth=1.5, label='Obstacle')
                ax.add_patch(obstacle_circle)
                ax.text(pos[0], pos[1], str(idx + 1), color='red', fontsize=12, ha='center', va='center')

    plt.grid(True)
    plt.legend(handles=[obstacle_circle], loc='best')
    plt.title(f'Path Planning Result - Episode {episode}')
    plt.savefig(os.path.join(algorithm_name, f'path_planning_episode_{episode}.pdf'), format='pdf', bbox_inches='tight')
    plt.show()
    plt.pause(0.01)


def evaluate_policy(env, shared_agent, num_eval_episodes=1, max_steps_per_episode=150, device='cpu'):
    """ 评估策略并记录航行路径 """
    total_agent_collisions = 0
    total_obstacle_collisions = 0
    total_steps = 0
    total_distance_traveled = 0.0
    all_eval_rewards = []
    all_positions_history = []  # 存储所有回合的路径
    for episode in range(num_eval_episodes):
        state = env.reset(initialization=True)
        episode_return = 0
        positions_history = [(np.copy(env.agent_positions), np.copy(env.agent_directions),
                              np.copy(env.goal_positions), np.copy(env.obstacle_positions))]
        for step in range(max_steps_per_episode):
            obs_n_ifs = [state[i][0] for i in range(env.num_agents)]  # 获取所有智能体的局部信息观测
            obs_n_rl = [state[i][1] for i in range(env.num_agents)]
            actions = [shared_agent.take_action(obs, deterministic=True)[0] for obs in obs_n_rl]  # 确定性动作
            env_actions = env.calculate_action(obs_n_ifs, step, policy_weight=actions)  # 使用策略权重计算智能体动作，传入局部信息观测
            next_state, reward, done = env.step(env_actions)

            positions_history.append((np.copy(env.agent_positions), np.copy(env.agent_directions),
                                      np.copy(env.goal_positions), np.copy(env.obstacle_positions)))

            episode_return += sum(reward)
            state = next_state

            if all(done):
                break
        all_eval_rewards.append(episode_return)
        all_positions_history.append(positions_history)
        # plot_positions(env, positions_history, episode)  # Plot the history of positions
        agent_collision, obstacle_collision, total_step = env.get_collision_statistics()
        total_agent_collisions += agent_collision
        total_obstacle_collisions += obstacle_collision
        total_steps += total_step

        # 计算总行驶距离
        distance_traveled = np.sum(
            np.linalg.norm(np.diff(np.array([pos[0] for pos in positions_history]), axis=0), axis=2))
        total_distance_traveled += distance_traveled

    avg_agent_collision = total_agent_collisions / num_eval_episodes
    avg_obstacle_collision = total_obstacle_collisions / num_eval_episodes
    avg_total_step = total_steps / env.num_agents / num_eval_episodes
    avg_distance_traveled = total_distance_traveled / env.num_agents / num_eval_episodes
    print(f'Agents Collision: {avg_agent_collision:.4f}, '
          f'Obstacles Collision: {avg_obstacle_collision:.4f}, '
          f'Decision Steps: {avg_total_step:.4f}, '
          f'Average Distance: {avg_distance_traveled:.4f}')
    return all_eval_rewards, all_positions_history


def train_pssac(env, shared_agent, num_episodes, max_steps_per_episode, device, run,
                 initial_noise_scale=1.0, final_noise_scale=0.001, noise_decay=0.994):
    """ 训练PSSAC agent，使用参数共享策略网络 """
    reward_list = []  # 保存每个回合的return
    noise_scale = initial_noise_scale  # 初始噪声规模
    best_episode_return = float('-inf')  # 保存最高的奖励值
    for episode in range(num_episodes):
        state = env.reset(initialization=True)  # 环境重置，获取状态
        episode_return = 0  # 累计每回合的reward
        for step in range(max_steps_per_episode):
            obs_n_ifs = [state[i][0] for i in range(env.num_agents)]  # 获取所有智能体的局部信息观测
            obs_n_rl = [state[i][1] for i in range(env.num_agents)]  # 获取所有智能体的强化学习观测
            # 获取带有探索噪声的动作
            actions = [shared_agent.take_action(obs)[0] for obs in obs_n_rl]
            # 应用动作到环境
            env_actions = env.calculate_action(obs_n_ifs, step, policy_weight=actions)
            next_state, reward, done = env.step(env_actions)  # 环境更新
            next_obs_n_rl = [next_state[i][1] for i in range(env.num_agents)]  # 获取下一个状态的强化学习观测
            # 将经验存储到经验回放缓冲区
            for i in range(env.num_agents):
                # 获取当前经验的最大优先级
                if len(shared_agent.replay_buffer) > 0:
                    max_priority = shared_agent.replay_buffer.tree.tree[
                                   -shared_agent.replay_buffer.tree.capacity:].max()
                    if max_priority == 0:
                        max_priority = 1.0
                else:
                    max_priority = 1.0
                shared_agent.replay_buffer.add((obs_n_rl[i], actions[i], reward[i], next_obs_n_rl[i], done[i]),
                                               priority=max_priority)
            # 更新 Agent
            shared_agent.update()

            episode_return += sum(reward)  # 累计回合奖励
            # print(f'Episode {episode}, Return: {episode_return:.4f}')
            state = next_state  # 更新状态
            if all(done):
                break
        reward_list.append(episode_return)  # 保存回合的return

        # 衰减噪声规模
        noise_scale = max(final_noise_scale, noise_scale * noise_decay)

        if episode % 10 == 0:
            print(f'Episode {episode}, Return: {episode_return:.4f}')
        # 保存奖励最高的策略网络
        if episode_return > best_episode_return:
            best_episode_return = episode_return
            best_model_path = os.path.join(algorithm_name, f'best_shared_actor_{run + 1}.pth')
            torch.save(shared_agent.actor.state_dict(), best_model_path)
            print(f"当前最优模型已保存至 {best_model_path}，奖励: {best_episode_return:.4f}")
    return reward_list


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    algorithm_name = 'PSSAC_PER'  # 算法名称
    os.makedirs(algorithm_name, exist_ok=True)  # 创建以算法名命名的文件夹，如果不存在
    # 定义超参数
    num_episodes = 1000
    max_steps_per_episode = 150  # 每回合的最大步数
    actor_lr = 1e-4  # Actor 网络的学习率
    critic_lr = 3e-4  # Critic 网络的学习率
    gamma = 0.98  # 折扣因子
    tau = 0.005  # 软更新参数
    hidden_dim = 64  # 隐藏层维度
    buffer_size = 10000
    minimal_size = 1000
    batch_size = 64  # 批量大小
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 定义环境和智能体参数
    num_agents = 10
    area_size = 100
    action_dim = 4  # 动作维度对应策略权重 4
    action_bound = 1.0  # 假设动作范围在 [-1, 1]

    for i in range(1):  # 算法重复执行次数
        start_time = time.time()  # 记录开始时间
        # 环境设置
        agent_positions = np.array(env_utils.generate_circle_points((0.5, 0.5), 0.4, num_agents, 0)) * area_size  # 生成圆周上的点
        agent_directions = np.random.uniform(-np.pi, np.pi, num_agents)  # 初始化在 [-pi, pi] 范围内的随机方向
        goal_positions = np.array(env_utils.generate_circle_points((0.5, 0.5), 0.43, num_agents, np.pi)) * area_size  # 生成圆周上的点
        obstacle_positions = np.array([[0.22, 0.55], [0.32, 0.29], [0.68, 0.34], [0.51, 0.70], [0.45, 0.46]]) * area_size
        obstacle_radii = np.array([0.06, 0.07, 0.08, 0.10, 0.05]) * area_size
        env = Environment(num_agents=num_agents, area_size=area_size,
                          agent_positions=agent_positions, agent_directions=agent_directions, goal_positions=goal_positions,
                          obstacle_positions=obstacle_positions, obstacle_radii=obstacle_radii)
        obs_n = env.reset(initialization=True)
        state_dim = obs_n[0][1].shape[0]  # 状态维度 8

        # 创建共享的 PSSAC agent
        shared_agent = PSSACAgent(state_dim, action_dim, hidden_dim, action_bound, actor_lr, critic_lr, tau, gamma,
                                  buffer_size, minimal_size, batch_size, device)

        # 开始训练
        print(f"Starting training of {algorithm_name} agent iteration {i + 1}...")
        return_list = train_pssac(env, shared_agent, num_episodes, max_steps_per_episode, device, i)

        # Save training results
        np.save(f'{algorithm_name}/reward_list_{i + 1}.npy', return_list)
        np.save(f'{algorithm_name}/actor_losses_{i + 1}.npy', shared_agent.actor_losses)
        np.save(f'{algorithm_name}/critic1_losses_{i + 1}.npy', shared_agent.critic1_losses)
        np.save(f'{algorithm_name}/critic2_losses_{i + 1}.npy', shared_agent.critic2_losses)

        # Save trained shared actor model
        actor_save_path = os.path.join(algorithm_name, f'shared_actor_{i + 1}.pth')
        torch.save(shared_agent.actor.state_dict(), actor_save_path)

        # Save trained critic models
        critic1_save_path = os.path.join(algorithm_name, f'shared_critic1_{i + 1}.pth')
        critic2_save_path = os.path.join(algorithm_name, f'shared_critic2_{i + 1}.pth')
        torch.save(shared_agent.critic1.state_dict(), critic1_save_path)
        torch.save(shared_agent.critic2.state_dict(), critic2_save_path)

        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算运行时间
        print(f"代码运行时间: {elapsed_time:.4f} 秒")

    # 绘制训练结果
    episodes = np.arange(len(return_list))
    plt.plot(episodes, rl_utils.moving_average(return_list, 9))
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.savefig(f'{algorithm_name}/reward_curve.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    # 绘制 Actor 和 Critic 损失曲线
    plt.figure()
    plt.plot(shared_agent.actor_losses, label='Actor Loss')
    plt.xlabel('Update Step')
    plt.ylabel('Actor Loss')
    plt.legend()
    plt.savefig(f'{algorithm_name}/actor_loss_curve.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(shared_agent.critic1_losses, label='Critic1 Loss')
    plt.plot(shared_agent.critic2_losses, label='Critic2 Loss')
    plt.xlabel('Update Step')
    plt.ylabel('Critic Loss')
    plt.legend()
    plt.savefig(f'{algorithm_name}/critic_loss_curve.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    # 保存训练后的共享策略模型
    torch.save(shared_agent.actor.state_dict(), actor_save_path)
