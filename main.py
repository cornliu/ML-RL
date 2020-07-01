# %%capture
from pyvirtualdisplay import Display
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

# %matplotlib inline
import matplotlib.pyplot as plt

from IPython import display

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm.notebook import tqdm
import gym
env = gym.make('LunarLander-v2')
print(env.observation_space)
print(env.action_space)
initial_state = env.reset()
print(initial_state)
random_action = env.action_space.sample()
print(random_action)
observation, reward, done, info = env.step(random_action)
print(done)
print(reward)
env.reset()

img = plt.imshow(env.render(mode='rgb_array'))

done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)

    img.set_data(env.render(mode='rgb_array'))
    display.display(plt.gcf())
    display.clear_output(wait=True)
class PolicyGradientNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, state):
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(self.fc2(hid))
        return F.softmax(self.fc3(hid), dim=-1)
class BaselineNetwork(nn.Module):                                    #REINFORCE with baseline
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(8, 64)                        
    self.fc2 = nn.Linear(64, 32)                         
    self.fc3 = nn.Linear(32, 1)
  def forward(self, state):
    hid = torch.relu(self.fc1(state))
    hid = torch.relu(self.fc2(hid))
    return self.fc3(hid)

class PolicyGradientAgent():

    def __init__(self, network, baselinenet):
        self.network = network
        self.baselinenetwork = baselinenet
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.001)
        self.optimizer_baseline = optim.SGD(self.baselinenetwork.parameters(), lr=0.001)#, amsgrad=True)
    def baselinelearn(self, vals, rewards):
        loss_fun = nn.MSELoss()
        loss = loss_fun(vals,rewards).sum()
        self.optimizer_baseline.zero_grad()
        loss.backward()
        self.optimizer_baseline.step()
    def learn(self, log_probs, rewards):
        loss = (-log_probs * rewards).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def sample(self, state):
        action_prob = self.network(torch.FloatTensor(state))
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob
network = PolicyGradientNetwork()
Baseline_network = BaselineNetwork()
agent = PolicyGradientAgent(network,Baseline_network)
agent.network.train()  # 訓練前，先確保 network 處在 training 模式
agent.baselinenetwork.train()
EPISODE_PER_BATCH = 10  # 每蒐集 5 個 episodes 更新一次 agent
NUM_BATCH = 400        # 總共更新 400 次

avg_total_rewards, avg_final_rewards = [], []

decay_rate = 0.98

prg_bar = tqdm(range(NUM_BATCH))
for batch in prg_bar:

    log_probs, rewards = [], []
    total_rewards, final_rewards = [], []
    record = []
    # 蒐集訓練資料
    for episode in range(EPISODE_PER_BATCH):
        
        decay_reward = []
        state = env.reset()
        total_reward, total_step = 0, 0
        value = []

        while True:
            action, log_prob = agent.sample(state)
            next_state, reward, done, _ = env.step(action)
            record.append(next_state)
            log_probs.append(log_prob)
            state = next_state
            if next_state[1]>0.3 and next_state[3]>-0.1:                                   #最後改進的部分
                reward -=0.3
            if next_state[6]==1 and next_state[7]==1 and action in (1,2,3):
                reward -=0.5
            total_reward += reward
            total_step += 1
            decay_reward.append(reward)
            if done:
                final_rewards.append(reward)
                total_rewards.append(total_reward)
                for i in range(len(decay_reward)-1):                                       #discounted reward
                    for j in range(i+1, len(decay_reward)):
                        decay_reward[i] += decay_reward[j]*decay_rate**(j-i)
                # rewards.append(np.full(total_step, total_reward))  # 設定同一個 episode 每個 action 的 reward 都是 total reward
                rewards.append(np.array(decay_reward))
                break

    # 紀錄訓練過程
    avg_total_reward = sum(total_rewards) / len(total_rewards)
    avg_final_reward = sum(final_rewards) / len(final_rewards)
    avg_total_rewards.append(avg_total_reward)
    avg_final_rewards.append(avg_final_reward)
    prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")

    baselinevalue = agent.baselinenetwork(torch.tensor(record)).squeeze(1)

    rewards = np.concatenate(rewards, axis=0)
    rewards -= baselinevalue.detach().numpy()
    rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # 將 reward 正規標準化
    agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))

    baselinevalue = (baselinevalue - torch.mean(baselinevalue)) / (torch.std(baselinevalue) + 1e-9)
    agent.baselinelearn(baselinevalue.type(torch.double), torch.from_numpy(rewards).type(torch.double))
plt.plot(avg_total_rewards)
plt.title("Total Rewards")
plt.show()
plt.plot(avg_final_rewards)
plt.title("Final Rewards")
plt.show()

agent.network.eval()  # 測試前先將 network 切換為 evaluation 模式


# img = plt.imshow(env.render(mode='rgb_array'))

total_reward = 0
for i in range(500):
    done = False
    state = env.reset()
    while not done:
        action, _ = agent.sample(state)
        state, reward, done, _ = env.step(action)

        total_reward += reward
print(total_reward/500)

    # img.set_data(env.render(mode='rgb_array'))
    # display.display(plt.gcf())
    # display.clear_output(wait=True)