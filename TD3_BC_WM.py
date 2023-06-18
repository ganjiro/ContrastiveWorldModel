import io
import random
from collections import namedtuple, deque

import PIL.Image
import numpy as np
import torch
import copy

from matplotlib import pyplot as plt

from scipy.special.cython_special import hyp0f1
from torch import nn
import torch.nn.functional as F
import gym
from torch.cuda import device
from torchvision.transforms import ToTensor


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3_BC_WM(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            world_model,
            aug_type=0,
            # 0 no aug, 1 perc batch size, 2 over estimation bias, 3 noise, 4 S4rl,  5 batch REW # 6 S4RL Rew
            # 7 Clipping # 8 mia idea # 9 azione # 10 Creo L'azione #11 batch + state #12 batch con s+eps
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            alpha=2.5,
            device='cpu',
            writer=None,
            action_model=None,
            eps=1,  # clipping eps
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.device = device
        self.writer = writer
        self.eps = eps
        self.total_it = 0

        self.aug_type = aug_type
        self.world_model = world_model
        self.action_model = action_model

        if self.aug_type == 8:
            self.replay_memory = ReplayMemory(20000)

        self.prob = 0.0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256, hyperparameter=0.5, writer=None):
        self.total_it += 1

        self.prob += 1 / (500000 * 5)

        if not self.total_it % 10000 and writer and self.world_model:
            self.test_distr(replay_buffer, writer)

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        if self.aug_type == 1 and hyperparameter > 0.001:
            to_aug = np.random.choice(len(next_state),
                                      int(len(next_state) * hyperparameter), replace=False)

            with torch.no_grad():
                next_state_aug = self.world_model(state, action)

            next_state[to_aug] = next_state_aug[to_aug]

        elif self.aug_type == 3:
            to_aug = np.random.choice(len(next_state),
                                      int(len(next_state) * hyperparameter), replace=False)

            next_state[to_aug] = next_state[to_aug] + torch.randn_like(next_state[to_aug]) / 10

        elif self.aug_type == 5:
            to_aug = np.random.choice(len(next_state),
                                      int(len(next_state) * hyperparameter), replace=False)

            with torch.no_grad():
                next_state_aug, reward_aug = self.world_model(state, action)

            next_state[to_aug] = next_state_aug[to_aug]
            reward[to_aug] = reward_aug[to_aug]

        elif self.aug_type == 8:

            states_toaug = replay_buffer.sample_state(64)

            with torch.no_grad():
                action_toaug = self.actor(states_toaug)

                next_state_aug, reward_aug = self.world_model(states_toaug, action_toaug)

            not_done_aug = torch.full([64, 1], 1, device=self.device)

            self.replay_memory.push(states_toaug, action_toaug, next_state_aug, reward_aug, not_done_aug)

            if random.random() < self.prob:
                batch = self.replay_memory.sample(32)
                for transition in batch:
                    state = torch.cat([state, transition.state])
                    action = torch.cat([action, transition.action])
                    next_state = torch.cat([next_state, transition.next_state])
                    reward = torch.cat([reward, transition.reward])
                    not_done = torch.cat([not_done, transition.done])

        elif self.aug_type == 9:
            to_aug = np.random.choice(len(next_state),
                                      int(len(next_state) * hyperparameter), replace=False)

            noise = (torch.randn_like(action[to_aug]) * 0.05
                     ).clamp(-self.noise_clip, self.noise_clip)

            action_aug = (action[to_aug] + noise
                          ).clamp(-self.max_action, self.max_action)

            with torch.no_grad():
                next_state_aug = self.world_model(state[to_aug], action_aug)

            next_state[to_aug] = next_state_aug
            action[to_aug] = action_aug

        if self.aug_type == 10:
            to_aug = np.random.choice(len(next_state),
                                      int(len(next_state) * hyperparameter), replace=False)

            with torch.no_grad():
                next_state_aug = self.world_model(state, action)
                action_aug = self.action_model(state, next_state_aug - state)

            next_state[to_aug] = next_state_aug[to_aug]
            action[to_aug] = action_aug

        if self.aug_type == 11 and hyperparameter > 0.001:
            to_aug = np.random.choice(len(next_state),
                                      int(len(next_state) * hyperparameter), replace=False)

            with torch.no_grad():
                next_state_aug, state_aug = self.world_model.forward_and_out(state, action)

            next_state[to_aug] = next_state_aug[to_aug]
            state[to_aug] = state_aug[to_aug]

        if self.aug_type == 12 and hyperparameter > 0.001:
            to_aug = np.random.choice(len(next_state),
                                      int(len(next_state) * hyperparameter), replace=False)
            state_noisy = state + torch.randn_like(state) / 10
            with torch.no_grad():
                next_state_aug = self.world_model(state_noisy, action)

            next_state[to_aug] = next_state_aug[to_aug]

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            if self.aug_type == 2:
                next_state_aug = self.world_model(state, action)

                next_action_aug = (
                        self.actor_target(next_state_aug) + noise
                ).clamp(-self.max_action, self.max_action)

                target_Q1_aug, target_Q2_aug = self.critic_target(next_state_aug, next_action)

                target_Q = torch.min(target_Q1, target_Q2)
                target_Q_aug = torch.min(target_Q1_aug, target_Q2_aug)
                target_Q = torch.min(target_Q, target_Q_aug)


            elif self.aug_type == 4:

                next_state_aug_0 = self.world_model(state, action)
                next_state_aug_1 = self.world_model(state, action)

                target_Q1_aug_0, target_Q2_aug_0 = self.critic_target(next_state_aug_0, next_action)
                target_Q1_aug_1, target_Q2_aug_1 = self.critic_target(next_state_aug_1, next_action)
                #
                # target_Q1 = target_Q1 * hyperparameter
                # target_Q2 = target_Q2 * hyperparameter
                #
                # target_Q1_aug_0 = target_Q1_aug_0 * (1 - (hyperparameter / 2))
                # target_Q2_aug_0 = target_Q2_aug_0 * (1 - (hyperparameter / 2))
                #
                # target_Q1_aug_1 = target_Q1_aug_1 * (1 - (hyperparameter / 2))
                # target_Q2_aug_1 = target_Q2_aug_1 * (1 - (hyperparameter / 2))

                target_Q1 = torch.mean(torch.cat([target_Q1, target_Q1_aug_0, target_Q1_aug_1]))
                target_Q2 = torch.mean(torch.cat([target_Q2, target_Q2_aug_0, target_Q2_aug_1]))

                target_Q = torch.min(target_Q1, target_Q2)

            elif self.aug_type == 7:

                next_state_aug_0 = self.world_model(state, action)

                target_Q1_aug_0, target_Q2_aug_0 = self.critic_target(next_state_aug_0, next_action)

                target_Q1 = torch.mean(torch.cat([target_Q1, target_Q1_aug_0])).clamp(min=target_Q1 - self.eps,
                                                                                      max=target_Q1 + self.eps)
                target_Q2 = torch.mean(torch.cat([target_Q2, target_Q2_aug_0])).clamp(min=target_Q2 - self.eps,
                                                                                      max=target_Q2 + self.eps)

                target_Q = torch.min(target_Q1, target_Q2)


            elif self.aug_type != 6:

                target_Q = torch.min(target_Q1, target_Q2)

            if self.aug_type == 6:
                next_state_aug_0, reward_aug = self.world_model(state, action)

                target_Q1_aug_0, target_Q2_aug_0 = self.critic_target(next_state_aug_0, next_action)
                target_Q1 = torch.mean(torch.cat([target_Q1, target_Q1_aug_0]))
                target_Q2 = torch.mean(torch.cat([target_Q2, target_Q2_aug_0]))

                reward = (reward + reward_aug) / 2
                target_Q = torch.min(target_Q1, target_Q2)

                target_Q = reward + not_done * self.discount * target_Q
            else:

                target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            lmbda = self.alpha / Q.abs().mean().detach()

            actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if self.writer: self.writer.add_scalar("TD3BC/actor_loss", actor_loss, self.total_it)
        if self.writer: self.writer.add_scalar("TD3BC/critic_loss", critic_loss, self.total_it)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

    def test_distr(self, replay_buffer, writer, batch_size=1024):

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            if self.aug_type == 5 or self.aug_type == 6 or self.aug_type == 8:
                next_state_aug, _ = self.world_model(state, action)
            else:
                next_state_aug = self.world_model(state, action)

            aug_target_Q1, aug_target_Q2 = self.critic_target(next_state_aug, next_action)

            diff_1 = target_Q1 - aug_target_Q1
            diff_2 = target_Q2 - aug_target_Q2

        diff_distr = torch.cat([diff_1, diff_2]).detach().cpu()

        diff_distr = torch.where(diff_distr > 200, 200, diff_distr)
        diff_distr = torch.where(diff_distr < -200, -200, diff_distr)
        # plt.hist(mse_distr, bins=100)
        # plt.ylim([0, 6000])

        # buf = io.BytesIO()
        # plt.savefig(buf, format='jpeg')
        # buf.seek(0)
        # image = PIL.Image.open(buf)
        #
        # image = ToTensor()(image).unsqueeze(0)

        mse_states = F.mse_loss(next_state_aug, next_state, reduction='none').detach().cpu().sum(1)
        mse_states = torch.cat([mse_states, mse_states])

        plt.scatter(mse_states, diff_distr)
        plt.ylabel("Q(S') - Q(S'WM)")
        plt.xlabel("MSE(S', S'WM)")
        # plt.savefig(writer, "distr_" + str(self.total_it) + ".png")

        writer.add_figure("Plot Distr", plt.gcf(), self.total_it)

        writer.add_histogram('Distr Q-Value', diff_distr, self.total_it, max_bins=1000)


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """ Save a transition """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ReplayBuffer(object):

    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def sample_state(self, size):
        ind = np.random.randint(0, self.size, size=size)
        return torch.FloatTensor(self.state[ind]).to(self.device)

    def convert_D4RL(self, dataset):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1, 1)
        self.not_done = 1. - dataset['terminals'].reshape(-1, 1)
        self.size = self.state.shape[0]

    def normalize_states(self, eps=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std


def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
    print("---------------------------------------")
    return d4rl_score
