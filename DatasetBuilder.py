import copy

import numpy as np
import torch
import torch.utils.data as data


class D4RLDataset(data.Dataset):

    def __init__(self, data, std_reward=False):
        np.random.seed(seed=42)

        self.data = copy.deepcopy(data)

        self.obs = torch.from_numpy(self.data['observations']).float().float()
        self.next = torch.from_numpy(self.data['next_observations']).float()
        self.acts = torch.from_numpy(self.data['actions']).float()
        self.reward = torch.from_numpy(self.data['rewards']).float()

        self.mean = self.obs.mean(0, keepdim=True)
        self.std = self.obs.std(0, keepdim=True) + 1e-5

        self.obs -= self.mean
        self.obs /= self.std

        self.next -= self.mean
        self.next /= self.std

        if std_reward:
            self.mean_reward = self.reward.mean(0, keepdim=True)
            self.std_reward = self.reward.std(0, keepdim=True) + 1e-5

            self.reward -= self.mean_reward
            self.reward /= self.std_reward

        new_index = np.random.permutation(len(self.obs))

        self.obs = self.obs[new_index]
        self.next = self.next[new_index]
        self.acts = self.acts[new_index]
        self.reward = self.reward[new_index]

        # self.obs[:, 2] = 0.
        # self.obs[:, 6] = 0.
        # self.obs[:, 7] = 0.
        # self.obs[:, 10] = 0.
        # self.obs[:, 16] = 0.
        # self.obs[:, 9] = 0.
        # self.obs[:, 14] = 0.
        #
        # self.next[:, 2] = 0.
        # self.next[:, 6] = 0.
        # self.next[:, 7] = 0.
        # self.next[:, 10] = 0.
        # self.next[:, 16] = 0.
        # self.next[:, 9] = 0.
        # self.next[:, 14] = 0.

        self.obs = self.obs.to(torch.float32)
        self.next = self.next.to(torch.float32)
        self.acts = self.acts.to(torch.float32)
        self.reward = self.reward.to(torch.float32)

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return self.obs[idx], self.acts[idx], self.next[idx], self.reward[idx]
