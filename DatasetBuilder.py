import numpy as np
import torch
import torch.utils.data as data


class D4RLDatasetTrain(data.Dataset):

    def __init__(self, data):
        np.random.seed(seed=42)
        self.data = data

        self.obs = torch.from_numpy(self.data['observations']).float().float()
        self.next = torch.from_numpy(self.data['next_observations']).float()
        self.acts = torch.from_numpy(self.data['actions']).float()

        self.mean = self.obs.mean(0, keepdim=True)
        self.var = self.obs.var(0, keepdim=True)

        self.obs -= self.mean
        self.obs /= torch.sqrt(self.var + 1e-9)

        self.next -= self.mean
        self.next /= torch.sqrt(self.var + 1e-9)

        new_index = np.random.permutation(len(self.obs))

        self.obs = self.obs[new_index]
        self.next = self.next[new_index]
        self.acts = self.acts[new_index]

        self.obs = self.obs[20000:500000]
        self.next = self.next[20000:500000]
        self.acts = self.acts[20000:500000]

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return self.obs[idx], self.acts[idx], self.next[idx]


class D4RLDatasetTest(data.Dataset):
    def __init__(self, data):
        np.random.seed(seed=42)
        self.data = data

        self.obs = torch.from_numpy(self.data['observations']).float().float()
        self.next = torch.from_numpy(self.data['next_observations']).float()
        self.acts = torch.from_numpy(self.data['actions']).float()

        self.mean = self.obs.mean(0, keepdim=True)
        self.var = self.obs.var(0, keepdim=True)

        self.obs -= self.mean
        self.obs /= torch.sqrt(self.var + 1e-9)

        self.next -= self.mean
        self.next /= torch.sqrt(self.var + 1e-9)

        new_index = np.random.permutation(len(self.obs))

        self.obs = self.obs[new_index]
        self.next = self.next[new_index]
        self.acts = self.acts[new_index]

        self.obs = self.obs[:20000]
        self.next = self.next[:20000]
        self.acts = self.acts[:20000]

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return self.obs[idx], self.acts[idx], self.next[idx]
