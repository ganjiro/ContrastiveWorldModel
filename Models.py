import torch
from torch import nn
import torch.nn.functional as F


class Contrastive_world_model_end_to_end(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, action_dim, hidden_dim_head):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_ = nn.Linear(hidden_dim, hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, z_dim)
        self.fc4 = nn.Linear(hidden_dim, z_dim)

        self.fc5 = nn.Linear(z_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6_ = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, input_dim)

        self.fc1_contr = nn.Linear(z_dim + action_dim, hidden_dim_head)
        self.fc2_contr = nn.Linear(hidden_dim_head, hidden_dim_head * 2)
        self.fc3_contr = nn.Linear(hidden_dim_head * 2, hidden_dim_head)
        self.fc4_contr = nn.Linear(hidden_dim_head, z_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc2_(h))
        mu, log_var = self.fc3(h), self.fc4(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        out = mu + eps * std
        return out

    def getZ(self, x):
        mu, std = self.encode(x)
        return self.reparameterize(mu, std), mu, std

    def decode(self, z):
        h = F.relu(self.fc5(z))
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc6_(h))
        return self.fc7(h)

    def reconstruct(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var, z

    def transitionZ(self, z, action):
        out = torch.cat([z, action], dim=1)
        out = F.relu(self.fc1_contr(out))
        out = F.relu(self.fc2_contr(out))
        out = F.relu(self.fc3_contr(out))
        out = self.fc4_contr(out)
        return out + z

    def forward(self, x, action):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z = self.transitionZ(z, action)
        return self.decode(z)


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc_extra1 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, z_dim)
        self.fc4 = nn.Linear(hidden_dim, z_dim)

        self.fc5 = nn.Linear(z_dim, hidden_dim)
        self.fc_extra2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc6 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc_extra1(h))
        mu, log_var = self.fc3(h), self.fc4(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        out = mu + eps * std

        return out

    def getZ(self, x):
        mu, std = self.encode(x)
        return self.reparameterize(mu, std)

    def decode(self, z):
        h = F.relu(self.fc5(z))
        h = F.relu(self.fc_extra2(h))
        h = F.relu(self.fc6(h))
        return self.fc7(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


class ContrastiveHead(nn.Module):
    def __init__(self, z_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim + action_dim, int(hidden_dim / 2))
        self.fc2 = nn.Linear(int(hidden_dim / 2), hidden_dim)
        self.fc_extra_1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc_extra_2 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc_extra_3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.fc4 = nn.Linear(int(hidden_dim / 2), z_dim)

    def forward(self, x, action):
        x = torch.cat([x, action], dim=1)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc_extra_1(out))
        out = F.relu(self.fc_extra_2(out))
        out = F.relu(self.fc_extra_3(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out


class Contrastive_world_model_end_to_end_reward(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, action_dim, hidden_dim_head):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_ = nn.Linear(hidden_dim, hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, z_dim)
        self.fc4 = nn.Linear(hidden_dim, z_dim)

        self.fc5 = nn.Linear(z_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6_ = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, input_dim)

        self.fc1_contr = nn.Linear(z_dim + action_dim, hidden_dim_head)
        self.fc2_contr = nn.Linear(hidden_dim_head, hidden_dim_head * 2)
        self.fc3_contr = nn.Linear(hidden_dim_head * 2, hidden_dim_head)
        self.fc4_contr = nn.Linear(hidden_dim_head, z_dim)

        self.fc1_reward = nn.Linear(hidden_dim_head * 2, hidden_dim_head)
        self.fc2_reward = nn.Linear(hidden_dim_head, 1)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc2_(h))
        mu, log_var = self.fc3(h), self.fc4(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        out = mu + eps * std
        return out

    def getZ(self, x):
        mu, std = self.encode(x)
        return self.reparameterize(mu, std), mu, std

    def decode(self, z):
        h = F.relu(self.fc5(z))
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc6_(h))
        return self.fc7(h)

    def reconstruct(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var, z

    def transitionZ(self, z, action):
        out = torch.cat([z, action], dim=1)
        out = F.relu(self.fc1_contr(out))
        out = F.relu(self.fc2_contr(out))

        rew = F.relu(self.fc1_reward(out))
        rew = self.fc2_reward(rew)

        out = F.relu(self.fc3_contr(out))
        out = self.fc4_contr(out)

        return out, rew

    def forward(self, x, action):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z, reward = self.transitionZ(z, action)
        return self.decode(z), reward


class ActionDecoder(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim * 2, int(hidden_dim / 2))
        self.fc2 = nn.Linear(int(hidden_dim / 2), hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.fc4 = nn.Linear(int(hidden_dim / 2), action_dim)

    def forward(self, x, delta_x):
        x = torch.cat([x, delta_x], dim=1)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out
