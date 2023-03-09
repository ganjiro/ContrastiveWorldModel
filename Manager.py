import copy
import os
import random

import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from DatasetBuilder import D4RLDataset  # D4RLDatasetTest
from Models import Contrastive_world_model_end_to_end, VAE, ContrastiveHead
from TD3_BC import TD3_BC, ReplayBuffer, eval_policy

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
os.environ["LD_LIBRARY_PATH"] += ":/usr/lib/nvidia"
import gym

import torch.nn as nn

import d4rl
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from datetime import datetime


class Manager:

    def __init__(self, model_name, env_name, perc=0.3, savepath=None, contrastive=True, writer_name="Manager",
                 test_aug=False, test_aug_dimension=100000):
        self.savepath = savepath
        self.model_name = model_name
        self.env_name = env_name
        self.contrastive = contrastive
        self.test_aug = test_aug

        self.env = gym.make(env_name)
        if self.test_aug:
            self.dataset_contr, self.dataset_rl, self.dataset_test = self.create_dataset_test_aug(
                self.env.get_dataset(), test_aug_dimension)
        else:
            self.dataset_contr, self.dataset_rl, self.dataset_test, self.corrupted_index = self.create_datasets(
                self.env.get_dataset(), perc)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loader = DataLoader(D4RLDataset(self.dataset_contr), batch_size=1024, shuffle=True, num_workers=8)
        self.test_loader = DataLoader(D4RLDataset(self.dataset_test), batch_size=500, shuffle=True, num_workers=8)

        self.writer = SummaryWriter(
            writer_name + '/' + model_name + ('' if contrastive else '_no_contrastive') + '/' + datetime.now().strftime(
                "%m-%d-%Y_%H:%M"))
        self.loss_function_contrastive = nn.CrossEntropyLoss()

        if model_name == "end_to_end":
            self.model = Contrastive_world_model_end_to_end(input_dim=17, hidden_dim=400, z_dim=12, action_dim=6,
                                                            hidden_dim_head=200).to(self.device)

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

            self.train_fn = self._train_end_to_end

    def load(self, load_path):
        if self.model_name == "end_to_end":
            self.model = torch.load(
                os.path.join(load_path, self.model_name + ('' if self.contrastive else '_no_contrastive') + ".pt"))

    def train(self, epochs):

        for epoch in range(epochs):
            loss = self.train_fn(epoch)
            print('Epoch {}/{}, Loss: {:.4f}'.format(epoch + 1, 500, loss))

        if self.savepath:
            if self.model_name == "end_to_end":
                torch.save(self.model, os.path.join(self.savepath, self.model_name + (
                    '' if self.contrastive else '_no_contrastive') + ".pt"))

        return

    def create_dataset_test_aug(self, dataset, n_transitions):
        np.random.seed(42)

        aa = np.random.permutation(int(len(dataset['next_observations']) ))
        index = [i * 1000 + j for i in aa for j in range(1000)]

        dataset_test = copy.deepcopy(dataset)

        index_test = index[-200000:]
        index_wl_rl = index[:n_transitions]

        dataset['actions'] = dataset['actions'][index_wl_rl]
        dataset['infos/action_log_probs'] = dataset['infos/action_log_probs'][index_wl_rl]
        dataset['next_observations'] = dataset['next_observations'][index_wl_rl]
        dataset['observations'] = dataset['observations'][index_wl_rl]
        dataset['rewards'] = dataset['rewards'][index_wl_rl]
        dataset['terminals'] = dataset['terminals'][index_wl_rl]
        dataset['timeouts'] = dataset['timeouts'][index_wl_rl]

        dataset_test['actions'] = np.delete(dataset_test['actions'], index_test, axis=0)
        dataset_test['infos/action_log_probs'] = np.delete(dataset_test['infos/action_log_probs'], index_test, axis=0)
        dataset_test['next_observations'] = np.delete(dataset_test['next_observations'], index_test, axis=0)
        dataset_test['observations'] = np.delete(dataset_test['observations'], index_test, axis=0)
        dataset_test['rewards'] = np.delete(dataset_test['rewards'], index_test, axis=0)
        dataset_test['terminals'] = np.delete(dataset_test['terminals'], index_test, axis=0)
        dataset_test['timeouts'] = np.delete(dataset_test['timeouts'], index_test, axis=0)

        dataset_contr = copy.deepcopy(dataset)
        
        return dataset_contr, dataset, dataset_test

    def create_datasets(self, dataset, perc):  # todo forse queste deep copy sono useless
        np.random.seed(42)

        aa = np.random.permutation(int(len(dataset['next_observations']) / 1000))
        index = [i * 1000 + j for i in aa for j in range(1000)]

        dataset_test = copy.deepcopy(dataset)

        index_test = index[-200000:]
        index_wl_rl = index[:500000]

        dataset['actions'] = dataset['actions'][index_wl_rl]
        dataset['infos/action_log_probs'] = dataset['infos/action_log_probs'][index_wl_rl]
        dataset['next_observations'] = dataset['next_observations'][index_wl_rl]
        dataset['observations'] = dataset['observations'][index_wl_rl]
        dataset['rewards'] = dataset['rewards'][index_wl_rl]
        dataset['terminals'] = dataset['terminals'][index_wl_rl]
        dataset['timeouts'] = dataset['timeouts'][index_wl_rl]

        dataset_test['actions'] = np.delete(dataset_test['actions'], index_test, axis=0)
        dataset_test['infos/action_log_probs'] = np.delete(dataset_test['infos/action_log_probs'], index_test, axis=0)
        dataset_test['next_observations'] = np.delete(dataset_test['next_observations'], index_test, axis=0)
        dataset_test['observations'] = np.delete(dataset_test['observations'], index_test, axis=0)
        dataset_test['rewards'] = np.delete(dataset_test['rewards'], index_test, axis=0)
        dataset_test['terminals'] = np.delete(dataset_test['terminals'], index_test, axis=0)
        dataset_test['timeouts'] = np.delete(dataset_test['timeouts'], index_test, axis=0)

        to_corrupt = np.random.choice(len(dataset['next_observations']),
                                      int(len(dataset['next_observations']) * perc), replace=False)

        dataset_contr = copy.deepcopy(dataset)

        dataset_contr['actions'] = np.delete(dataset_contr['actions'], to_corrupt, axis=0)
        dataset_contr['infos/action_log_probs'] = np.delete(dataset_contr['infos/action_log_probs'], to_corrupt, axis=0)
        dataset_contr['next_observations'] = np.delete(dataset_contr['next_observations'], to_corrupt, axis=0)
        dataset_contr['observations'] = np.delete(dataset_contr['observations'], to_corrupt, axis=0)
        dataset_contr['rewards'] = np.delete(dataset_contr['rewards'], to_corrupt, axis=0)
        dataset_contr['terminals'] = np.delete(dataset_contr['terminals'], to_corrupt, axis=0)
        dataset_contr['timeouts'] = np.delete(dataset_contr['timeouts'], to_corrupt, axis=0)

        return dataset_contr, dataset, dataset_test, to_corrupt

    def close_writer(self):
        self.writer.flush()
        self.writer.close()

    def _train_end_to_end(self, epoch):
        train_loss = 0
        contr_loss = 0
        VAE_loss = 0
        recon_loss = 0

        for batch_idx, (data, act, next) in enumerate(self.loader):
            self.optimizer.zero_grad()
            data = data.to(self.device)
            act = act.to(self.device)
            next = next.to(self.device)

            z_pos, mu_pos, log_var_pos = self.model.getZ(next)
            recon_batch, mu_data, log_var_data, z_data = self.model.reconstruct(data)
            emb_q = self.model.transitionZ(z_data, act)
            x_t1_hat = self.model.decode(emb_q)

            l_pos = torch.sum(emb_q * z_pos, dim=1, keepdim=True)
            l_neg = torch.mm(emb_q, z_data.t())
            logits = torch.cat([l_pos, l_neg], dim=1)
            positive_label = torch.zeros(logits.size(0), dtype=torch.long).to(self.device)

            loss_vae = self.loss_function_vae(recon_batch, data, mu_data, log_var_data)
            loss_recon = F.huber_loss(x_t1_hat, next, reduction='mean')
            loss_contr = self.loss_function_contrastive(logits / 0.20, positive_label)

            if self.contrastive:
                loss = loss_vae + loss_contr + loss_recon
            else:
                loss = loss_vae + loss_recon
            # print(loss_vae.item(), loss_contr.item(), loss_recon.item())

            loss.backward()

            train_loss += loss.item()
            VAE_loss += loss_vae.item()
            contr_loss += loss_contr.item()
            recon_loss += loss_recon.item()

            self.optimizer.step()

        self.writer.add_scalar("Train/Total_loss", train_loss / len(self.loader), epoch)
        self.writer.add_scalar("Train/loss_vae", VAE_loss / len(self.loader), epoch)
        self.writer.add_scalar("Train/loss_next", recon_loss / len(self.loader), epoch)
        self.writer.add_scalar("Train/loss_contr", contr_loss / len(self.loader), epoch)
        if epoch % 50 == 0: self.test_world_model(epoch)
        return train_loss / len(self.loader)

    @staticmethod
    def loss_function_vae(recon_x, x, mu, log_var):

        BCE = F.mse_loss(recon_x, x, reduction='mean')

        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE  # + KLD

    def test_world_model(self, epoch=0):
        dist_rec = 0
        dist_trans = 0
        for batch_idx, (data, act, next) in enumerate(self.test_loader):
            data = data.to(self.device)
            act = act.to(self.device)
            next = next.to(self.device)

            with torch.no_grad():
                if self.model_name == "end_to_end":
                    recon_batch, _, _, _ = self.model.reconstruct(data)

                    trans_batch = self.model(data, act)

            dist_rec += F.mse_loss(recon_batch, data)
            dist_trans += F.mse_loss(trans_batch, next)

        self.writer.add_scalar("Test/MSE_Recon", dist_rec / len(self.test_loader), epoch)
        self.writer.add_scalar("Test/MSE_Trans", dist_trans / len(self.test_loader), epoch)

        print('Recon Dist: {:.4f}'.format(dist_rec / len(self.test_loader)))
        print('Trans Dist: {:.4f}'.format(dist_trans / len(self.test_loader)))

    def test_td3_bc(self, corr_type=0):  # 0 with model; 1 no corr; 2 mean; 3 noise; 4 remove

        max_action = float(self.env.action_space.high[0])

        # self.dataset_rl = self.env.get_dataset()
        # sus = np.random.randint(2000000, size=200000)
        # self.dataset_rl['actions'] = self.dataset_rl['actions'][sus]
        # self.dataset_rl['infos/action_log_probs'] = self.dataset_rl['infos/action_log_probs'][sus]
        # self.dataset_rl['next_observations'] = self.dataset_rl['next_observations'][sus]
        # self.dataset_rl['observations'] = self.dataset_rl['observations'][sus]
        # self.dataset_rl['rewards'] = self.dataset_rl['rewards'][sus]
        # self.dataset_rl['terminals'] = self.dataset_rl['terminals'][sus]
        # self.dataset_rl['timeouts'] = self.dataset_rl['timeouts'][sus]

        replay_buffer = ReplayBuffer(state_dim=17, action_dim=6)  # todo questo codice puzza
        replay_buffer.convert_D4RL(self.dataset_rl)
        mean, std = replay_buffer.normalize_states()

        if self.test_aug and corr_type != 1:  #todo questo puo essere fatto meglio
            self.augment_dataset(mean, std)
        else:
            if corr_type == 0:
                self.corrupt_w_worldmodel(mean, std)
            elif corr_type == 2:
                self.corrupt_w_mean(mean)
            elif corr_type == 3:
                self.corrupt_w_noise()
            elif corr_type == 4:
                self.remove()

        replay_buffer_cor = ReplayBuffer(state_dim=17, action_dim=6)
        replay_buffer_cor.convert_D4RL(self.dataset_rl)
        mean, std = replay_buffer_cor.normalize_states()

        policy = TD3_BC(state_dim=17, action_dim=6, max_action=max_action, device=self.device)

        for t in range(500000):
            policy.train(replay_buffer_cor, batch_size=256)
            # Evaluate episode
            if (t + 1) % 5000 == 0:
                print(f"Time steps: {t + 1}")

                self.writer.add_scalar("D4RL_score",
                                       eval_policy(policy, self.env_name, 42, mean, std), t)

    def augment_dataset(self, mean, std):

        new_obs = torch.empty(0)
        new_act = torch.empty(0)
        new_next = torch.empty(0)
        new_reward = torch.empty(0)
        new_terminal = torch.empty(0)
        new_timeout = torch.empty(0)

        for idx in np.array_split(np.arange(len(self.dataset_rl['observations'])), 100):
            obs = torch.Tensor(self.dataset_rl['observations'][idx])
            obs = (obs - mean) / std
            act = torch.Tensor(self.dataset_rl['actions'][idx])
            reward = torch.Tensor(self.dataset_rl['rewards'][idx])
            terminal = torch.Tensor(self.dataset_rl['terminals'][idx])
            timeout = torch.Tensor(self.dataset_rl['timeouts'][idx])


            new_obs = torch.cat([new_obs, obs, obs, obs])
            new_act = torch.cat([new_act, act, act, act])
            new_reward = torch.cat([new_reward, reward, reward, reward])
            new_terminal = torch.cat([new_terminal, terminal, terminal, terminal])
            new_timeout = torch.cat([new_timeout, timeout, timeout, timeout])

            obs = obs.to(self.device)
            act = act.to(self.device)
            obs = obs.to(torch.float32)
            act = act.to(torch.float32)

            next_1 = self.model(obs, act).cpu().detach()
            next_2 = self.model(obs, act).cpu().detach()
            next_3 = self.model(obs, act).cpu().detach()

            new_next = torch.cat([new_next, next_1, next_2, next_3])

        self.dataset_rl['observations'] = np.concatenate((self.dataset_rl['observations'], new_obs.numpy()), axis=0)
        self.dataset_rl['actions'] = np.concatenate((self.dataset_rl['actions'], new_act.numpy()), axis=0)
        self.dataset_rl['rewards'] = np.concatenate((self.dataset_rl['rewards'], new_reward.numpy()), axis=0)
        self.dataset_rl['terminals'] = np.concatenate((self.dataset_rl['terminals'], new_terminal.numpy()), axis=0)
        self.dataset_rl['timeouts'] = np.concatenate((self.dataset_rl['timeouts'], new_timeout.numpy()), axis=0)
        self.dataset_rl['next_observations'] = np.concatenate((self.dataset_rl['next_observations'], new_next.numpy()),
                                                              axis=0)

    def corrupt_w_mean(self, mean):

        self.dataset_rl['next_observations'][self.corrupted_index] = mean
        self.dataset_rl['observations'][self.corrupted_index[self.corrupted_index % 1000 != 999] + 1] = mean

    def corrupt_w_worldmodel(self, mean, std):

        obs = torch.Tensor(self.dataset_rl['observations'][self.corrupted_index])
        obs = (obs - mean) / std
        obs = obs.to(self.device)
        act = torch.Tensor(self.dataset_rl['actions'][self.corrupted_index]).to(self.device)

        obs = obs.to(torch.float32)
        act = act.to(torch.float32)

        prediction = torch.empty(0).to(self.device)

        for idx in np.array_split(np.arange(len(obs)), 3):
            if self.model_name == "end_to_end":
                prediction_itr = self.model(obs[idx], act[idx])

            prediction = torch.cat([prediction, prediction_itr])

        prediction = prediction.cpu().detach().numpy()
        prediction = (prediction * std) + mean

        self.dataset_rl['next_observations'][self.corrupted_index] = prediction

        self.dataset_rl['observations'][self.corrupted_index[self.corrupted_index % 1000 != 999] + 1] = prediction[
            self.corrupted_index % 1000 != 999]

    def remove(self):

        self.dataset_rl['actions'] = np.delete(self.dataset_rl['actions'], self.corrupted_index, axis=0)
        self.dataset_rl['infos/action_log_probs'] = np.delete(self.dataset_rl['infos/action_log_probs'],
                                                              self.corrupted_index, axis=0)
        self.dataset_rl['next_observations'] = np.delete(self.dataset_rl['next_observations'], self.corrupted_index,
                                                         axis=0)
        self.dataset_rl['observations'] = np.delete(self.dataset_rl['observations'], self.corrupted_index, axis=0)
        self.dataset_rl['rewards'] = np.delete(self.dataset_rl['rewards'], self.corrupted_index, axis=0)
        self.dataset_rl['terminals'] = np.delete(self.dataset_rl['terminals'], self.corrupted_index, axis=0)
        self.dataset_rl['timeouts'] = np.delete(self.dataset_rl['timeouts'], self.corrupted_index, axis=0)

    def corrupt_w_noise(self):
        noise = np.random.normal(0, 0.5, self.dataset_rl['next_observations'][self.corrupted_index].shape)
        noisy = self.dataset_rl['next_observations'][self.corrupted_index] + noise
        self.dataset_rl['next_observations'][self.corrupted_index] = noisy

        self.dataset_rl['observations'][self.corrupted_index[self.corrupted_index % 1000 != 999] + 1] = noisy[
            self.corrupted_index % 1000 != 999]

    def train_nd_test(self, epochs, corr_type):

        self.train(epochs)
        self.test_td3_bc(corr_type)

    def test_render(self):
        tmp_dataset = D4RLDataset(self.dataset)
        var = tmp_dataset.var
        mean = tmp_dataset.mean

        obs, act, _next = next(iter(self.test_loader))
        obs = obs.to(self.device)
        act = act.to(self.device)
        _next = _next.to(self.device)

        if self.model_name == "end_to_end":
            prediction = self.model(obs, act)

        mse = (prediction - _next).pow(2).mean(1)

        worst = torch.argmax(mse, dim=0)

        env = self.env.unwrapped
        f, axarr = plt.subplots(2, 2)
        env.reset()

        env.state = ((_next[worst].cpu() * torch.sqrt(var)) + mean).numpy()
        img = env.render(mode='rgb_array')
        axarr[0, 0].set_title("Original")
        axarr[0, 0].imshow(img)
        axarr[0, 0].set_ylabel('Worst')
        axarr[0, 0].xaxis.set_ticklabels([])
        axarr[0, 0].yaxis.set_ticklabels([])

        env.reset()
        env.state = ((prediction[worst].cpu() * torch.sqrt(var)) + mean).detach().numpy()
        img = env.render(mode='rgb_array')
        axarr[0, 1].set_title("Prediction")
        axarr[0, 1].imshow(img)
        axarr[0, 1].xaxis.set_ticklabels([])
        axarr[0, 1].yaxis.set_ticklabels([])

        best = torch.argmin(mse, dim=0)

        env.reset()
        env.state = ((_next[best].cpu() * torch.sqrt(var)) + mean).numpy()
        img = env.render(mode='rgb_array')
        axarr[1, 0].imshow(img)
        axarr[1, 0].set_ylabel('Best')
        axarr[1, 0].xaxis.set_ticklabels([])
        axarr[1, 0].yaxis.set_ticklabels([])

        env.reset()
        env.state = ((prediction[best].cpu() * torch.sqrt(var)) + mean).detach().numpy()
        img = env.render(mode='rgb_array')
        axarr[1, 1].imshow(img)
        axarr[1, 1].xaxis.set_ticklabels([])
        axarr[1, 1].yaxis.set_ticklabels([])

        plt.show()
