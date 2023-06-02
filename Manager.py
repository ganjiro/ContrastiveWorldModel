import copy
import os

import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from DatasetBuilder import D4RLDataset  # D4RLDatasetTest
from Models import Contrastive_world_model_end_to_end, VAE, \
    ContrastiveHead, Contrastive_world_model_end_to_end_reward, ActionDecoder

from TD3_BC_WM import TD3_BC_WM, ReplayBuffer, eval_policy

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
os.environ["LD_LIBRARY_PATH"] = ":/home/ganjiro/.mujoco/mujoco210/bin:/usr/lib/nvidia"

import d4rl

import gym

import torch.nn as nn

import numpy as np

from torch.utils.tensorboard import SummaryWriter

from datetime import datetime


class Manager:

    def __init__(self, model_name, env_name, perc=0.3, savepath=None, contrastive=True, writer_name="Manager",
                 test_aug=False, dimension=None, entire_trajectory=True, test_name="", std_reward=False,
                 all_dataset=False,
                 action=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.savepath = savepath
        self.model_name = model_name
        self.env_name = env_name
        self.contrastive = contrastive
        self.test_aug = test_aug
        self.action = action
        self.entire_trajectory = entire_trajectory
        self.model_action = None

        self.env = gym.make(env_name)

        if not os.path.exists(savepath):
            os.mkdir(savepath)

        if all_dataset:
            self.dataset_contr, self.dataset_rl, self.dataset_test, dimension = self.create_complete_dataset(
            )
        elif self.entire_trajectory and not dimension:
            self.dataset_contr, self.dataset_rl, self.dataset_test, dimension = self.create_full_dataset(
                self.env.get_dataset())
        elif self.test_aug:
            self.dataset_contr, self.dataset_rl, self.dataset_test = self.create_dataset_test_aug(
                self.env.get_dataset(), dimension)
        else:
            self.dataset_contr, self.dataset_rl, self.dataset_test, self.corrupted_index = self.create_datasets(
                self.env.get_dataset(), dimension, perc)

        self.loader = DataLoader(D4RLDataset(self.dataset_contr, std_reward), batch_size=4096, shuffle=True,
                                 num_workers=8)
        self.test_loader = DataLoader(D4RLDataset(self.dataset_test, std_reward), batch_size=500, shuffle=True,
                                      num_workers=8)

        self.writer_name = test_name + '_' + self.model_name + \
                           ('_testaug' if self.test_aug else '') + (
                               '_traj' if self.entire_trajectory else '_random') + '_' + str(dimension) + (
                               '' if contrastive else '_no_contrastive') + '_' + env_name + (
                               '' if perc == 0 else str(perc))

        self.writer = SummaryWriter(
            writer_name + '/' + self.writer_name + '/' + datetime.now().strftime(
                "%m-%d-%Y_%H:%M"))
        self.loss_function_contrastive = nn.CrossEntropyLoss()

        if model_name == "end_to_end":
            self.model = Contrastive_world_model_end_to_end(input_dim=self.env.observation_space.shape[0],
                                                            hidden_dim=500,
                                                            z_dim=12, action_dim=self.env.action_space.shape[0],
                                                            hidden_dim_head=300).to(self.device)

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)

            self.train_fn = self._train_end_to_end_new

            if self.contrastive:
                self.model_vae = VAE(input_dim=self.env.observation_space.shape[0], hidden_dim=500, z_dim=12).to(
                    self.device)
                self.optimizer_vae = torch.optim.Adam(self.model_vae.parameters(), lr=1e-4)
                self.loaded_vae = False

            if self.action:
                self.model_action = ActionDecoder(state_dim=self.env.observation_space.shape[0], hidden_dim=500,
                                                  action_dim=self.env.action_space.shape[0]).to(self.device)
                self.optimizer_action = torch.optim.Adam(self.model_action.parameters(), lr=1e-5)
                self.loaded_act = False


        elif model_name == "reward":
            self.model = Contrastive_world_model_end_to_end_reward(input_dim=self.env.observation_space.shape[0],
                                                                   hidden_dim=500,
                                                                   z_dim=12, action_dim=self.env.action_space.shape[0],
                                                                   hidden_dim_head=300).to(self.device)

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)

            self.train_fn = self._train_end_to_end_reward

            if self.contrastive:
                self.model_vae = VAE(input_dim=self.env.observation_space.shape[0], hidden_dim=500, z_dim=12).to(
                    self.device)
                self.optimizer_vae = torch.optim.Adam(self.model_vae.parameters(), lr=1e-4)
                self.loaded_vae = False

        # elif model_name == "splitted":
        #
        #     self.model_vae = VAE(input_dim=self.env.observation_space.shape[0], hidden_dim=500, z_dim=12).to(
        #         self.device)
        #     self.optimizer_vae = torch.optim.Adam(self.model_vae.parameters(), lr=1e-3)
        #     self.loaded_vae = False
        #
        #     self.model_contr = ContrastiveHead(z_dim=12, action_dim=self.env.action_space.shape[0], hidden_dim=300).to(
        #         self.device)
        #     self.optimizer_contr = torch.optim.Adam(self.model_contr.parameters(), lr=1e-4)
        #
        #     self.train_fn = self._train_splitted
        else:
            self.model = None

    def load(self, load_path):
        if self.model_name == "end_to_end":
            self.model = torch.load(
                os.path.join(load_path, self.model_name + ('' if self.contrastive else '_no_contrastive') + ".pt"),
                map_location='cuda:0')
            if self.contrastive:
                self.loaded_vae = True
                self.model_vae = torch.load(os.path.join(load_path, "VAE.pt"), map_location='cuda:0').to(self.device)

                self.model_vae.eval()
            if self.action:
                self.loaded_act = True
                self.model_action = torch.load(os.path.join(load_path, "action.pt"), map_location='cuda:0').to(
                    self.device)
                self.model_action.eval()

        elif self.model_name == "splitted":
            self.loaded_vae = True
            self.model_vae = torch.load(os.path.join(load_path, "VAE.pt"))
            self.model_vae.eval()
            self.model_contr = torch.load(
                os.path.join(load_path, self.model_name + ('' if self.contrastive else '_no_contrastive') + ".pt"))

    def train(self, epochs):
        if (self.model_name == "splitted" and not self.loaded_vae) or (self.contrastive and not self.loaded_vae):
            self.model_vae.train()
            self._train_vae(200)
            self.model_vae.eval()

        if self.action:
            self.model_action.train()
            self._train_action(200)
            self.model_action.eval()

        for epoch in range(epochs):
            loss = self.train_fn(epoch)

            print('Epoch {}/{}, Loss: {:.4f}'.format(epoch + 1, 500, loss))

        if self.savepath:
            if self.model_name == "end_to_end":
                torch.save(self.model, os.path.join(self.savepath, self.model_name + (
                    '' if self.contrastive else '_no_contrastive') + ".pt"))
            elif self.model_name == "splitted":
                torch.save(self.model_contr, os.path.join(self.savepath, self.model_name + (
                    '' if self.contrastive else '_no_contrastive') + ".pt"))

        return

    def create_dataset_test_aug(self, dataset, n_transitions):
        np.random.seed(42)

        if self.entire_trajectory:
            index = [i * 1000 + j for i in np.random.permutation(int(len(dataset['next_observations']) / 1000)) for j in
                     range(1000)]
        else:
            index = np.random.permutation(int(len(dataset['next_observations'])))

        index_test = index[-20000:]
        index_wl_rl = index[:n_transitions]

        dataset_wl_rl = {}

        dataset_wl_rl['actions'] = dataset['actions'][index_wl_rl]
        dataset_wl_rl['infos/action_log_probs'] = dataset['infos/action_log_probs'][index_wl_rl]
        dataset_wl_rl['next_observations'] = dataset['next_observations'][index_wl_rl]
        dataset_wl_rl['observations'] = dataset['observations'][index_wl_rl]
        dataset_wl_rl['rewards'] = dataset['rewards'][index_wl_rl]
        dataset_wl_rl['terminals'] = dataset['terminals'][index_wl_rl]
        dataset_wl_rl['timeouts'] = dataset['timeouts'][index_wl_rl]

        dataset_test = {}

        dataset_test['actions'] = dataset['actions'][index_test]
        dataset_test['infos/action_log_probs'] = dataset['infos/action_log_probs'][index_test]
        dataset_test['next_observations'] = dataset['next_observations'][index_test]
        dataset_test['observations'] = dataset['observations'][index_test]
        dataset_test['rewards'] = dataset['rewards'][index_test]
        dataset_test['terminals'] = dataset['terminals'][index_test]
        dataset_test['timeouts'] = dataset['timeouts'][index_test]

        return dataset_wl_rl, dataset, dataset_test

    def create_datasets(self, dataset, dimension, perc):  # todo forse queste deep copy sono useless

        try:
            dataset['next_observations']
        except:
            dataset['next_observations'] = np.append(dataset['observations'][1:],
                                                     np.expand_dims(dataset['observations'][1], axis=0), axis=0)

        # np.random.seed(42)

        if self.entire_trajectory:
            index = [i * 1000 + j for i in np.random.permutation(int(len(dataset['next_observations']) / 1000)) for j in
                     range(1000)]
        else:
            index = np.random.permutation(int(len(dataset['next_observations'])))

        # sus = copy.deepcopy(dataset)
        # noise = np.random.normal(0, 0.01, dataset['next_observations'].shape)
        # noisy = dataset['next_observations'] + noise
        # dataset['next_observations'] = noisy
        # tmp = np.arange(len(dataset['next_observations']))
        # dataset['observations'][tmp[tmp % 1000 != 999] + 1] = noisy[tmp % 1000 != 999]

        dataset_test = copy.deepcopy(dataset)

        index_test = index[-20000:]
        index_wl_rl = index[:dimension]

        dataset['actions'] = dataset['actions'][index_wl_rl]
        # dataset['infos/action_log_probs'] = dataset['infos/action_log_probs'][index_wl_rl]
        dataset['next_observations'] = dataset['next_observations'][index_wl_rl]
        dataset['observations'] = dataset['observations'][index_wl_rl]
        dataset['rewards'] = dataset['rewards'][index_wl_rl]
        dataset['terminals'] = dataset['terminals'][index_wl_rl]
        dataset['timeouts'] = dataset['timeouts'][index_wl_rl]

        dataset_test['actions'] = dataset_test['actions'][index_test]
        # dataset_test['infos/action_log_probs'] = dataset_test['infos/action_log_probs'][index_test]
        dataset_test['next_observations'] = dataset_test['next_observations'][index_test]
        dataset_test['observations'] = dataset_test['observations'][index_test]
        dataset_test['rewards'] = dataset_test['rewards'][index_test]
        dataset_test['terminals'] = dataset_test['terminals'][index_test]
        dataset_test['timeouts'] = dataset_test['timeouts'][index_test]

        to_corrupt = np.random.choice(len(dataset['next_observations']),
                                      int(len(dataset['next_observations']) * perc), replace=False)

        dataset_contr = copy.deepcopy(dataset)

        dataset_contr['actions'] = np.delete(dataset_contr['actions'], to_corrupt, axis=0)
        # dataset_contr['infos/action_log_probs'] = np.delete(dataset_contr['infos/action_log_probs'], to_corrupt, axis=0)
        dataset_contr['next_observations'] = np.delete(dataset_contr['next_observations'], to_corrupt, axis=0)
        dataset_contr['observations'] = np.delete(dataset_contr['observations'], to_corrupt, axis=0)
        dataset_contr['rewards'] = np.delete(dataset_contr['rewards'], to_corrupt, axis=0)
        dataset_contr['terminals'] = np.delete(dataset_contr['terminals'], to_corrupt, axis=0)
        dataset_contr['timeouts'] = np.delete(dataset_contr['timeouts'], to_corrupt, axis=0)

        index = np.concatenate(
            (np.where(dataset_contr['timeouts'] == True), np.where(dataset_contr['terminals'] == True)), axis=1)

        dataset_contr['actions'] = np.delete(dataset_contr['actions'], index, axis=0)
        dataset_contr['observations'] = np.delete(dataset_contr['observations'], index, axis=0)
        dataset_contr['next_observations'] = np.delete(dataset_contr['next_observations'], index, axis=0)
        dataset_contr['rewards'] = np.delete(dataset_contr['rewards'], index, axis=0)
        dataset_contr['terminals'] = np.delete(dataset_contr['terminals'], index, axis=0)
        dataset_contr['timeouts'] = np.delete(dataset_contr['timeouts'], index, axis=0)

        return dataset_contr, dataset, dataset_test, to_corrupt

    def create_new_scenario_dataset(self, dataset, dimension, entire_trajectory=False):

        try:
            dataset['next_observations']
        except:
            dataset['next_observations'] = np.append(dataset['observations'][1:],
                                                     np.expand_dims(dataset['observations'][1], axis=0), axis=0)

        np.random.seed(42)

        if entire_trajectory:
            index = [i * 1000 + j for i in np.random.permutation(int(len(dataset['next_observations']) / 1000)) for j in
                     range(1000)]
        else:
            index = np.random.permutation(int(len(dataset['next_observations'])))

        index_wl_rl = index[:dimension]

        dataset['actions'] = dataset['actions'][index_wl_rl]
        dataset['infos/action_log_probs'] = dataset['infos/action_log_probs'][index_wl_rl]
        dataset['next_observations'] = dataset['next_observations'][index_wl_rl]
        dataset['observations'] = dataset['observations'][index_wl_rl]
        dataset['rewards'] = dataset['rewards'][index_wl_rl]
        dataset['terminals'] = dataset['terminals'][index_wl_rl]
        dataset['timeouts'] = dataset['timeouts'][index_wl_rl]

        return dataset

    def create_full_dataset(self, dataset):

        try:
            dataset['next_observations']
        except:
            dataset['next_observations'] = np.append(dataset['observations'][1:],
                                                     np.expand_dims(dataset['observations'][1], axis=0), axis=0)

        dataset_test = copy.deepcopy(dataset)
        dataset_contr = copy.deepcopy(dataset)

        dataset_test['actions'] = dataset_test['actions'][-20000:]
        dataset_test['observations'] = dataset_test['observations'][-20000:]
        dataset_test['next_observations'] = dataset_test['next_observations'][-20000:]
        dataset_test['rewards'] = dataset_test['rewards'][-20000:]
        dataset_test['terminals'] = dataset_test['terminals'][-20000:]
        dataset_test['timeouts'] = dataset_test['timeouts'][-20000:]

        index = np.concatenate(
            (np.where(dataset_contr['timeouts'] == True), np.where(dataset_contr['terminals'] == True)), axis=1)

        dataset_contr['actions'] = np.delete(dataset_contr['actions'], index, axis=0)
        dataset_contr['observations'] = np.delete(dataset_contr['observations'], index, axis=0)
        dataset_contr['next_observations'] = np.delete(dataset_contr['next_observations'], index, axis=0)
        dataset_contr['rewards'] = np.delete(dataset_contr['rewards'], index, axis=0)
        dataset_contr['terminals'] = np.delete(dataset_contr['terminals'], index, axis=0)
        dataset_contr['timeouts'] = np.delete(dataset_contr['timeouts'], index, axis=0)

        return dataset_contr, dataset, dataset_test, len(dataset['actions'])

    def create_complete_dataset(self):
        env = gym.make("halfcheetah-medium-replay-v2")
        dataset_mr = env.get_dataset()
        env = gym.make("halfcheetah-medium-expert-v2")
        dataset_me = env.get_dataset()
        env = gym.make("halfcheetah-random-v2")
        dataset_r = env.get_dataset()

        dataset = {}

        dataset['actions'] = np.concatenate((dataset_mr['actions'], dataset_me['actions'], dataset_r['actions']))
        dataset['observations'] = np.concatenate(
            (dataset_mr['observations'], dataset_me['observations'], dataset_r['observations']))
        dataset['next_observations'] = np.concatenate(
            (dataset_mr['next_observations'], dataset_me['next_observations'], dataset_r['next_observations']))
        dataset['rewards'] = np.concatenate((dataset_mr['rewards'], dataset_me['rewards'], dataset_r['rewards']))
        dataset['terminals'] = np.concatenate(
            (dataset_mr['terminals'], dataset_me['terminals'], dataset_r['terminals']))
        dataset['timeouts'] = np.concatenate((dataset_mr['timeouts'], dataset_me['timeouts'], dataset_r['timeouts']))

        dataset_test = copy.deepcopy(dataset)
        dataset_contr = copy.deepcopy(dataset)

        dataset_test['actions'] = dataset_test['actions'][-20000:]
        dataset_test['observations'] = dataset_test['observations'][-20000:]
        dataset_test['next_observations'] = dataset_test['next_observations'][-20000:]
        dataset_test['rewards'] = dataset_test['rewards'][-20000:]
        dataset_test['terminals'] = dataset_test['terminals'][-20000:]
        dataset_test['timeouts'] = dataset_test['timeouts'][-20000:]

        return dataset_contr, dataset, dataset_test, len(dataset['actions'])

    def close_writer(self):
        self.writer.flush()
        self.writer.close()

    def _train_end_to_end_new(self, epoch):
        train_loss = 0
        rec_z_loss = 0
        contr_loss = 0
        VAE_loss = 0
        recon_loss = 0
        mse_vae_loss = 0
        kl_vae_loss_ = 0

        for batch_idx, (data, act, next, _) in enumerate(self.loader):
            self.optimizer.zero_grad()
            data = data.to(self.device)
            act = act.to(self.device)
            next = next.to(self.device)

            recon_batch, mu_data, log_var_data, z_data = self.model.reconstruct(data)
            emb_q = self.model.transitionZ(z_data, act)
            x_t1_hat = self.model.decode(emb_q)

            loss_vae, mse_vae, kl_vae = self.loss_function_vae(recon_batch, data, mu_data, log_var_data,
                                                               batch_idx + len(self.loader) * epoch)

            loss_recon = F.mse_loss(x_t1_hat, next, reduction='mean')

            if self.contrastive:
                with torch.no_grad():
                    z_pos = self.model_vae.getZ(next)
                    z_neg = self.model_vae.getZ(data)

                l_pos = torch.sum(emb_q * z_pos, dim=1, keepdim=True)
                l_neg = torch.mm(emb_q, z_neg.t())
                logits = torch.cat([l_pos, l_neg], dim=1)
                positive_label = torch.zeros(logits.size(0), dtype=torch.long).to(self.device)

                loss_contr = self.loss_function_contrastive(logits, positive_label)

                loss = loss_vae + loss_contr + loss_recon * 10
                contr_loss += loss_contr.item()
            else:

                with torch.no_grad():
                    z_next, _, _ = self.model.getZ(next)

                loss_rec_z = F.mse_loss(emb_q, z_next, reduction='mean')

                loss = loss_vae + loss_recon * 10 + loss_rec_z
                rec_z_loss += loss_rec_z.item()
            # print(loss_vae.item(), loss_contr.item(), loss_recon.item())

            loss.backward()

            train_loss += loss.item()
            VAE_loss += loss_vae.item()

            recon_loss += loss_recon.item()
            mse_vae_loss += mse_vae.item()
            kl_vae_loss_ += kl_vae.item()

            self.optimizer.step()

        self.writer.add_scalar("Train/Total_loss", train_loss / len(self.loader), epoch)
        self.writer.add_scalar("Train/loss_vae", VAE_loss / len(self.loader), epoch)
        self.writer.add_scalar("Train/loss_next", recon_loss / len(self.loader), epoch)
        self.writer.add_scalar("Train/mse_vae", mse_vae_loss / len(self.loader), epoch)
        self.writer.add_scalar("Train/kl_vae", kl_vae_loss_ / len(self.loader), epoch)

        if self.contrastive:
            self.writer.add_scalar("Train/loss_contr", contr_loss / len(self.loader), epoch)
        else:
            self.writer.add_scalar("Train/loss_rec_z", rec_z_loss / len(self.loader), epoch)

        if epoch % 10 == 0: self.test_world_model(epoch)
        return train_loss / len(self.loader)

    def _train_end_to_end_reward(self, epoch):
        train_loss = 0
        rec_z_loss = 0
        contr_loss = 0
        VAE_loss = 0
        recon_loss = 0
        mse_vae_loss = 0
        kl_vae_loss_ = 0
        reward_loss = 0

        for batch_idx, (data, act, next, reward) in enumerate(self.loader):
            self.optimizer.zero_grad()
            data = data.to(self.device)
            act = act.to(self.device)
            next = next.to(self.device)
            reward = reward.to(self.device)

            recon_batch, mu_data, log_var_data, z_data = self.model.reconstruct(data)
            emb_q, pred_reward = self.model.transitionZ(z_data, act)
            x_t1_hat = self.model.decode(emb_q)

            loss_vae, mse_vae, kl_vae = self.loss_function_vae(recon_batch, data, mu_data, log_var_data,
                                                               batch_idx + len(self.loader) * epoch)

            loss_recon = F.mse_loss(x_t1_hat, next, reduction='mean')

            loss_reward = F.mse_loss(pred_reward.squeeze(), reward, reduction='mean')

            if self.contrastive:
                with torch.no_grad():
                    z_pos = self.model_vae.getZ(next)
                    z_neg = self.model_vae.getZ(data)

                l_pos = torch.sum(emb_q * z_pos, dim=1, keepdim=True)
                l_neg = torch.mm(emb_q, z_neg.t())
                logits = torch.cat([l_pos, l_neg], dim=1)
                positive_label = torch.zeros(logits.size(0), dtype=torch.long).to(self.device)

                loss_contr = self.loss_function_contrastive(logits, positive_label)

                loss = loss_vae + loss_contr + loss_recon * 10 + loss_reward
                contr_loss += loss_contr.item()
            else:

                with torch.no_grad():
                    z_next, _, _ = self.model.getZ(next)

                loss_rec_z = F.mse_loss(emb_q, z_next, reduction='mean')

                loss = loss_vae + loss_recon * 10 + loss_rec_z + loss_reward
                rec_z_loss += loss_rec_z.item()
            # print(loss_vae.item(), loss_contr.item(), loss_recon.item())

            loss.backward()

            train_loss += loss.item()
            VAE_loss += loss_vae.item()
            reward_loss += loss_reward
            recon_loss += loss_recon.item()
            mse_vae_loss += mse_vae.item()
            kl_vae_loss_ += kl_vae.item()

            self.optimizer.step()

        self.writer.add_scalar("Train/Total_loss", train_loss / len(self.loader), epoch)
        self.writer.add_scalar("Train/loss_vae", VAE_loss / len(self.loader), epoch)
        self.writer.add_scalar("Train/loss_next", recon_loss / len(self.loader), epoch)
        self.writer.add_scalar("Train/mse_vae", mse_vae_loss / len(self.loader), epoch)
        self.writer.add_scalar("Train/kl_vae", kl_vae_loss_ / len(self.loader), epoch)
        self.writer.add_scalar("Train/reward", reward_loss / len(self.loader), epoch)

        if self.contrastive:
            self.writer.add_scalar("Train/loss_contr", contr_loss / len(self.loader), epoch)
        else:
            self.writer.add_scalar("Train/loss_rec_z", rec_z_loss / len(self.loader), epoch)

        if epoch % 10 == 0: self.test_world_model(epoch)
        return train_loss / len(self.loader)

    # def _train_splitted(self, epoch):
    #     train_loss = 0
    #     contr_loss = 0
    #     recon_loss = 0
    #     for batch_idx, (data, act, next, _) in enumerate(self.loader):
    #         self.optimizer_contr.zero_grad()
    #
    #         data = data.to(self.device)
    #         act = act.to(self.device)
    #         next = next.to(self.device)
    #
    #         emp_k_neg = self.model_vae.getZ(data)
    #         emp_k_pos = self.model_vae.getZ(next)
    #
    #         emb_q = self.model_contr(emp_k_neg, act)
    #
    #         x_t1_hat = self.model_vae.decode(emb_q)
    #
    #         l_pos = torch.sum(emb_q * emp_k_pos, dim=1, keepdim=True)
    #         l_neg = torch.mm(emb_q, emp_k_neg.t())
    #         logits = torch.cat([l_pos, l_neg], dim=1)
    #
    #         positive_label = torch.zeros(logits.size(0), dtype=torch.long).to(self.device)
    #         loss_recon = F.huber_loss(x_t1_hat, next, reduction='mean')
    #         loss_contr = self.loss_function_contrastive(logits / 0.2, positive_label)
    #         if self.contrastive:
    #             loss = loss_contr + loss_recon
    #         else:
    #             loss = loss_recon
    #         # print(loss_contr.item(), loss_recon.item())
    #
    #         loss.backward()
    #         train_loss += loss.item()
    #         contr_loss += loss_contr.item()
    #         recon_loss += loss_recon.item()
    #         self.optimizer_contr.step()
    #
    #     self.writer.add_scalar("Train/Total_loss", train_loss / len(self.loader), epoch)
    #     self.writer.add_scalar("Train/loss_contr", contr_loss / len(self.loader), epoch)
    #     self.writer.add_scalar("Train/loss_next", recon_loss / len(self.loader), epoch)
    #     if epoch % 10 == 0: self.test_world_model(epoch)
    #     return train_loss / len(self.loader)

    def _train_vae(self, epochs):

        # self.model_vae = VAE(input_dim=self.env.observation_space.shape[0], hidden_dim=400, z_dim=12).to(self.device)
        # self.optimizer_vae = torch.optim.Adam(self.model_vae.parameters(), lr=1e-4)

        for epoch in range(epochs):
            train_loss = 0
            sum_log_var = 0
            bce_, kl_ = 0, 0

            for batch_idx, (data, _, _, _) in enumerate(self.loader):
                self.optimizer_vae.zero_grad()
                data = data.to(self.device)
                recon_batch, mu, log_var = self.model_vae(data)
                loss, bce, kl = self.loss_function_vae(recon_batch, data, mu, log_var,
                                                       batch_idx + len(self.loader) * epoch)

                loss.backward()
                train_loss += loss.item()
                bce_ += bce.item()
                kl_ += kl.item()
                sum_log_var += torch.sum(log_var.detach()).item()

                self.optimizer_vae.step()

            self.writer.add_scalar("VAE/total_loss", train_loss / len(self.loader), epoch)
            self.writer.add_scalar("VAE/bce", bce_ / len(self.loader), epoch)
            self.writer.add_scalar("VAE/kl", kl_ / len(self.loader), epoch)
            self.writer.add_scalar("VAE/var", np.exp(sum_log_var / (len(self.loader) * 4096)), epoch)
            if epoch % 20 == 0: self.test_VAE(epoch)

        if self.savepath:
            torch.save(self.model_vae, os.path.join(self.savepath, "VAE.pt"))

    def _train_action(self, epochs):

        for epoch in range(epochs):
            train_loss = 0

            for batch_idx, (data, action, next, _) in enumerate(self.loader):
                self.optimizer_action.zero_grad()
                data = data.to(self.device)
                next = next.to(self.device)
                action = action.to(self.device)
                recon_action = self.model_action(data, next - data)
                loss = F.mse_loss(recon_action, action, reduction='mean')

                loss.backward()
                train_loss += loss.item()

                self.optimizer_action.step()

            self.writer.add_scalar("Action/train_loss", train_loss / len(self.loader), epoch)

            if epoch % 20 == 0: self.test_action(epoch)

        if self.savepath:
            torch.save(self.model_action, os.path.join(self.savepath, "action.pt"))

    def loss_function_vae(self, recon_x, x, mu, log_var, itr):

        recons_loss = F.mse_loss(recon_x, x)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        itr_ = itr % (len(self.loader) * 50)

        beta = min((itr_ / (len(self.loader) * 25)), 1)

        self.writer.add_scalar("itr", itr_, itr)

        self.writer.add_scalar("beta", beta, itr)

        kld_weight = 1 / len(self.loader)
        loss = recons_loss + beta * kld_weight * kld_loss

        return loss, recons_loss, kld_loss

    def test_VAE(self, epoch=0):

        dist_rec = 0

        for batch_idx, (data, act, next, reward) in enumerate(self.test_loader):
            data = data.to(self.device)

            with torch.no_grad():
                recon_batch, _, _ = self.model_vae(data)

            dist_rec += F.mse_loss(recon_batch, data)

            sas = np.mean(np.abs(recon_batch.cpu().numpy() - data.cpu().numpy()), axis=0)
            # print(sas)

        self.writer.add_scalar("Test_vae/MSE_Recon", dist_rec / len(self.test_loader), epoch)
        print('Recon Dist: {:.4f}'.format(dist_rec / len(self.test_loader)))

    def test_action(self, epoch=0):

        dist_rec = 0

        for batch_idx, (data, act, next, reward) in enumerate(self.test_loader):
            data = data.to(self.device)
            next = next.to(self.device)
            act = act.to(self.device)

            with torch.no_grad():
                recon_action = self.model_action(data, next - data)

            dist_rec += F.mse_loss(recon_action, act)

        self.writer.add_scalar("Action/MSE_test", dist_rec / len(self.test_loader), epoch)
        print('Action Dist: {:.4f}'.format(dist_rec / len(self.test_loader)))

    def test_world_model(self, epoch=0):
        dist_rec = 0
        dist_trans = 0
        dist_reward = 0
        for batch_idx, (data, act, next, reward) in enumerate(self.test_loader):
            data = data.to(self.device)
            act = act.to(self.device)
            next = next.to(self.device)
            reward = reward.to(self.device)

            with torch.no_grad():
                if self.model_name == "end_to_end":
                    recon_batch, _, _, _ = self.model.reconstruct(data)

                    trans_batch = self.model(data, act)
                elif self.model_name == "reward":
                    recon_batch, _, _, _ = self.model.reconstruct(data)
                    trans_batch, reward_pred = self.model(data, act)
                elif self.model_name == "splitted":
                    recon_batch, _, _ = self.model_vae(data)
                    z_data = self.model_vae.getZ(data)
                    z_next = self.model_contr(z_data, act)

                    trans_batch = self.model_vae.decode(z_next)

            dist_rec += F.mse_loss(recon_batch, data)
            dist_trans += F.mse_loss(trans_batch, next)
            if self.model_name == "reward": dist_reward += F.mse_loss(reward_pred.squeeze(), reward)

        self.writer.add_scalar("Test/MSE_Recon", dist_rec / len(self.test_loader), epoch)
        self.writer.add_scalar("Test/MSE_Trans", dist_trans / len(self.test_loader), epoch)
        self.writer.add_scalar("Test/MSE_reward", dist_reward / len(self.test_loader), epoch)

        print('Recon Dist: {:.4f}'.format(dist_rec / len(self.test_loader)))
        print('Trans Dist: {:.4f}'.format(dist_trans / len(self.test_loader)))

    def test_td3_bc(self, corr_type=0, aug=0, eps=0,
                    hyperparameter=None,
                    iterations=500000):  # corr_type 0 with model; 1 no corr; 2 mean; 3 noise; 4 remove;
        # Aug 0 No aug; 1 Perc Batch; 2 Over esitmation # 3 noise # 4 S4rl # 5 batch REW
        run = []

        max_action = float(self.env.action_space.high[0])

        mean_ = self.dataset_rl["observations"].mean(0, keepdims=True)
        std_ = self.dataset_rl["observations"].std(0, keepdims=True) + 1e-9

        if corr_type == 0:
            # if self.test_aug:
            #     self.augment_dataset(mean_, std_)
            # else:
            self.corrupt_w_worldmodel(mean_, std_)
        elif corr_type == 2:
            self.corrupt_w_mean(mean_)
        elif corr_type == 3:
            self.corrupt_w_noise()
        elif corr_type == 4:
            self.remove()
        elif corr_type == 5:
            self.new_scenario()

        replay_buffer = ReplayBuffer(state_dim=self.env.observation_space.shape[0],
                                     action_dim=self.env.action_space.shape[0], device=self.device)
        replay_buffer.convert_D4RL(self.dataset_rl)
        mean, std = replay_buffer.normalize_states()

        policy = TD3_BC_WM(state_dim=self.env.observation_space.shape[0], action_dim=self.env.action_space.shape[0],
                           max_action=max_action, world_model=self.model, aug_type=aug, writer=self.writer,
                           device=self.device, action_model=self.model_action)
        # hyperparameter = 1
        for t in range(iterations):
            # hyperparameter *= 0.999996
            policy.train(replay_buffer, writer=self.writer, batch_size=256, hyperparameter=hyperparameter)

            # Evaluate episode
            if (t + 1) % 5000 == 0:
                print(f"Time steps: {t + 1}")
                score = eval_policy(policy, self.env_name, 42, mean, std)
                run.append(score)
                self.writer.add_scalar("D4RL_score",
                                       score, t)

        return run

    def new_scenario(self, env_name="halfcheetah-medium-expert-v2"):

        env = gym.make(env_name)

        new_dataset = self.create_new_scenario_dataset(env.get_dataset(), 10000, False)

        mean_ = new_dataset["observations"].mean(0, keepdims=True)
        std_ = new_dataset["observations"].std(0, keepdims=True) + 1e-9

        new_act = torch.empty(0)
        new_obs = torch.empty(0)
        new_next = torch.empty(0)
        new_reward = torch.empty(0)
        new_terminal = torch.empty(0)
        new_timeout = torch.empty(0)

        for idx in np.array_split(np.arange(len(self.dataset_rl['observations'])), 100):
            obs = torch.Tensor(self.dataset_rl['observations'][idx])
            new_obs = torch.cat([new_obs, obs])

            obs = (obs - mean_) / std_
            act = torch.Tensor(self.dataset_rl['actions'][idx])
            reward = torch.Tensor(self.dataset_rl['rewards'][idx])
            terminal = torch.Tensor(self.dataset_rl['terminals'][idx])
            timeout = torch.Tensor(self.dataset_rl['timeouts'][idx])

            new_act = torch.cat([new_act, act])
            new_reward = torch.cat([new_reward, reward])
            new_terminal = torch.cat([new_terminal, terminal])
            new_timeout = torch.cat([new_timeout, timeout])

            obs = obs.to(self.device)
            act = act.to(self.device)
            obs = obs.to(torch.float32)
            act = act.to(torch.float32)

            next_1 = self.model(obs, act).cpu().detach()

            new_next = torch.cat([new_next, next_1])

        self.dataset_rl['observations'] = np.concatenate((self.dataset_rl['observations'], new_obs), axis=0)
        self.dataset_rl['actions'] = np.concatenate((self.dataset_rl['actions'], new_act.numpy()), axis=0)
        self.dataset_rl['rewards'] = np.concatenate((self.dataset_rl['rewards'], new_reward.numpy()), axis=0)
        self.dataset_rl['terminals'] = np.concatenate((self.dataset_rl['terminals'], new_terminal.numpy()), axis=0)
        self.dataset_rl['timeouts'] = np.concatenate((self.dataset_rl['timeouts'], new_timeout.numpy()), axis=0)
        self.dataset_rl['next_observations'] = np.concatenate((self.dataset_rl['next_observations'], new_next.numpy()),
                                                              axis=0)

    def erase_variables(self):
        self.dataset_rl['observations'][:, 2] = 0.
        self.dataset_rl['observations'][:, 6] = 0.
        self.dataset_rl['observations'][:, 7] = 0.
        self.dataset_rl['observations'][:, 10] = 0.
        self.dataset_rl['observations'][:, 16] = 0.
        self.dataset_rl['observations'][:, 9] = 0.
        self.dataset_rl['observations'][:, 14] = 0.

        self.dataset_rl['next_observations'][:, 2] = 0.
        self.dataset_rl['next_observations'][:, 6] = 0.
        self.dataset_rl['next_observations'][:, 7] = 0.
        self.dataset_rl['next_observations'][:, 10] = 0.
        self.dataset_rl['next_observations'][:, 16] = 0.
        self.dataset_rl['next_observations'][:, 9] = 0.
        self.dataset_rl['next_observations'][:, 14] = 0.

    def augment_dataset(self, mean, std):

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

            # new_obs = torch.cat([new_obs, obs])
            new_act = torch.cat([new_act, act])
            new_reward = torch.cat([new_reward, reward])
            new_terminal = torch.cat([new_terminal, terminal])
            new_timeout = torch.cat([new_timeout, timeout])

            obs = obs.to(self.device)
            act = act.to(self.device)
            obs = obs.to(torch.float32)
            act = act.to(torch.float32)

            next_1 = self.model(obs, act).cpu().detach()

            new_next = torch.cat([new_next, next_1])

        new_obs = new_next.numpy()
        new_obs[[i for i in range(len(new_obs)) if i % 1000 == 0]] = self.dataset_rl['observations'][
            [i for i in range(len(new_obs)) if i % 1000 == 0]]

        self.dataset_rl['observations'] = np.concatenate((self.dataset_rl['observations'], new_obs), axis=0)
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
        obs -= mean  # obs = (obs - mean) / std
        obs = obs.to(self.device)
        act = torch.Tensor(self.dataset_rl['actions'][self.corrupted_index]).to(self.device)

        next = self.dataset_rl['next_observations'][self.corrupted_index]

        obs = obs.to(torch.float32)
        act = act.to(torch.float32)

        prediction = torch.empty(0).to(self.device)
        reward = torch.empty(0).to(self.device)

        for idx in np.array_split(np.arange(len(obs)), 10000):
            with torch.no_grad():
                if self.model_name == "end_to_end":
                    prediction_itr = self.model(obs[idx], act[idx])
                elif self.model_name == "reward":
                    prediction_itr, reward_itr = self.model(obs[idx], act[idx])
                    reward = torch.cat([reward, reward_itr])
                elif self.model_name == "splitted":

                    z_data = self.model_vae.getZ(obs[idx])
                    z_next = self.model_contr(z_data, act[idx])

                    prediction_itr = self.model_vae.decode(z_next)
            prediction = torch.cat([prediction, prediction_itr])

        prediction = prediction.cpu().detach().numpy()
        prediction = (prediction * std) + mean

        self.dataset_rl['next_observations'][self.corrupted_index] = prediction

        if self.model_name == "reward":
            self.dataset_rl['rewards'][self.corrupted_index] = reward

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

    def train_nd_test(self, epochs, corr_type, aug_td3=0):

        self.train(epochs)
        self.test_td3_bc(corr_type, aug_td3)

    def test_distr(self):

        mse_distr = torch.empty(0)
        log_var_distr = torch.empty(0)
        self.model = self.model.to(self.device)
        for batch_idx, (data, act, next, reward) in enumerate(self.test_loader):
            data = data.to(self.device)

            act = act.to(self.device)
            next = next.to(self.device)
            reward = reward.to(self.device)

            with torch.no_grad():
                if self.model_name == "end_to_end":
                    recon_batch, _, log_var, _ = self.model.reconstruct(data)

                    trans_batch = self.model(data, act)
                elif self.model_name == "reward":
                    recon_batch, _, _ = self.model_vae(data)
                    z_data = self.model_vae.getZ(data)
                    z_next = self.model_contr(z_data, act)

                    trans_batch = self.model_vae.decode(z_next)

            log_var_distr = torch.cat([log_var_distr, log_var.cpu()])

            mse = F.mse_loss(trans_batch, next, reduce=False).detach().cpu()
            mse_distr = torch.cat([mse_distr, mse])

        log_var_distr = log_var_distr.exp().mean(1)
        print(log_var_distr.max().item())
        print(log_var_distr.mean().item())

        plt.hist(log_var_distr, bins=100)
        plt.title("Var distr " + self.env_name)
        if self.savepath:
            plt.savefig(os.path.join(self.savepath, "Var_distr_" + self.env_name + '.png'))
            plt.clf()

        for s in range(mse_distr.shape[1]):
            plt.hist(torch.where(mse_distr[:, s] > 5, 5, mse_distr[:, s]), bins=100)
            plt.title("MSE Feature " + str(s) + " " + self.env_name)
            plt.ylim([-10, 17000])
            plt.xlim([-0.5, 5])
            if self.savepath:
                plt.savefig(os.path.join(self.savepath, "Feature" + str(s) + self.env_name + '.png'))
                plt.clf()

        mse_distr = mse_distr.mean(1)
        mse_distr = torch.where(mse_distr > 2, 2, mse_distr)
        plt.hist(mse_distr, bins=100)
        plt.ylim([0, 6000])
        plt.title("Total MSE " + self.env_name)
        if self.savepath:
            plt.savefig(os.path.join(self.savepath, "MSE" + self.env_name + '.png'))
            plt.clf()

    def test_render(self):
        tmp_dataset = D4RLDataset(self.dataset)
        var = tmp_dataset.var
        mean = tmp_dataset.mean

        obs, act, _next, _ = next(iter(self.test_loader))
        obs = obs.to(self.device)
        act = act.to(self.device)
        _next = _next.to(self.device)

        if self.model_name == "end_to_end":
            prediction = self.model(obs, act)
        elif self.model_name == "reward":
            prediction, _ = self.model(obs, act)
        elif self.model_name == "splitted":

            z_data = self.model_vae.getZ(obs)
            z_next = self.model_contr(z_data, act)

            prediction = self.model_vae.decode(z_next)

        mse = (prediction - _next).pow(2).mean(1)  # todo fix

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

    def test_depth_mse(self):
        tmp_dataset = D4RLDataset(self.dataset_test)
        mse_depth = []
        starting_point = np.random.randint(0, 20000 - 10, size=(1000))  # - tmp_dataset.mean) / tmp_dataset.std

        act_itr = torch.FloatTensor(self.dataset_test['actions'][starting_point]).to(self.device)
        obs_itr = torch.FloatTensor(self.dataset_test['observations'][starting_point]).to(self.device)
        next_itr = torch.FloatTensor(self.dataset_test['next_observations'][starting_point]).to(
            self.device)  # [starting_point]

        mean = torch.FloatTensor(tmp_dataset.mean).to(self.device)
        std = torch.FloatTensor(tmp_dataset.std).to(self.device)

        obs_itr -= mean
        # obs_itr /= std

        next_itr -= mean
        # next_itr /= var

        for i in range(10):
            with torch.no_grad():
                prediction = self.model(obs_itr, act_itr)
                reconstruction, _, _, _ = self.model.reconstruct(obs_itr)

            # pred = mean + prediction #(prediction * std) + mean
            if i == 0: print("Recon Error",
                             np.mean(np.square(reconstruction.cpu().numpy() - obs_itr.cpu().numpy()), axis=0))
            if i == 0: print("Trans Error",
                             np.mean(np.square(prediction.cpu().numpy() - next_itr.cpu().numpy()), axis=0))

            mse_depth.append(F.mse_loss(prediction, next_itr).detach().cpu().item())

            act_itr = torch.Tensor(self.dataset_test['actions'][starting_point + i + 1]).to(self.device)
            obs_itr = prediction.detach()
            next_itr = torch.Tensor(self.dataset_test['next_observations'][starting_point + i + 1]).to(self.device)

            next_itr -= mean
            # next_itr /= std

        print(mse_depth)
