import os
import random

import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from DatasetBuilder import D4RLDatasetTrain, D4RLDatasetTest
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

    def __init__(self, model_name, env_name, savepath=None, contrastive=True):
        self.savepath = savepath
        self.model_name = model_name
        self.env_name = env_name
        self.contrastive = contrastive

        self.env = gym.make(env_name)

        self.dataset = self.env.get_dataset()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader = DataLoader(D4RLDatasetTrain(self.dataset), batch_size=2048, shuffle=True)
        self.test_loader = DataLoader(D4RLDatasetTest(self.dataset), batch_size=500, shuffle=True)

        self.writer = SummaryWriter('Manager_test_03/' + model_name + ('' if contrastive else '_no_contrastive')+ '/' + datetime.now().strftime("%m-%d-%Y_%H:%M"))
        self.loss_function_contrastive = nn.CrossEntropyLoss()

        if model_name == "end_to_end":

            self.model = Contrastive_world_model_end_to_end(input_dim=26, hidden_dim=400, z_dim=12, action_dim=6,
                                                            hidden_dim_head=200).to(self.device)

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

            self.train_fn = self._train_end_to_end

        elif model_name == "splitted":

            self.model_vae = VAE(input_dim=26, hidden_dim=400, z_dim=12).to(self.device)
            self.optimizer_vae = torch.optim.Adam(self.model_vae.parameters(), lr=1e-3)

            self.model_contr = ContrastiveHead(z_dim=12, action_dim=6, hidden_dim=200).to(self.device)
            self.optimizer_contr = torch.optim.Adam(self.model_contr.parameters(), lr=1e-4)

            self.train_fn = self._train_splitted

    def load(self, load_path):
        if self.model_name == "end_to_end":

            self.model = torch.load(os.path.join(load_path, self.model_name + ('' if self.contrastive else '_no_contrastive')+ ".pt"))

        elif self.model_name == "splitted":

            self.model_vae = torch.load(os.path.join(load_path, "VAE.pt"))
            self.model_contr = torch.load(os.path.join(load_path, self.model_name + ('' if self.contrastive else '_no_contrastive')+ ".pt"))

    def train(self, epochs):
        if self.model_name == "splitted":
            self._train_vae(100)

        for epoch in range(epochs):
            loss = self.train_fn(epoch)
            print('Epoch {}/{}, Loss: {:.4f}'.format(epoch + 1, 100, loss))

        if self.savepath:
            if self.model_name == "end_to_end":
                torch.save(self.model, os.path.join(self.savepath, self.model_name + ('' if self.contrastive else '_no_contrastive')+ ".pt"))
            elif self.model_name == "splitted":
                torch.save(self.model_contr, os.path.join(self.savepath, self.model_name + ('' if self.contrastive else '_no_contrastive') + ".pt"))

        return

    def close_writer(self):
        self.writer.flush()
        self.writer.close()

    def _train_end_to_end(self, epoch):
        train_loss = 0
        contr_loss = 0
        VAE_loss = 0
        recon_loss = 0

        for batch_idx, (data, act, next) in enumerate(self.train_loader):
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
            loss_recon = F.mse_loss(x_t1_hat, next, reduction='mean')
            loss_contr = self.loss_function_contrastive(logits / 0.20, positive_label)

            loss = loss_vae + loss_contr * (1 if self.contrastive else 0) + loss_recon

            loss.backward()

            train_loss += loss.item()
            VAE_loss += loss_vae.item()
            contr_loss += loss_contr.item()
            recon_loss += loss_recon.item()

            self.optimizer.step()

        self.writer.add_scalar("Train/Total_loss", train_loss / len(self.train_loader), epoch)
        self.writer.add_scalar("Train/loss_vae", VAE_loss / len(self.train_loader), epoch)
        self.writer.add_scalar("Train/loss_next", recon_loss / len(self.train_loader), epoch)
        self.writer.add_scalar("Train/loss_contr", contr_loss / len(self.train_loader), epoch)
        self.test_world_model(epoch)
        return train_loss / len(self.train_loader)

    def _train_splitted(self, epoch):
        train_loss = 0
        contr_loss = 0
        recon_loss = 0
        for batch_idx, (data, act, next) in enumerate(self.train_loader):
            self.optimizer_contr.zero_grad()

            data = data.to(self.device)
            act = act.to(self.device)
            next = next.to(self.device)

            emp_k_neg = self.model_vae.getZ(data)
            emp_k_pos = self.model_vae.getZ(next)

            emb_q = self.model_contr(emp_k_neg, act)

            x_t1_hat = self.model_vae.decode(emb_q)

            l_pos = torch.sum(emb_q * emp_k_pos, dim=1, keepdim=True)
            l_neg = torch.mm(emb_q, emp_k_neg.t())
            logits = torch.cat([l_pos, l_neg], dim=1)

            positive_label = torch.zeros(logits.size(0), dtype=torch.long).to(self.device)
            loss_recon = F.mse_loss(x_t1_hat, next, reduction='mean')
            loss_contr = self.loss_function_contrastive(logits / 0.2, positive_label)
            loss = loss_contr * (1 if self.contrastive else 0) + loss_recon

            loss.backward()
            train_loss += loss.item()
            contr_loss += loss_contr.item()
            recon_loss += loss_recon.item()
            self.optimizer_contr.step()

        self.writer.add_scalar("Train/loss_total", train_loss / len(self.train_loader), epoch)
        self.writer.add_scalar("Train/loss_contrastive", contr_loss / len(self.train_loader), epoch)
        self.writer.add_scalar("Train/loss_recon", recon_loss / len(self.train_loader), epoch)
        self.test_world_model(epoch)
        return train_loss / len(self.train_loader)

    def _train_vae(self, epochs):
        for epoch in range(epochs):
            train_loss = 0
            for batch_idx, (data, _, _) in enumerate(self.train_loader):
                self.optimizer_vae.zero_grad()
                data = data.to(self.device)
                recon_batch, mu, log_var = self.model_vae(data)
                loss = self.loss_function_vae(recon_batch, data, mu, log_var)

                loss.backward()
                train_loss += loss.item()
                self.optimizer_vae.step()
            self.writer.add_scalar("Loss/VAE", train_loss / len(self.train_loader), epoch)
        if self.savepath:
            torch.save(self.model_vae, os.path.join(self.savepath, "VAE.pt"))

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
                elif self.model_name == "splitted":
                    recon_batch, _, _ = self.model_vae(data)
                    z_data = self.model_vae.getZ(data)
                    z_next = self.model_contr(z_data, act)

                    trans_batch = self.model_vae.decode(z_next)

            dist_rec += F.mse_loss(recon_batch, data)
            dist_trans += F.mse_loss(trans_batch, next)

        self.writer.add_scalar("Test/MSE_Recon", dist_rec / len(self.test_loader), epoch)
        self.writer.add_scalar("Test/MSE_Trans", dist_trans / len(self.test_loader), epoch)

        print('Testing VAE NewData Recon Dist: {:.4f}'.format(dist_rec / len(self.test_loader)))
        print('Testing VAE NewData Trans Dist: {:.4f}'.format(dist_trans / len(self.test_loader)))

    def test_td3_bc(self, perc, corr_type=0): #0 with model; 1 no corr; 2 mean; 3 noise; 4 remove

        max_action = float(self.env.action_space.high[0])

        self.dataset = self.env.get_dataset()

        replay_buffer = ReplayBuffer(state_dim=26, action_dim=6)
        replay_buffer.convert_D4RL(self.dataset)
        mean, std = replay_buffer.normalize_states()

        if corr_type==0:
            self.corrupt_dataset(mean, std, perc)
        elif corr_type==2:
            self.corrupt_with_mean(mean, perc)
        elif corr_type == 3:
            self.corrupt_with_noise(perc)
        elif corr_type == 4:
            self.remove(perc)

        replay_buffer_cor = ReplayBuffer(state_dim=26, action_dim=6)
        replay_buffer_cor.convert_D4RL(self.dataset)
        mean, std = replay_buffer_cor.normalize_states()

        policy = TD3_BC(state_dim=26, action_dim=6, max_action=max_action, device=self.device)

        for t in range(500000):
            policy.train(replay_buffer_cor, batch_size=256)
            # Evaluate episode
            if (t + 1) % 5000 == 0:
                print(f"Time steps: {t + 1}")

                self.writer.add_scalar("D4RL_score",
                                       eval_policy(policy, self.env_name, 42, mean, std), t)



    def corrupt_with_mean(self, mean, perc=0.3):
        np.random.seed(42)
        index = np.random.choice(len(self.dataset['next_observations']),
                                 int(len(self.dataset['next_observations']) * perc))

        self.dataset['next_observations'][index] = mean

        self.dataset['observations'][index + 1] = mean

    def corrupt_dataset(self, mean, std, perc=0.3):

        np.random.seed(42)
        index = np.random.choice(len(self.dataset['next_observations']),
                                 int(len(self.dataset['next_observations']) * perc))

        obs = torch.Tensor(self.dataset['observations'][index])
        obs = (obs - mean) / std
        obs = obs.to(self.device)
        act = torch.Tensor(self.dataset['actions'][index]).to(self.device)

        prediction = torch.empty(0).to(self.device)

        for idx in np.array_split(np.arange(len(obs)), 3):
            if self.model_name == "end_to_end":
                prediction_itr = self.model(obs[idx], act[idx])
            elif self.model_name == "splitted":

                z_data = self.model_vae.getZ(obs[idx])
                z_next = self.model_contr(z_data, act[idx])

                prediction_itr = self.model_vae.decode(z_next)

            prediction = torch.cat([prediction, prediction_itr])

        prediction = prediction.cpu().detach().numpy()
        prediction = (prediction * std) + mean

        self.dataset['next_observations'][index] = prediction

        self.dataset['observations'][index + 1] = prediction



    def remove(self, perc):
        np.random.seed(42)
        index = np.random.choice(len(self.dataset['next_observations']),
                                 int(len(self.dataset['next_observations']) * perc))
        self.dataset['actions'] = np.delete(self.dataset['actions'], index, axis=0)
        self.dataset['infos/action_log_probs'] = np.delete(self.dataset['infos/action_log_probs'], index, axis=0)
        self.dataset['next_observations'] = np.delete(self.dataset['next_observations'], index, axis=0)
        self.dataset['observations'] = np.delete(self.dataset['observations'], index, axis=0)
        self.dataset['rewards'] = np.delete(self.dataset['rewards'], index, axis=0)
        self.dataset['terminals'] = np.delete(self.dataset['terminals'], index, axis=0)
        self.dataset['timeouts'] = np.delete(self.dataset['timeouts'], index, axis=0)

    def corrupt_with_noise(self, perc):
        np.random.seed(42)
        index = np.random.choice(len(self.dataset['next_observations']),
                                 int(len(self.dataset['next_observations']) * perc))

        noisy = self.dataset['next_observations'][index] + np.random.normal(0,1,self.dataset['next_observations'][index].shape)
        self.dataset['next_observations'][index] = noisy

        self.dataset['observations'][index + 1] = noisy


    def test_render(self):
        tmp_dataset = D4RLDatasetTrain(self.dataset)
        var = tmp_dataset.var
        mean = tmp_dataset.mean

        obs, act, _next = next(iter(self.test_loader))
        obs = obs.to(self.device)
        act = act.to(self.device)
        _next = _next.to(self.device)

        if self.model_name == "end_to_end":
            prediction = self.model(obs, act)
        elif self.model_name == "splitted":

            z_data = self.model_vae.getZ(obs)
            z_next = self.model_contr(z_data, act)

            prediction = self.model_vae.decode(z_next)

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
