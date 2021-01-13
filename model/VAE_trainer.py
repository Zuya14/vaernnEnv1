import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

from .VAE import VAE

class VAE_trainer:

    def __init__(self, first_channel, latent_size, red_times, repeat, channel_inc, device='cpu'):
        self.vae = VAE(first_channel, latent_size, red_times, repeat, channel_inc)
        self.vae = self.vae.to(device, non_blocking=True)
        self.device = device

        self.optimizer = optim.Adam(self.vae.parameters())

    def train(self, train_data, k=1.0):
        self.vae.train()

        mse_loss = 0
        KLD_loss = 0

        for batch_idx, data in enumerate(train_data):
            data = data.to(self.device).view(-1, 1, 1080)

            self.optimizer.zero_grad()
            
            recon_batch, mu, logvar = self.vae(data)
            
            mse, KLD = self.vae.loss_function(recon_batch, data, mu, logvar)
            
            KLD = k*KLD
            loss = mse + KLD
            loss.backward()
            
            nn.utils.clip_grad_norm_(self.vae.parameters(), 1000, norm_type=2)
            
            mse_loss += mse.item()
            KLD_loss += KLD.item()
            
            self.optimizer.step()

        mse_loss /= len(train_data)
        KLD_loss /= len(train_data)

        return mse_loss, KLD_loss

    def test(self, test_data, k=1.0):
        self.vae.eval()

        test_loss = 0
        
        with torch.no_grad():
            for batch_idx, data in enumerate(test_data):
                data = data.to(self.device).view(-1, 1, 1080)

                recon_batch, mu, logvar = self.vae(data)
                
                mse, KLD = self.vae.loss_function(recon_batch, data, mu, logvar)
                KLD = k*KLD
                
                loss = mse + KLD
                test_loss += loss.item()
                        
        test_loss /= len(test_data)

        return test_loss

    def train2(self, train_data, k=1.0):
        self.vae.train()

        mse_loss = 0
        KLD_loss = 0

        mu_array = []

        for batch_idx, data in enumerate(train_data):
            data = data.to(self.device).view(-1, 1, 1080)

            self.optimizer.zero_grad()
            
            recon_batch, mu, logvar = self.vae(data)
            
            mse, KLD = self.vae.loss_function(recon_batch, data, mu, logvar)
            
            KLD = k*KLD
            loss = mse + KLD
            loss.backward()
            
            nn.utils.clip_grad_norm_(self.vae.parameters(), 100, norm_type=2)
            
            mse_loss += mse.item()
            KLD_loss += KLD.item()
            
            self.optimizer.step()

            mu_array.append(mu.detach())

        mse_loss /= len(train_data)
        KLD_loss /= len(train_data)

        return mse_loss, KLD_loss, torch.stack(mu_array, dim=0)

    def test2(self, test_data, k=1.0):
        self.vae.eval()

        test_loss = 0
        
        mu_array = []

        with torch.no_grad():
            for batch_idx, data in enumerate(test_data):
                data = data.to(self.device).view(-1, 1, 1080)

                recon_batch, mu, logvar = self.vae(data)
                
                mse, KLD = self.vae.loss_function(recon_batch, data, mu, logvar)
                KLD = k*KLD
                
                loss = mse + KLD
                test_loss += loss.item()
                        
                mu_array.append(mu.detach())

        test_loss /= len(test_data)

        return test_loss, torch.stack(mu_array, dim=0)

    def save(self, path):
        torch.save(self.vae.to('cpu').state_dict(), path)
        self.vae.to(self.device)

    def load(self, path):
        self.vae.load_state_dict(torch.load(path))