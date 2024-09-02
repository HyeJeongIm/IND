import torch
from torch import optim
import torch.nn.functional as F

import numpy as np
from barbar import Bar

from model import autoencoder, network
from utils.utils import weights_init_normal

import os

class TrainerDeepSVDD:
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device

        self.net = network(self.args.latent_dim).to(self.device)
        self.result_dir = os.path.join(self.args.output_path, self.args.dataset_name)
        self.fine_tune_results_dir = os.path.join(self.args.output_path, "IND_buffer", f"{self.args.testdataset_version}_fine_tune")
        self.pretrained_weights_path = f'{self.result_dir}/pretrained_parameters.pth'
        self.trained_weights_path = f'{self.result_dir}/trained_parameters.pth'
        self.fine_tune_weights_path = f'{self.result_dir}/fine_tuned_parameters.pth'
        self.buffer_fine_tune_weights_path = f'{self.fine_tune_results_dir}/buffer_fine_tuned_parameters.pth'

        self.ensure_directory_exists(self.result_dir)
        self.ensure_directory_exists(self.fine_tune_results_dir)


    def ensure_directory_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")

    
    def pretrain(self):
        """ Pretraining the weights for the deep SVDD network using autoencoder"""
        ae = autoencoder(self.args.latent_dim).to(self.device)
        ae.apply(weights_init_normal)
        optimizer = optim.Adam(ae.parameters(), lr=self.args.lr_ae,
                               weight_decay=self.args.weight_decay_ae)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=self.args.lr_milestones, gamma=0.1)
        
        ae.train()
        for epoch in range(self.args.num_epochs_ae):
            total_loss = 0
            for x, _ in Bar(self.train_loader):
                x = x.float().to(self.device)
                
                optimizer.zero_grad()
                x_hat = ae(x)
                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()
                
                total_loss += reconst_loss.item()
            scheduler.step()
            print('Pretraining Autoencoder... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))
        self.save_weights_for_DeepSVDD(ae, self.train_loader) 
    

    def save_weights_for_DeepSVDD(self, model, dataloader):
        """Initialize Deep SVDD weights using the encoder weights of the pretrained autoencoder."""
        c = self.set_c(model, dataloader)
        self.net = network(self.args.latent_dim).to(self.device)
        state_dict = model.state_dict()
        self.net.load_state_dict(state_dict, strict=False)
        torch.save({'center': c.cpu().numpy(), 'net_dict': self.net.state_dict()}, self.pretrained_weights_path)

    def set_c(self, model, dataloader, eps=0.1):
        """Initializing the center for the hypersphere"""
        model.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.float().to(self.device)
                z = model.encode(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def train(self):
        """Training the Deep SVDD model"""
        
        if self.args.pretrain==True:
            state_dict = torch.load(self.pretrained_weights_path)
            self.net.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            self.net.apply(weights_init_normal)
            c = torch.randn(self.args.latent_dim).to(self.device)
        
        optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr,
                               weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=self.args.lr_milestones, gamma=0.1)

        self.net.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in Bar(self.train_loader):
                x = x.float().to(self.device)

                optimizer.zero_grad()
                z = self.net(x)
                loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            scheduler.step()
            print('Training Deep SVDD... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))
        self.net = self.net
        self.c = c
        torch.save({'center': self.c.cpu().numpy(), 'net_dict': self.net.state_dict()}, self.trained_weights_path)

    def fine_tune(self):
        """Fine-tuning the Deep SVDD model"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.lr_milestones, gamma=0.1)

        self.net.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in Bar(self.test_loader):
                x = x.float().to(self.device)

                optimizer.zero_grad()
                z = self.net(x)
                loss = torch.mean(torch.sum((z - self.c) ** 2, dim=1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            scheduler.step()
            print(f'Fine-tuning Deep SVDD... Epoch: {epoch + 1}, Loss: {total_loss / len(self.test_loader):.3f}')
        
        # Save fine-tuned weights
        torch.save({'center': self.c.cpu().numpy(), 'net_dict': self.net.state_dict()}, self.fine_tune_weights_path)

    def buffer_fine_tune(self):
        """Fine-tuning the Deep SVDD model"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.lr_milestones, gamma=0.1)

        self.net.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in Bar(self.test_loader):
                x = x.float().to(self.device)

                optimizer.zero_grad()
                z = self.net(x)
                loss = torch.mean(torch.sum((z - self.c) ** 2, dim=1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            scheduler.step()
            print(f'Fine-tuning Deep SVDD... Epoch: {epoch + 1}, Loss: {total_loss / len(self.test_loader):.3f}')
        
        # Save fine-tuned weights
        torch.save({'center': self.c.cpu().numpy(), 'net_dict': self.net.state_dict()}, self.buffer_fine_tune_weights_path)

                

                

        