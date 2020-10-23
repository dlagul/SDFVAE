import torch
import os
import time
import argparse
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from model import *
from tqdm import *
from util import KpiReader
from logger import Logger

class Trainer(object):
    def __init__(self, model, train, trainloader, log_path='log_trainer',
                 log_file='loss', epochs=50, batch_size=64, learning_rate=0.001,
                 checkpoints='kpi_model.path', checkpoints_interval = 10, device=torch.device('cuda:0')):
        self.trainloader = trainloader
        self.train = train
        self.log_path = log_path
        self.log_file = log_file
        self.start_epoch = 0
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.model.to(device)
        self.learning_rate = learning_rate
        self.checkpoints = checkpoints
        self.checkpoints_interval = checkpoints_interval
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        self.epoch_losses = []
        self.loss = {}
        self.logger = Logger(self.log_path, self.log_file)
 
    def save_checkpoint(self, epoch):
        torch.save({'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'losses': self.epoch_losses},
                    self.checkpoints + '_epochs{}.pth'.format(epoch+1))

    def load_checkpoint(self, start_ep):
        try:
            print ("Loading Chechpoint from ' {} '".format(self.checkpoints+'_epochs{}.pth'.format(start_ep)))
            checkpoint = torch.load(self.checkpoints+'_epochs{}.pth'.format(start_ep))
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch_losses = checkpoint['losses']
            print ("Resuming Training From Epoch {}".format(self.start_epoch))
        except:
            print ("No Checkpoint Exists At '{}', Starting Fresh Training".format(self.checkpoints))
            self.start_epoch = 0

    def loss_fn(self, original_seq, recon_seq_mu, recon_seq_logvar, s_mean, 
                s_logvar, d_post_mean, d_post_logvar, d_prior_mean, d_prior_logvar):
        batch_size = original_seq.size(0)
        loglikelihood = -0.5 * torch.sum(torch.pow(((original_seq.float()-recon_seq_mu.float())/torch.exp(recon_seq_logvar.float())), 2) 
                                         + 2 * recon_seq_logvar.float() 
                                         + np.log(np.pi*2))
        kld_s = -0.5 * torch.sum(1 + s_logvar - torch.pow(s_mean, 2) - torch.exp(s_logvar))
        
        d_post_var = torch.exp(d_post_logvar)
        d_prior_var = torch.exp(d_prior_logvar)
        kld_d = 0.5 * torch.sum(d_prior_logvar - d_post_logvar 
                                + ((d_post_var + torch.pow(d_post_mean - d_prior_mean, 2)) / d_prior_var) 
                                - 1)
        return (-loglikelihood + kld_s + kld_d)/batch_size, loglikelihood/batch_size, kld_s/batch_size, kld_d/batch_size

    def train_model(self):
        self.model.train()
        for epoch in range(self.start_epoch, self.epochs):
            losses = []
            llhs = []
            kld_ss = []
            kld_ds = []
            print ("Running Epoch : {}".format(epoch+1))
            for i, dataitem in tqdm(enumerate(self.trainloader,1)):
                _,_,data = dataitem
                data = data.to(self.device)
                self.optimizer.zero_grad()
                s_mean, s_logvar, s, d_post_mean, d_post_logvar, d, d_prior_mean, d_prior_logvar, recon_x_mu, recon_x_logvar = self.model(data)
                loss, llh, kld_s, kld_d = self.loss_fn(data, recon_x_mu, recon_x_logvar, s_mean, s_logvar, 
                                                       d_post_mean, d_post_logvar, d_prior_mean, d_prior_logvar)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                llhs.append(llh.item())
                kld_ss.append(kld_s.item())
                kld_ds.append(kld_d.item())
            meanloss = np.mean(losses)
            meanllh = np.mean(llhs)
            means = np.mean(kld_ss)
            meand = np.mean(kld_ds)
            self.epoch_losses.append(meanloss)
            print ("Epoch {} : Average Loss: {} Loglikelihood: {} KL of s: {} KL of d: {}".format(epoch+1, meanloss, meanllh, means, meand))
            self.loss['Epoch'] = epoch+1
            self.loss['Avg_loss'] = meanloss
            self.loss['Llh'] = meanllh
            self.loss['KL_s'] = means
            self.loss['KL_d'] = meand
            self.logger.log_trainer(epoch+1, self.loss)
            if (self.checkpoints_interval > 0
                and (epoch+1)  % self.checkpoints_interval == 0):
                self.save_checkpoint(epoch)
        print ("Training is complete!")


def main():
    parser = argparse.ArgumentParser()
    # GPU option
    parser.add_argument('--gpu_id', type=int, default=0)
    # Dataset options
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--data_nums', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--T', type=int, default=20)
    parser.add_argument('--win_size', type=int, default=36)
    parser.add_argument('--l', type=int, default=10)
    parser.add_argument('--n', type=int, default=24)

    # Model options
    parser.add_argument('--s_dims', type=int, default=8)
    parser.add_argument('--d_dims', type=int, default=10)
    parser.add_argument('--conv_dims', type=int, default=100)
    parser.add_argument('--hidden_dims', type=int, default=40)
    parser.add_argument('--enc_dec', type=str, default='CNN')

    # Training options
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--checkpoints_path', type=str, default='model')
    parser.add_argument('--checkpoints_file', type=str, default='')
    parser.add_argument('--checkpoints_interval', type=int, default=10)
    parser.add_argument('--log_path', type=str, default='log_trainer')
    parser.add_argument('--log_file', type=str, default='')

    args = parser.parse_args()

    # Set up GPU
    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device('cuda:%d' % args.gpu_id)
    else:
        device = torch.device('cpu')
    
    if not os.path.exists(args.dataset_path):
        raise ValueError('Unknown dataset path: {}'.format(args.dataset_path))

    if args.data_nums == 0:
        raise ValueError('Wrong data numbers: {}'.format(args.data_nums))
   
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    if not os.path.exists(args.checkpoints_path):
        os.makedirs(args.checkpoints_path)
    if args.checkpoints_file == '':
        args.checkpoints_file = 'sdim{}_ddim{}_cdim{}_hdim{}_winsize{}_T{}_l{}'.format(
                                 args.s_dims,
                                 args.d_dims,
                                 args.conv_dims,
                                 args.hidden_dims,
                                 args.win_size,
                                 args.T,args.l)
    if args.log_file == '':
        args.log_file = 'sdim{}_ddim{}_cdim{}_hdim{}_winsize{}_T{}_l{}_loss'.format(
                         args.s_dims,
                         args.d_dims,
                         args.conv_dims,
                         args.hidden_dims,
                         args.win_size,
                         args.T,args.l)
        
    kpi_value_train = KpiReader(args.dataset_path, args.data_nums)

    train_loader = torch.utils.data.DataLoader(kpi_value_train, 
                                               batch_size  = args.batch_size, 
                                               shuffle     = True, 
                                               num_workers = args.num_workers)
 
    sdfvae = SDFVAE(s_dim      = args.s_dims, 
                    d_dim      = args.d_dims, 
                    conv_dim   = args.conv_dims, 
                    hidden_dim = args.hidden_dims, 
                    T          = args.T, 
                    w          = args.win_size, 
                    n          = args.n, 
                    enc_dec    = args.enc_dec, 
                    device     = device)

    trainer = Trainer(sdfvae, kpi_value_train, train_loader, 
                      log_path             = args.log_path, 
                      log_file             = args.log_file, 
                      batch_size           = args.batch_size, 
                      epochs               = args.epochs, 
                      learning_rate        = args.learning_rate,
                      checkpoints          = os.path.join(args.checkpoints_path,args.checkpoints_file), 
                      checkpoints_interval = args.checkpoints_interval, 
                      device               = device)

    trainer.load_checkpoint(args.start_epoch)
    trainer.train_model()

if __name__ == '__main__':
    main() 
