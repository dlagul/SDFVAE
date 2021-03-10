import torch
import os
import argparse
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from model import *
from tqdm import *
from util import KpiReader
from logger import Logger

class Tester(object):
    def __init__(self, model, device, test, testloader, log_path='log_tester', log_file='loss', learning_rate=0.0002, checkpoints=None):
        self.model = model
        self.model.to(device)
        self.device = device
        self.test = test
        self.testloader = testloader
        self.log_path = log_path
        self.log_file = log_file
        self.learning_rate = learning_rate
        self.checkpoints = checkpoints
        self.start_epoch = 0
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        self.epoch_losses = []
        self.logger = Logger(self.log_path, self.log_file)         
        self.loss = {}


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
            print ("No Checkpoint Exists At '{}', Starting Fresh Training".format(self.checkpoints+'_epochs{}.pth'.format(start_ep)))
            self.start_epoch = 0

    def model_test(self):
        self.model.eval()
        for i, dataitem in enumerate(self.testloader,1):
            timestamps,labels,data = dataitem
            data = data.to(self.device)
            s_mean, s_logvar, s, d_post_mean, d_post_logvar, d, d_prior_mean, d_prior_logvar, recon_x_mu, recon_x_logvar = self.forward_test(data)
            avg_loss, llh, kld_s, kld_d = self.loss_fn(data, recon_x_mu, recon_x_logvar, s_mean, s_logvar, 
                                      d_post_mean, d_post_logvar, d_prior_mean, d_prior_logvar) 
            last_timestamp = timestamps[-1,-1,-1,-1]
            label_last_timestamp_tensor = labels[-1,-1,-1,-1]
            
            anomaly_index = (label_last_timestamp_tensor.numpy() == 1)
            anomaly_nums = len(label_last_timestamp_tensor.numpy()[anomaly_index])
            if anomaly_nums >= 1:
                isanomaly = "Anomaly"
            else:
                isanomaly = "Normaly"
            llh_last_timestamp = self.loglikelihood_last_timestamp(data[-1,-1,-1,:,-1], 
                                                                   recon_x_mu[-1,-1,-1,:,-1],
                                                                   recon_x_logvar[-1,-1,-1,:,-1])
            
            self.loss['Last_timestamp'] = last_timestamp.item()
            self.loss['Avg_loss'] = avg_loss.item()
            self.loss['Llh'] = llh.item()
            self.loss['KL_s'] = kld_s.item()
            self.loss['KL_d'] = kld_d.item()
            self.loss['Llh_Lt'] = llh_last_timestamp.item()
            self.loss['IA'] = isanomaly
            self.logger.log_tester(self.start_epoch, self.loss)
        print ("Testing is complete!")

    def forward_test(self, data):
        with torch.no_grad():
            s_mean, s_logvar, s, d_post_mean, d_post_logvar, d, d_prior_mean, d_prior_logvar, recon_x_mu, recon_x_logvar= self.model(data)
            return s_mean, s_logvar, s, d_post_mean, d_post_logvar, d, d_prior_mean, d_prior_logvar, recon_x_mu, recon_x_logvar



    def loss_fn(self, original_seq, recon_seq_mu, recon_seq_logvar, s_mean, s_logvar, d_post_mean, d_post_logvar, d_prior_mean, d_prior_logvar):
        batch_size = original_seq.size(0)
        # See https://arxiv.org/pdf/1606.05908.pdf, Page 9, Section 2.2 for details.
        # N(x|mu,var)
        # = log{1/(sqrt(2*pi)*var)exp{-(x-mu)^2/(2*var^2)}} 
        # = -0.5*{log(2*pi)+2*log(var)+[(x-mu)/exp{log(var)}]^2}
        loglikelihood = -0.5 * torch.sum(torch.pow(((original_seq.float()-recon_seq_mu.float())/torch.exp(recon_seq_logvar.float())), 2) 
                                         + 2 * recon_seq_logvar.float() 
                                         + np.log(np.pi*2))
        # See https://arxiv.org/pdf/1606.05908.pdf, Page 9, Section 2.2, Equation (7) for details.
        kld_s = -0.5 * torch.sum(1 + s_logvar - torch.pow(s_mean, 2) - torch.exp(s_logvar))
        # See https://arxiv.org/pdf/1606.05908.pdf, Page 9, Section 2.2, Equation (6) for details.
        d_post_var = torch.exp(d_post_logvar)
        d_prior_var = torch.exp(d_prior_logvar)
        kld_d = 0.5 * torch.sum(d_prior_logvar - d_post_logvar + 
                                ((d_post_var + torch.pow(d_post_mean - d_prior_mean, 2)) / d_prior_var) 
                                - 1)
        return (-loglikelihood + kld_s + kld_d)/batch_size, loglikelihood/batch_size, kld_s/batch_size, kld_d/batch_size

    
    def loglikelihood_last_timestamp(self, x, recon_x_mu, recon_x_logvar):
        # See https://arxiv.org/pdf/1606.05908.pdf, Page 9, Section 2.2 for details.
        # N(x|mu,var)
        # = log{1/(sqrt(2*pi)*var)exp{-(x-mu)^2/(2*var^2)}} 
        # = -0.5*{log(2*pi)+2*log(var)+[(x-mu)/exp{log(var)}]^2}
        llh = -0.5 * torch.sum(torch.pow(((x.float()-recon_x_mu.float())/torch.exp(recon_x_logvar.float())), 2) 
              + 2 * recon_x_logvar.float() 
              + np.log(np.pi*2))
        return llh


def main():
    parser = argparse.ArgumentParser()
    # GPU option
    parser.add_argument('--gpu_id', type=int, default=1)
    # Dataset options
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
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

    # Training and testing options
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=30)
    parser.add_argument('--checkpoints_path', type=str, default='model')
    parser.add_argument('--checkpoints_file', type=str, default='')
    parser.add_argument('--checkpoints_interval', type=int, default=10)
    parser.add_argument('--log_path', type=str, default='log_tester')
    parser.add_argument('--log_file', type=str, default='') 
 
    args = parser.parse_args()
    
    # Set up GPU
    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device('cuda:%d' % args.gpu_id)
    else:
        device = torch.device('cpu')
    
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    
    if not os.path.exists(args.dataset_path):
        raise ValueError('Unknown dataset path: {}'.format(args.dataset_path))

    if not os.path.exists(args.checkpoints_path):
        raise ValueError('Unknown checkpoints path: {}'.format(checkpoints_path))

    if args.checkpoints_file == '':
        args.checkpoints_file = 'sdim{}_ddim{}_cdim{}_hdim{}_winsize{}_T{}_l{}'.format(
                                 args.s_dims,
                                 args.d_dims,
                                 args.conv_dims,
                                 args.hidden_dims,
                                 args.win_size,
                                 args.T,
                                 args.l)
    if args.log_file == '':
        args.log_file = 'sdim{}_ddim{}_cdim{}_hdim{}_winsize{}_T{}_l{}_epochs{}_loss'.format(
                         args.s_dims,
                         args.d_dims,
                         args.conv_dims,
                         args.hidden_dims,
                         args.win_size,
                         args.T,
                         args.l,
                         args.start_epoch)
    
    kpi_value_test = KpiReader(args.dataset_path)
    
    test_loader = torch.utils.data.DataLoader(kpi_value_test, 
                                              batch_size  = args.batch_size, 
                                              shuffle     = False, 
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

    tester = Tester(sdfvae, device, kpi_value_test, test_loader,
                    log_path      = args.log_path,
                    log_file      = args.log_file,
                    learning_rate = args.learning_rate,
                    checkpoints   = os.path.join(args.checkpoints_path,args.checkpoints_file))

    tester.load_checkpoint(args.start_epoch)

    tester.model_test()

if __name__ == '__main__':
    main()
