import os

import yaml
import torch
import torch.nn as nn

from model import AE
from data_utils import get_data_loader
from data_utils import PickleDataset
from utils import cc, Logger, infinite_iter

class Solver(object):
    def __init__(self, config, args):
        # config store the value of hyperparameters
        self.config = config

        # args store other information
        self.log_dir = args.log_dir
        self.save_path = args.save_path
        self.save_steps = args.save_steps
        self.summary_steps = args.summary_steps
        self.tag = args.tag

        # logger to use tensorboard
        self.logger = Logger(self.log_dir)

        # get dataset, data loader, and iter
        train_set_file = os.path.join(args.data_dir, f'{args.train_set}.pkl')
        train_index_file = os.path.join(args.data_dir, args.train_index_file)
        self.train_dataset = PickleDataset(train_set_file, train_index_file,
                                           segment_size=config['data_loader']['segment_size'])
        self.train_loader = get_data_loader(self.train_dataset,
                                            frame_size=config['data_loader']['frame_size'],
                                            batch_size=config['data_loader']['batch_size'],
                                            shuffle=config['data_loader']['shuffle'],
                                            num_workers=4)
        self.train_iter = infinite_iter(self.train_loader)

        # init model and optimizer
        self.model, self.opt = self.build_model()

        # load model and optimizer from checkpoint
        if args.load_model and args.load_path is not None:
            self.load_model(args.load_path)

        # save the config and args
        os.makedirs(self.save_path, exist_ok=True)
        with open(os.path.join(self.save_path, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)

        with open(os.path.join(self.save_path, 'args.yaml'), 'w') as f:
            yaml.dump(vars(args), f)

    def build_model(self):
        model = cc(AE(self.config))
        opt = self.build_optimizer(model)
        return model, opt

    def build_optimizer(self, model):
        opt_config = self.config['optimizer']
        opt = torch.optim.Adam(model.parameters(),
                               lr=opt_config['lr'],
                               betas=(opt_config['beta1'], opt_config['beta2']),
                               amsgrad=opt_config['amsgrad'],
                               weight_decay=opt_config['weight_decay'])
        return opt

    def load_model(self, path):
        model_path = os.path.join(path, 'model.ckpt')
        opt_path = os.path.join(path, 'optimizer.ckpt')
        self.model.load_state_dict(torch.load(model_path))
        self.opt.load_state_dict(torch.load(opt_path))

    def save_model(self):
        model_path = os.path.join(self.save_path, 'model.ckpt')
        opt_path = os.path.join(self.save_path, 'optimizer.ckpt')
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.opt.state_dict(), opt_path)

    def get_lambda(self, name, iteration):
        val = self.config['lambda'][f'lambda_{name}']
        if iteration < self.config['lambda'][f'{name}_annealing']:
            val *= iteration / self.config['lambda'][f'{name}_annealing']
        return val

    def train_step(self, data, iteration):
        # to cuda
        x = cc(data)

        # record important information
        loss_info = {}
        lambda_info = {}

        # Autoencoder reconstruction
        loss_rec, loss_kl, (mu, log_sigma, emb, dec) = self.ae_step(x)

        lambda_rec = self.config['lambda']['lambda_rec']
        lambda_kl = self.get_lambda('kl', iteration)

        loss_info['loss_rec'] = loss_rec.item()
        loss_info['loss_kl'] = loss_kl.item()
        lambda_info['lambda_kl'] = lambda_kl

        loss = lambda_rec * loss_rec + lambda_kl * loss_kl

        self.opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   max_norm=self.config['optimizer']['grad_norm'])
        loss_info['grad_norm'] = grad_norm.item()
        self.opt.step()

        return loss_info, lambda_info

    def ae_step(self, x):
        mu, log_sigma, emb, dec = self.model(x)
        criterion = nn.L1Loss()
        loss_rec = criterion(dec, x)
        loss_kl = 0.5 * torch.mean(torch.exp(log_sigma) + mu ** 2 - 1 - log_sigma)
        return loss_rec, loss_kl, (mu, log_sigma, emb, dec)

    def print_info(self, iteration, n_iterations, loss_info):
        info = f'AE:[{iteration + 1}/{n_iterations}], '
        for key, val in loss_info.items():
            info = info + f'{key}: {val:.4f}  '
        info = info + '           '
        print(info, end='\r')

    def train(self, n_iterations):
        for iteration in range(n_iterations):
            data = next(self.train_iter)
            loss_info, lambda_info = self.train_step(data, iteration)
            # add to logger
            if (iteration + 1) % self.summary_steps == 0:
                self.logger.scalars_summary(f'{self.tag}/loss', loss_info, iteration + 1)
                self.logger.scalars_summary(f'{self.tag}/lambda', lambda_info, iteration + 1)
            # print information
            self.print_info(iteration, n_iterations, loss_info)
            # save model and optimizer
            if (iteration + 1) % self.save_steps == 0 or iteration + 1 == n_iterations:
                self.save_model()
