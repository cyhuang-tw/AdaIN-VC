import argparse
import os

import torch
import torch.nn as nn
import yaml
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange

from data import InfiniteDataLoader, SpeakerDataset, infinite_iterator
from model import AdaINVC


def main(
    config_file: str,
    data_dir: str,
    save_dir: str,
    n_steps: int,
    save_steps: int,
    log_steps: int,
    n_spks: int,
    n_uttrs: int,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # Load config
    config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)

    # Prepare data
    data = SpeakerDataset(data_dir, segment=128, n_uttrs=n_uttrs)

    # split train/valid sets
    train_set, valid_set = random_split(
        data, [int(len(data) * 0.8), len(data) - int(len(data) * 0.8)]
    )

    # construct loader
    train_loader = InfiniteDataLoader(
        train_set, batch_size=n_spks, shuffle=True, num_workers=8
    )
    valid_loader = InfiniteDataLoader(
        valid_set, batch_size=n_spks, shuffle=True, num_workers=8
    )

    # construct iterator
    train_iter = infinite_iterator(train_loader)
    valid_iter = infinite_iterator(valid_loader)

    # Build model
    model = AdaINVC(config["Model"]).to(device)
    model = torch.jit.script(model)

    # Optimizer
    opt = torch.optim.Adam(
        model.parameters(),
        lr=config["Optimizer"]["lr"],
        betas=(config["Optimizer"]["beta1"], config["Optimizer"]["beta2"]),
        amsgrad=config["Optimizer"]["amsgrad"],
        weight_decay=config["Optimizer"]["weight_decay"],
    )

    # Tensorboard logger
    writer = SummaryWriter(save_dir)
    criterion = nn.L1Loss()
    pbar = trange(n_steps, ncols=0)
    valid_steps = 32

    for step in pbar:
        # get features
        org_mels = next(train_iter)
        org_mels = org_mels.flatten(0, 1)
        org_mels = org_mels.to(device)

        # reconstruction
        mu, log_sigma, emb, rec_mels = model(org_mels)

        # compute loss
        rec_loss = criterion(rec_mels, org_mels)
        kl_loss = 0.5 * (log_sigma.exp() + mu ** 2 - 1 - log_sigma).mean()
        rec_lambda = config["Lambda"]["rec"]
        kl_lambda = min(
            config["Lambda"]["kl"] * step / config["Lambda"]["kl_annealing"],
            config["Lambda"]["kl"],
        )
        loss = rec_lambda * rec_loss + kl_lambda * kl_loss

        # update parameters
        opt.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        opt.step()

        # save model and optimizer
        if (step + 1) % save_steps == 0:
            model_path = os.path.join(save_dir, f"model-{step + 1}.ckpt")
            model.cpu()
            model.save(model_path)
            model.to(device)
            opt_path = os.path.join(save_dir, f"opt-{step + 1}.ckpt")
            torch.save(opt.state_dict(), opt_path)

        if (step + 1) % log_steps == 0:
            # validation
            model.eval()
            valid_loss = 0
            for _ in range(valid_steps):
                org_mels = next(valid_iter)
                org_mels = org_mels.flatten(0, 1)
                org_mels = org_mels.to(device)
                mu, log_sigma, emb, rec_mels = model(org_mels)
                loss = criterion(rec_mels, org_mels)
                valid_loss += loss.item()
            valid_loss /= valid_steps
            model.train()

            # record information
            writer.add_scalar("training/rec_loss", rec_loss, step + 1)
            writer.add_scalar("training/kl_loss", kl_loss, step + 1)
            writer.add_scalar("training/grad_norm", grad_norm, step + 1)
            writer.add_scalar("lambda/kl", kl_lambda, step + 1)
            writer.add_scalar("validation/rec_loss", valid_loss, step + 1)

        # update tqdm bar
        pbar.set_postfix({"rec_loss": rec_loss.item(), "kl_loss": kl_loss.item()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("data_dir", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--n_steps", type=int, default=int(1e6))
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--log_steps", type=int, default=250)
    parser.add_argument("--n_spks", type=int, default=32)
    parser.add_argument("--n_uttrs", type=int, default=4)
    main(**vars(parser.parse_args()))
