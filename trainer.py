import argparse
import torch
import torch.nn as nn
from torch import optim
import torchvision
from torchvision import transforms, datasets
import pandas as pd
import numpy as np
import importlib
from tqdm import tqdm
from collections import defaultdict


transform_fn = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(178)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', type=str, help='File Name for Model')
    parser.add_argument('--num_epoch', type=int, default=100, help='Number of Epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--learning_rate', type=float, default=1e04, help='Learning Rate')
    args = parser.parse_args()

    train_data = datasets.CelebA('./data', split='train', download=True, transform=transform_fn)
    test_data = datasets.CelebA('./data', split='test', download=True, transform=transform_fn)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = importlib.import_module('model.{}'.format(args.model))

    if args.model == 'cgan':
        discriminator = model.DiscriminatorModel()
        generator = model.GeneratorModel()
        discriminator.to(device)
        generator.to(device)

        loss_fn = nn.BCELoss()
        doptimizer = optim.Adam(discriminator.parameters(), lr=args.learning_rate)
        goptimizer = optim.Adam(generator.parameters(), lr=args.learning_rate)
    elif args.model == 'cvae':
        # TODO: Add model specification and optimizer of CVAE
        pass
    elif args.model == 'bvgan':
        # TODO: Add model specification and optimizer of BVGAN
        pass

    pbar = tqdm(total=args.num_epoch * len(train_loader))
    loss_dic = defaultdict(list)
    for _ in range(args.num_epoch):
        for imgs, features in train_loader:
            data_dic = {'images': imgs.to(device), 'labels': features.to(device)}
            if args.model == 'cgan':
                data_dic['noises']: torch.randn(imgs.size()).to(device)
                data_dic['fake_labels']: torch.randint(0, 1, features.size()).to(device)

                doptimizer.zero_grad()
                with torch.no_grad():
                    g = generator(data_dic['noises'], data_dic['fake_labels'])
                dr = discriminator(data_dic['images'], data_dic['labels'])
                df = discriminator(g, data_dic['fake_labels'])
                dloss = loss_fn(dr, torch.ones(dr.size())) + loss_fn(df, torch.zeros(df.size()))
                dloss.backward()
                doptimizer.step()
                loss_dic['dloss'].append(dloss.data.item())

                goptimizer.zero_grad()
                g = generator(data_dic['noises'], data_dic['fake_labels']) # batch_size X 784
                df = discriminator(g, data_dic['fake_labels'])
                gloss = loss_fn(df, torch.zeros(df.size()))
                gloss.backward()
                goptimizer.step()
                loss_dic['gloss'].append(gloss.data.item())
            elif args.model == 'cvae':
                # TODO: Add training loop of CVAE
                pass
            elif args.model == 'bvgan':
                # TODO: Add training loop of BVGAN
                pass
            pbar.set_description(''.join(['[{}:{:.4f}]'.format(k, np.mean(v[-1000:])) for k, v in loss_dic.items()]))
            pbar.update(1)
    pbar.close()
