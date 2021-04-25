import argparse
import importlib
from collections import defaultdict
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms, datasets
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', type=str, default='cgan', help='File Name for Model')
    parser.add_argument('--num_epoch', type=int, default=100, help='Number of Epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch Size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning Rate')
    args = parser.parse_args()

    transform_fn = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(178), transforms.Resize(64)])
    train_data = datasets.CelebA('./data', split='train', download=True, transform=transform_fn)
    # test_data = datasets.CelebA('./data', split='test', download=True, transform=transform_fn)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=4, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = importlib.import_module('models.{}'.format(args.model))

    if args.model == 'cgan':
        discriminator = module.DiscriminatorModel()
        generator = module.GeneratorModel()
        try:
            g, d = torch.load('cgan.pt', map_location='cpu')
            generator.load_state_dict(g)
            discriminator.load_state_dict(d)
            print('Loading model successful.')
        except:
            print('Start new training.')
        discriminator.to(device)
        generator.to(device)

        loss_fn = nn.BCELoss()
        doptimizer = optim.Adam(discriminator.parameters(), lr=args.learning_rate)
        goptimizer = optim.Adam(generator.parameters(), lr=args.learning_rate)
    elif args.model == 'cvae':
        model = module.CVAE()
        model.to(device)
        bce_fn = nn.BCELoss()
        loss_fn = lambda y, y_hat, mu, logvar: bce_fn(y_hat, y) - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.model == 'bvgan':
        # TODO: Add model specification and optimizer of BVGAN
        pass

    pbar = tqdm(total=args.num_epoch * len(train_loader))
    loss_dic = defaultdict(list)
    for _ in range(args.num_epoch):
        for imgs, features in train_loader:
            data_dic = {'images': imgs.to(device), 'labels': (features.to(device) + 1) / 2}
            if args.model == 'cgan':
                # data_dic['noises'] = torch.randn(imgs.size()).to(device)
                data_dic['fake_labels'] = torch.randint(0, 1, features.size()).to(device)

                doptimizer.zero_grad()
                with torch.no_grad():
                    g = generator(data_dic['images'], data_dic['fake_labels'])
                dr = discriminator(data_dic['images'], data_dic['labels'])
                df = discriminator(g, data_dic['fake_labels'])
                r = torch.cat([data_dic['labels'], torch.ones((features.size(0), 1)).to(device)], dim=1)
                f = torch.cat([1. - data_dic['fake_labels'], torch.zeros((features.size(0), 1)).to(device)], dim=1)
                dloss = loss_fn(dr, r.float()) + loss_fn(df, f.float())
                dloss.backward()
                doptimizer.step()
                loss_dic['cgan_dloss'].append(dloss.data.item())

                goptimizer.zero_grad()
                g = generator(data_dic['images'], data_dic['fake_labels']) # batch_size X 784
                df = discriminator(g, data_dic['fake_labels'])
                f = torch.cat([data_dic['fake_labels'], torch.ones((features.size(0), 1)).to(device)], dim=1)
                gloss = loss_fn(df, f.float())
                gloss.backward()
                goptimizer.step()
                loss_dic['cgan_gloss'].append(gloss.data.item())
            elif args.model == 'cvae':
                optimizer.zero_grad()
                y_hat, mu, logvar = model(data_dic['images'])
                loss = loss_fn(data_dic['images'], y_hat, mu, logvar)
                loss.backward()
                optimizer.step()
                loss_dic['cvae_loss'].append(loss.data.item())
            elif args.model == 'bvgan':
                # TODO: Add training loop of BVGAN
                pass
            pbar.set_description(''.join(['[{}:{:.4f}]'.format(k, np.mean(v[-1000:])) for k, v in loss_dic.items()]))
            pbar.update(1)
        if args.model == 'cgan':
            torch.save([generator.state_dict(), discriminator.state_dict()], 'cgan.pt')
    pbar.close()
    pd.DataFrame.from_dict(loss_dic).to_csv('{}_log.csv'.format(args.model))
