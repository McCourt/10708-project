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
from matplotlib import pyplot as plt


def plot(imgs, rec_imgs, model):
    f, axs = plt.subplots(2, 10, figsize=(20, 4))
    axs = axs.flatten()
    for i, (img, rec_img) in enumerate(zip(imgs, rec_imgs)):
        axs[i].imshow(np.moveaxis(img, 0, 2))
        axs[i].axis('off')
        axs[i + 10].imshow(np.moveaxis(rec_img, 0, 2))
        axs[i + 10].axis('off')
    plt.savefig('vis_{}.png'.format(model))
    plt.close()


def gradient_penalty(y, x, device):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', type=str, default='cgan', help='File Name for Model')
    parser.add_argument('--num_epoch', type=int, default=100, help='Number of Epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch Size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning Rate')
    args = parser.parse_args()

    transform_fn = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(178), transforms.Resize(64)])
    train_data = datasets.CelebA('./data', split='train', download=True, transform=transform_fn)
    # train_data = torch.utils.data.Subset(train_data, [i for i in range(10)])
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
        reg_fn = nn.MSELoss()
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
        for index, (imgs, features) in enumerate(train_loader):
            data_dic = {'images': imgs.to(device), 'labels': (features.to(device) + 1) / 2}
            if args.model == 'cgan':
                data_dic['fake_labels'] = torch.randint(0, 1, features.size()).float().to(device)

                doptimizer.zero_grad()
                g = generator(data_dic['images'], data_dic['fake_labels'])
                dr, cfr = discriminator(data_dic['images'])
                df, _ = discriminator(g)

                dlr = - torch.mean(dr)
                dlf = torch.mean(df)
                dloss = dlr + dlf
                closs = loss_fn(cfr, data_dic['labels'])

                alpha = torch.rand(data_dic['images'].size(0), 1, 1, 1).to(device)
                x_hat = (alpha * data_dic['images'].data + (1 - alpha) * g.data).requires_grad_(True)
                out_src, _ = discriminator(x_hat)
                d_loss_gp = gradient_penalty(out_src, x_hat, device)

                loss = dloss + closs + d_loss_gp * 10
                loss.backward()
                doptimizer.step()
                loss_dic['cgan_dloss'].append(dloss.data.item())
                loss_dic['cgan_closs'].append(closs.data.item())

                if index % 5 == 0:
                    goptimizer.zero_grad()
                    g = generator(data_dic['images'], data_dic['fake_labels']) # batch_size X 784
                    df, cff = discriminator(g)

                    dloss = - torch.mean(df)
                    closs = loss_fn(cff, data_dic['fake_labels'])
                    reg = reg_fn(g, data_dic['images'])
                    loss = dloss + closs + 10 * reg
                    loss.backward()
                    goptimizer.step()

                    loss_dic['cgan_gdloss'].append(dlf.data.item())
                    loss_dic['cgan_gcloss'].append(closs.data.item())
                    loss_dic['cgan_greg'].append(reg.data.item())
                if index % 50 == 0:
                    plot(data_dic['images'].detach().cpu().numpy()[:10], g.detach().cpu().numpy()[:10], args.model)
            elif args.model == 'cvae':
                optimizer.zero_grad()
                y_hat, mu, logvar = model(data_dic['images'], data_dic['labels'])
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
