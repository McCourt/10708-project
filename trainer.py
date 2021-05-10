import argparse
from ast import parse
import importlib
from collections import defaultdict
import os

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms, datasets
from torch.distributions.normal import Normal
from tqdm import tqdm
from matplotlib import pyplot as plt
import wandb

seed = 1
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# use 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'
# 8, 9, 11, 20, 39
attr_names = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
    'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
    'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
    'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
    'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
    'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
    'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
    'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]
attr_selector = [8, 9, 11, 20, 39]


def plot(imgs, rec_imgs, model, model_dir, expID=None, epoch=None, idx=None):
    f, axs = plt.subplots(2, 10, figsize=(20, 4))
    axs = axs.flatten()
    for i, (img, rec_img) in enumerate(zip(imgs, rec_imgs)):
        axs[i].imshow(np.moveaxis(img, 0, 2))
        axs[i].axis('off')
        axs[i + 10].imshow(np.moveaxis(rec_img, 0, 2))
        axs[i + 10].axis('off')
    if expID is None or idx is None or epoch is None:
        plt.savefig(os.path.join(model_dir, 'vis_{}.png'.format(model)))
    else:
        plt.savefig(os.path.join(model_dir, 'vis_{}_{}_{}_{}.png'.format(model, expID, epoch, idx)))
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


def pretrain_encoder(model, train_loader, lr=1e-3, num_epoch=1):
    print('pretaining encoder for BV-GAN entropy loss')
    try:
        sd = torch.load('ckpt/pretrain_ae.pt', map_location='cpu')
        model.load_state_dict(sd)
        print('Loading model successful.')
        return model, []
    except:
        print('Start new training.')
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    pbar = tqdm(total=num_epoch * len(train_loader))
    loss_dic = defaultdict(list)
    for epoch in range(num_epoch):
        for index, (imgs, features) in enumerate(train_loader):
            features = features[:, attr_selector]
            data_dic = {'images': imgs.to(device), 'labels': features.float().to(device), 'fake_labels': torch.randint(0, 1, features.size()).float().to(device)}

            optimizer.zero_grad()
            rec = model(data_dic['images'], data_dic['labels'])
            loss = loss_fn(rec, data_dic['images'])
            loss.backward()
            optimizer.step()
            loss_dic['pretain_ae'].append(loss.data.item())

            pbar.set_description('[{}]'.format('pretrain_ae') + ''.join(['[{}:{:.4e}]'.format(k, np.mean(v[-1000:])) for k, v in loss_dic.items()]))
            pbar.update(1)
        torch.save(model.state_dict(), 'ckpt/pretrain_ae.pt')
    pbar.close()
    return model, loss_dic['pretain_ae']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', type=str, default='cgan', help='File Name for Model')
    parser.add_argument('--expID', type=str, default='0000', help='Name of exp')
    parser.add_argument('--num_epoch', type=int, default=100, help='Number of Epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch Size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning Rate')
    parser.add_argument('--g_lr', type=float, default=1e-3, help='Generator Learning Rate')
    parser.add_argument('--d_lr', type=float, default=1e-4, help='Discriminator Learning Rate')
    parser.add_argument('--reg', type=float, default=1, help='weights for reg term in loss')
    parser.add_argument('--gp', type=float, default=10, help='weights for gradient penalty for wgan')
    parser.add_argument('--lambda_c', type=float, default=0.3, help='weights for classification loss')
    parser.add_argument('--lambda_d', type=float, default=0.3, help='weights for discriminator loss')
    parser.add_argument('--lambda_kl', type=float, default=0.01, help='weights for vae')
    parser.add_argument('--lambda_ent', type=float, default=0.01, help='weights for entropy loss')
    parser.add_argument('--n_critics', type=float, default=5, help='every n_critics we update generator of wgan')
    parser.add_argument('--vis_every', type=int, default=50, help='every vis_every we visualize the training results')
    args = parser.parse_args()

    wandb.init(project='bvgan', entity='s21_10708_team2', config=vars(args),
               name=f'exp_{args.expID}')

    transform_fn = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(178), transforms.Resize(64)])
    train_data = datasets.CelebA('./data', split='train', download=True, transform=transform_fn)
    # train_data = torch.utils.data.Subset(train_data, [i for i in range(100)])
    # test_data = datasets.CelebA('./data', split='test', download=True, transform=transform_fn)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=4, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = importlib.import_module('models.{}'.format(args.model))

    if args.expID is None:
        model_dir = '.'
    else:
        model_dir = 'exp_{}'.format(args.expID)
        if os.path.exists(model_dir):
            print('[WARNING] exp folder exists!!! overwriting expected!!!')
        else:
            os.makedirs(model_dir)

    if args.model == 'cgan':
        discriminator = module.DiscriminatorModel()
        generator = module.GeneratorModel()
        try:
            g, d = torch.load('ckpt/cgan.pt', map_location='cpu')
            generator.load_state_dict(g)
            discriminator.load_state_dict(d)
            print('Loading model successful.')
        except:
            print('Start new training.')
        discriminator.to(device)
        generator.to(device)

        loss_fn = nn.BCELoss()
        reg_fn = nn.L1Loss()
        doptimizer = optim.Adam(discriminator.parameters(), lr=args.learning_rate)
        goptimizer = optim.Adam(generator.parameters(), lr=args.learning_rate)
    elif args.model == 'cvae':
        model = module.GeneratorModel(device)
        classifier = module.ClassifierModel()
        model.to(device)
        classifier.to(device)
        bce_fn = nn.BCELoss()
        loss_fn = lambda x, x_hat, mu, logvar: bce_fn(x_hat, x) - 0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        coptimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    elif args.model == 'bvgan':
        discriminator = module.DiscriminatorModel()
        generator = module.GeneratorModel()
        discriminator.to(device)
        generator.to(device)
        loss_fn = nn.BCELoss()
        kl_loss_fn = lambda mu, std: -0.5 * (1 + torch.log(std.pow(2)) - mu.pow(2) - std.pow(2)).mean()
        norm = Normal(torch.tensor(0.0), torch.tensor(1.0))
        ent_loss_fn = lambda z: -torch.var(z)
        d_optim = optim.Adam(discriminator.parameters(), lr=args.d_lr)
        g_optim = optim.Adam(generator.parameters(), lr=args.g_lr)
        pretrained_encoder, pretrain_loss = pretrain_encoder(module.AutoEncoder(), train_loader)
        print('pretrain finished')

    pbar = tqdm(total=args.num_epoch * len(train_loader))
    loss_dic = defaultdict(list)
    step = 0
    for epoch in range(args.num_epoch):
        for index, (imgs, features) in enumerate(train_loader):
            if args.model == 'bvgan':
                features = features[:, attr_selector]
            edit_indices = np.random.randint(0, features.shape[-1], size=features.shape[0])
            fake_labels = features.detach().cpu().numpy()
            fake_labels[np.arange(features.shape[0]), edit_indices] = 1 - fake_labels[np.arange(features.shape[0]), edit_indices]
            data_dic = {
                'images': imgs.to(device),
                'labels': features.float().to(device),
                'fake_labels': torch.Tensor(fake_labels).float().to(device)
            }
            if args.model == 'cgan':
                doptimizer.zero_grad()
                g = generator(data_dic['images'], data_dic['fake_labels'])
                import pdb; pdb.set_trace()
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

                loss = dloss + d_loss_gp * args.gp + args.lambda_c * closs
                loss.backward()
                doptimizer.step()
                loss_dic['dloss'].append(dloss.data.item())
                loss_dic['closs'].append(closs.data.item())

                if index % args.n_critics == 0:
                    goptimizer.zero_grad()
                    g = generator(data_dic['images'], data_dic['fake_labels']) # batch_size X 784
                    df, cff = discriminator(g)

                    dloss = - torch.mean(df)
                    closs = loss_fn(cff, data_dic['fake_labels'])
                    reg = reg_fn(g, data_dic['images'])
                    loss = dloss + args.lambda_c * closs + args.reg * reg
                    loss.backward()
                    goptimizer.step()

                    loss_dic['gdloss'].append(dlf.data.item())
                    loss_dic['gcloss'].append(closs.data.item())
                    loss_dic['greg'].append(reg.data.item())

                if index % args.vis_every == 0:
                    sample_input = data_dic['images'].detach().cpu().numpy()[:10]
                    sample_pred = g.detach().cpu().numpy()[:10]
                    plot(sample_input, sample_pred, args.model, model_dir, args.expID, epoch, index)

                    vis_images = np.concatenate([
                        np.concatenate([sample_input[i] for i in range(10)], axis=1),
                        np.concatenate([sample_pred[i] for i in range(10)], axis=1)
                    ], axis=0)
                    wandb.log({'sample_rec', [wandb.Image(vis_images)]}, step=step)
            elif args.model == 'cvae':
                coptimizer.zero_grad()
                cfr = classifier(data_dic['images'])
                closs = bce_fn(cfr, data_dic['labels'])
                closs.backward()
                coptimizer.step()
                loss_dic['closs'].append(closs.data.item())

                optimizer.zero_grad()
                x_hat, mu, logvar = model(data_dic['images'], data_dic['fake_labels'])
                vae_loss = loss_fn(data_dic['images'], x_hat, mu, logvar)
                cfr = classifier(x_hat)
                closs = bce_fn(cfr, data_dic['fake_labels'])
                loss = vae_loss + args.lambda_c * closs
                loss.backward()
                optimizer.step()
                loss_dic['cvae_loss'].append(loss.data.item())

                if index % args.vis_every == 0:
                    model.eval()
                    plot(data_dic['images'].detach().cpu().numpy()[:10], x_hat.detach().cpu().numpy()[:10], args.model, model_dir, args.expID, epoch, index)
                    
                    original_x = torch.cat([data_dic['images'][:1]] * 10, dim=0)
                    fake_x_labels = data_dic['fake_labels'][:10]
                    fake_x, _, _ = model(original_x, fake_x_labels)
                    plot(original_x.detach().cpu().numpy(), fake_x.detach().cpu().numpy(), args.model, model_dir, args.expID + '_fake', epoch, index)
                    
                    model.train()
            
            elif args.model == 'bvgan':
                loss_dic = {}

                # train discriminator
                if index % args.n_critics == 0:
                    g, _, _, _ = generator(data_dic['images'], data_dic['fake_labels'])
                    d_real_output, c_real_output = discriminator(data_dic['images'])
                    d_fake_output, c_fake_output = discriminator(g)

                    d_loss_real = -torch.log(d_real_output).mean()
                    d_loss_fake = - torch.log(1 - d_fake_output).mean()
                    c_loss = loss_fn(c_real_output, data_dic['labels'])
                    d_total_loss = args.lambda_d * d_loss_real + d_loss_fake + args.lambda_c * c_loss

                    d_optim.zero_grad()
                    d_total_loss.backward()
                    d_optim.step()

                    loss_dic['dis_d_loss'] = (d_loss_real + d_loss_fake).data.item()
                    loss_dic['dis_c_loss'] = c_loss.data.item()

                # train generator
                batch_size = data_dic['images'].shape[0]
                x_hat_fake, mu_fake, std_fake, x_hats_fake = generator(data_dic['images'], data_dic['fake_labels'], discriminator)
                x_hat_real, mu_real, std_real, x_hats_real = generator(data_dic['images'], data_dic['labels'], discriminator)
                d_output_1, c_output_1 = discriminator(x_hat_fake)
                d_output_2, c_output_2 = discriminator(x_hat_real)
                d_outputs = torch.cat([d_output_1, d_output_2], dim=0)
                vae_loss = (kl_loss_fn(mu_fake, std_fake) + kl_loss_fn(mu_real, std_real)) / 2
                d_loss = -torch.log(d_outputs).mean()
                c_loss = loss_fn(c_output_1[np.arange(batch_size), edit_indices],
                                 data_dic['fake_labels'][np.arange(batch_size), edit_indices]) / 2
                c_loss += loss_fn(c_output_2[np.arange(batch_size), edit_indices],
                                  data_dic['labels'][np.arange(batch_size), edit_indices]) / 2
                rec_loss = torch.mean((x_hat_real - data_dic['images']) ** 2)
                z_fake = pretrained_encoder.encoder(x_hats_fake)
                z_real = pretrained_encoder.encoder(x_hats_real)
                ent_loss = (ent_loss_fn(z_fake).sum() + ent_loss_fn(z_real).sum()) / 2
                g_total_loss = args.lambda_d * d_loss + args.lambda_c * c_loss + args.reg * rec_loss + args.lambda_kl * vae_loss + args.lambda_ent * ent_loss

                g_optim.zero_grad()
                g_total_loss.backward()
                g_optim.step()

                loss_dic.update({
                    'gen_d_loss': d_loss.data.item(),
                    'gen_c_loss': c_loss.data.item(),
                    'gen_kl_loss': vae_loss.data.item(),
                    'gen_rec_loss': rec_loss.data.item(),
                    'gen_ent_loss': ent_loss.data.item()
                })

                if index % args.vis_every == 0:
                    generator.eval()
                    vis_x_hat_fake, _, _, _ = generator(data_dic['images'], data_dic['fake_labels'], discriminator)
                    vis_x_hat_real, _, _, _ = generator(data_dic['images'], data_dic['labels'], discriminator)
                    num_samples = 10
                    gt_samples = data_dic['images'].detach().cpu().numpy()[:num_samples].transpose(0, 2, 3, 1)
                    rec_samples = vis_x_hat_real.detach().cpu().numpy()[:num_samples].transpose(0, 2, 3, 1)
                    edit_samples = vis_x_hat_fake.detach().cpu().numpy()[:num_samples].transpose(0, 2, 3, 1)
                    for i in range(num_samples):
                        edit_sample = (edit_samples[i] * 255).copy().astype(np.uint8)
                        annotated_img = cv2.putText(edit_sample,
                                                    attr_names[edit_indices[i]],
                                                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                                    (255, 255, 255), 1,
                                                    cv2.LINE_AA)
                        edit_samples[i] = annotated_img / 255.0
                    vis_images = np.concatenate([
                        np.concatenate([gt_samples[i] for i in range(num_samples)], axis=1),
                        np.concatenate([rec_samples[i] for i in range(num_samples)], axis=1),
                        np.concatenate([edit_samples[i] for i in range(num_samples)], axis=1)
                    ], axis=0)
                    wandb.log({'sample_images': [wandb.Image(vis_images)]}, step=step)
                    generator.train()

            pbar.set_description(f'training: vae loss - {vae_loss.data.item()}')
            pbar.update(1)

            wandb.log(loss_dic, step=step)
            step += 1

        if args.model == 'cgan':
            torch.save([generator.state_dict(), discriminator.state_dict()], 'ckpt/cgan.pt')
        elif args.model == 'cvae':
            torch.save([model.state_dict(), classifier.state_dict()], os.path.join(model_dir, 'ckpt/cvae_{}_{}.pt'.format(args.expID, args.num_epoch * epoch + index)))
    pbar.close()
    pd.DataFrame.from_dict(loss_dic).to_csv('{}_log.csv'.format(args.model))
