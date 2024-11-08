import os

import cv2
import torch
import torchvision
from torch import nn, optim
from torchvision.utils import make_grid,save_image
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import DEVICE, Z_DIM, LAMBDA_GP, W_DIM, IN_CHANNELS, LEARNING_RATE, ROOT, PROGRESSIVE_EPOCHS
from dataset import getloader
from model import Generator, Discriminator
from util import gradient_penalty


def train(
        D,
        G,
        loader,
        step,
        alpha,
        opt_D,
        opt_G,
        dataset
):
    loop = tqdm(loader, leave=True)
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)
        batch_size = real.size(0)

        noise = torch.randn(batch_size, Z_DIM).to(DEVICE)
        fake = G(noise, alpha, step)
        D_real = D(real, alpha, step)
        D_fake = D(fake.detach(), alpha, step)
        gp = gradient_penalty(D, real, fake, alpha, step, device=DEVICE)
        loss_D = (
                -(torch.mean(D_real) - torch.mean(D_fake))
                + LAMBDA_GP * gp
                + (0.001 * torch.mean(D_real ** 2))
        )
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        opt_G.zero_grad()
        G_fake = D(fake, alpha, step)
        loss_G = -(torch.mean(G_fake))
        loss_G.backward()
        opt_G.step()

        # Update alpha and ensure less than 1
        alpha += batch_size / (
                (PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset)
        )
        alpha = min(alpha, 1)

        loop.set_postfix(
            gp=gp.item(),
            lossD=loss_D.item(),
        )

        return alpha


if __name__ == '__main__':
    G = Generator(Z_DIM, W_DIM, IN_CHANNELS).to(DEVICE)
    D = Discriminator(IN_CHANNELS).to(DEVICE)

    opt_G = optim.Adam([{"params": [param for name, param in G.named_parameters() if "map" not in name]},
                        {"params": G.map.parameters(), "lr": 1e-5}], lr=LEARNING_RATE, betas=(0.0, 0.99))
    opt_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    step = 6
    num_epochs = 50

    G.train()
    D.train()
    alpha = 1e-5
    dataset, loader = getloader(ROOT, image_size=256)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        alpha = train(
            D,
            G,
            loader,
            step,
            alpha,
            opt_D,
            opt_G,
            dataset
        )
        if epoch + 1 == num_epochs:
            if not os.path.exists(f'results'):
                os.makedirs(f'results')
            noise = torch.randn(size=(32, Z_DIM)).to(DEVICE)
            fake = G(noise, alpha, step)
            grid_images = make_grid(fake.detach().cpu(), nrow=8)
            save_image(grid_images, 'results/fake.png')

            torch.save(G.state_dict(), f'results/G.pth')
            torch.save(D.state_dict(), f'results/D.pth')



