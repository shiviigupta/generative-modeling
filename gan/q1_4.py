import os

import torch
import torch.nn.functional as F
from utils import get_args

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO: 1.4: Implement LSGAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ##################################################################
    discrim_fake = discrim_fake.view(-1)
    discrim_real = discrim_real.view(-1)

    real_target = torch.ones((discrim_real.shape[0])).cuda().half()
    fake_target = torch.zeros((discrim_fake.shape[0])).cuda().half()

    loss_real = F.mse_loss(discrim_real, real_target)
    loss_fake = F.mse_loss(discrim_fake, fake_target)
    loss = 0.5*(loss_real + loss_fake)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO: 1.4: Implement LSGAN loss for generator.
    ##################################################################
    discrim_fake = discrim_fake.view(-1)
    target = torch.ones((discrim_fake.shape[0])).cuda()
    loss = F.mse_loss(discrim_fake, target)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss

if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_ls_gan/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
