import argparse
import os
from utils import get_args

import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO 1.3: Implement GAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ##################################################################
    discrim_fake = discrim_fake.view(-1)
    discrim_real = discrim_real.view(-1)

    real_target = torch.ones((discrim_real.shape[0])).cuda()
    fake_target = torch.zeros((discrim_fake.shape[0])).cuda()

    loss_real = F.binary_cross_entropy_with_logits(discrim_real, real_target)
    loss_fake = F.binary_cross_entropy_with_logits(discrim_fake, fake_target)

    loss = loss_real+loss_fake
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO 1.3: Implement GAN loss for the generator.
    ##################################################################
    discrim_fake = discrim_fake.view(-1)
    target = torch.ones((discrim_fake.shape[0])).cuda()
    loss = F.binary_cross_entropy_with_logits(discrim_fake, target)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
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
