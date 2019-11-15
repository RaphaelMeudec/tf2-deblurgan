import os
import datetime
from functools import partial

import click
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam

from deblurgan.datasets import IndependantDataLoader
from deblurgan.losses import wasserstein_loss, perceptual_loss
from deblurgan.model import (
    generator_model,
    discriminator_model,
    generator_containing_discriminator_multiple_outputs,
)


BASE_DIR = "weights/"


def save_all_weights(d, g, epoch_number, current_loss):
    now = datetime.datetime.now()
    save_dir = os.path.join(BASE_DIR, "{}{}".format(now.month, now.day))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    g.save_weights(
        os.path.join(save_dir, "generator_{}_{}.h5".format(epoch_number, current_loss)),
        True,
    )
    d.save_weights(
        os.path.join(save_dir, "discriminator_{}.h5".format(epoch_number)), True
    )


def train(batch_size, log_dir, epoch_num, critic_updates=5):
    patch_size = (256, 256)
    dataset, dataset_length = IndependantDataLoader().load(
        "gopro",
        mode="train",
        batch_size=batch_size,
        patch_size=patch_size,
        shuffle=True,
    )

    g = generator_model()
    d = discriminator_model()
    d_on_g = generator_containing_discriminator_multiple_outputs(g, d)

    d_opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    d_on_g_opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    d.trainable = True
    d.compile(optimizer=d_opt, loss=wasserstein_loss)
    d.trainable = False
    loss = [partial(perceptual_loss, input_shape=(*patch_size, 3)), wasserstein_loss]
    loss_weights = [100, 1]
    d_on_g.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights)
    d.trainable = True

    output_true_batch, output_false_batch = (
        np.ones((batch_size, 1)),
        -np.ones((batch_size, 1)),
    )

    log_path = "./logs"

    for epoch_index in range(epoch_num):
        progbar = tf.keras.utils.Progbar(dataset_length)

        for step_index, (image_full_batch, image_blur_batch) in enumerate(dataset):
            if step_index > dataset_length:
                break

            # TODO: Convert to GradientTape loops
            generated_images = g.predict(x=image_blur_batch, batch_size=batch_size)

            for _ in range(critic_updates):
                d_loss_real = d.train_on_batch(image_full_batch, output_true_batch)
                d_loss_fake = d.train_on_batch(generated_images, output_false_batch)
                d_loss = 0.5 * tf.math.add(d_loss_fake, d_loss_real)

            d.trainable = False

            d_on_g_loss = d_on_g.train_on_batch(
                image_blur_batch, [image_full_batch, output_true_batch]
            )

            d.trainable = True

            # TODO: PSNR metric
            # TODO: SSIM metric

            # TODO: Tensorboard callback
            # TODO: Checkpoint callback

            progbar.update(
                step_index,
                values=[("d_loss", tf.math.reduce_mean(d_loss)), ("g_loss", tf.math.reduce_mean(d_on_g_loss))]
            )


@click.command()
@click.option("--batch_size", default=16, help="Size of batch")
@click.option("--log_dir", required=True, help="Path to the log_dir for Tensorboard")
@click.option("--epoch_num", default=4, help="Number of epochs for training")
@click.option("--critic_updates", default=5, help="Number of discriminator training")
def train_command(batch_size, log_dir, epoch_num, critic_updates):
    return train(batch_size, log_dir, epoch_num, critic_updates)


if __name__ == "__main__":
    train_command()
