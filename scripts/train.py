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


@tf.function
def train_step(image_blur_batch, image_full_batch, g, d, g_opt, d_opt, critic_updates, loss_model):
    for _ in range(critic_updates):
        with tf.GradientTape() as tape:
            predictions_real = d(image_full_batch)
            d_loss_real = wasserstein_loss(tf.ones_like(predictions_real), predictions_real)

            generated_images = g(image_blur_batch)
            predictions_fake = d(generated_images)
            d_loss_fake = wasserstein_loss(-tf.ones_like(predictions_fake), predictions_fake)

            d_loss = 0.5 * tf.math.add(d_loss_real, d_loss_fake)

        gradients = tape.gradient(d_loss, d.trainable_weights)
        d_opt.apply_gradients(zip(gradients, d.trainable_weights))

    with tf.GradientTape() as tape:
        deblurred_images = g(image_blur_batch)
        predictions = d(deblurred_images)

        d_loss = wasserstein_loss(tf.ones_like(predictions), predictions)
        image_loss = perceptual_loss(image_full_batch, deblurred_images, loss_model=loss_model)

        g_loss = 100 * image_loss + d_loss

    gradients = tape.gradient(g_loss, g.trainable_weights)
    g_opt.apply_gradients(zip(gradients, g.trainable_weights))

    return g_loss, d_loss


def train(batch_size, log_dir, epochs, critic_updates=5):
    patch_size = (256, 256)
    dataset, dataset_length = IndependantDataLoader().load(
        "gopro",
        mode="train",
        batch_size=batch_size,
        patch_size=patch_size,
        shuffle=True,
    )

    # steps_per_epoch = dataset_length // batch_size
    steps_per_epoch = 3

    d = discriminator_model()
    d_opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    g = generator_model()
    g_opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    callbacks = [
        TensorBoard(log_dir=log_dir),
    ]

    [callback.set_model(g) for callback in callbacks]
    [callback.set_params({
        'epochs': epochs,
        'steps': steps_per_epoch,
        'verbose': True,
        'do_validation': False,
    }) for callback in callbacks]

    [callback.on_train_begin() for callback in callbacks]

    vgg = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(*patch_size, 3))
    loss_model = tf.keras.Model(inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output)
    loss_model.trainable = False

    for epoch_index in range(epochs):
        [callback.on_epoch_begin(epoch_index) for callback in callbacks]

        progbar = tf.keras.utils.Progbar(steps_per_epoch)

        for step_index, (image_full_batch, image_blur_batch) in enumerate(dataset):
            [callback.on_batch_begin(step_index) for callback in callbacks]

            if step_index > steps_per_epoch:
                break

            g_loss, d_loss = train_step(
                image_blur_batch=image_blur_batch,
                image_full_batch=image_full_batch,
                g=g,
                d=d,
                g_opt=g_opt,
                d_opt=d_opt,
                critic_updates=critic_updates,
                loss_model=loss_model,
            )

            # TODO: PSNR metric
            # TODO: SSIM metric

            # TODO: Tensorboard callback
            # TODO: Checkpoint callback

            progbar.update(
                step_index,
                values=[("d_loss", tf.math.reduce_mean(d_loss)), ("g_loss", tf.math.reduce_mean(g_loss))]
            )

            [callback.on_batch_end(step_index) for callback in callbacks]
        [callback.on_epoch_end(epoch_index) for callback in callbacks]
    [callback.on_train_end() for callback in callbacks]


@click.command()
@click.option("--batch_size", default=16, help="Size of batch")
@click.option("--log_dir", required=True, help="Path to the log_dir for Tensorboard")
@click.option("--epochs", default=4, help="Number of epochs for training")
@click.option("--critic_updates", default=5, help="Number of discriminator training")
def train_command(batch_size, log_dir, epochs, critic_updates):
    return train(batch_size, log_dir, epochs, critic_updates)


if __name__ == "__main__":
    train_command()
