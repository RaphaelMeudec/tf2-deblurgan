import os
import datetime

import click
import tensorflow as tf
from loguru import logger
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam

from deblurgan.datasets import IndependantDataLoader
from deblurgan.losses import wasserstein_loss, perceptual_loss
from deblurgan.model import (
    generator_model,
    discriminator_model,
)

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


BASE_DIR = "weights/"


@tf.function
def train_step(
    image_blur_batch, image_sharp_batch, g, d, g_opt, d_opt, critic_updates, loss_model
):
    for _ in range(critic_updates):
        with tf.GradientTape() as tape:
            predictions_real = d(image_sharp_batch)
            d_loss_real = wasserstein_loss(
                tf.ones_like(predictions_real), predictions_real
            )

            generated_images = g(image_blur_batch)
            predictions_fake = d(generated_images)
            d_loss_fake = wasserstein_loss(
                -tf.ones_like(predictions_fake), predictions_fake
            )

            d_loss = tf.math.reduce_mean(0.5 * tf.math.add(d_loss_real, d_loss_fake))

        gradients = tape.gradient(d_loss, d.trainable_weights)
        d_opt.apply_gradients(zip(gradients, d.trainable_weights))

    with tf.GradientTape() as tape:
        deblurred_images = g(image_blur_batch)
        predictions = d(deblurred_images)

        d_loss = tf.math.reduce_mean(
            wasserstein_loss(tf.ones_like(predictions), predictions)
        )
        image_loss = perceptual_loss(
            image_sharp_batch, deblurred_images, loss_model=loss_model
        )

        g_loss = tf.math.reduce_mean(100 * image_loss + d_loss)

    gradients = tape.gradient(g_loss, g.trainable_weights)
    g_opt.apply_gradients(zip(gradients, g.trainable_weights))

    return g_loss, d_loss


def evaluate_psnr(model, dataset, evaluation_steps=10):
    return tf.math.reduce_mean(
        [
            tf.image.psnr(sharp, model(blur), max_val=1)
            for blur, sharp in dataset.take(evaluation_steps)
        ]
    )


def train(batch_size, log_dir, epochs, critic_updates=5, restore_checkpoint=None):
    logger.info("Start experiment.")
    patch_size = (256, 256)
    train_dataset, train_dataset_length = IndependantDataLoader().load(
        "gopro",
        mode="train",
        batch_size=batch_size,
        patch_size=patch_size,
        shuffle=True,
    )
    logger.info("Train dataset loaded.")
    validation_dataset, validation_dataset_length = IndependantDataLoader().load(
        "gopro",
        mode="test",
        batch_size=batch_size,
        patch_size=patch_size,
        shuffle=False,
    )
    logger.info("Validation dataset loaded.")
    steps_per_epoch = train_dataset_length // batch_size

    d = discriminator_model()
    d_opt = Adam(lr=3e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    logger.info("Discriminator loaded.")
    g = generator_model()
    g_opt = Adam(lr=3e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    logger.info("Generator loaded.")

    callbacks = [
        TensorBoard(log_dir=log_dir),
    ]

    checkpoint_path = "./checkpoints"

    ckpt = tf.train.Checkpoint(
        generator=g,
        discriminator=d,
        generator_optimizer=g_opt,
        discriminator_optimizer=d_opt,
    )
    if restore_checkpoint is not None:
        ckpt.restore(restore_checkpoint)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    [callback.set_model(g) for callback in callbacks]
    [
        callback.set_params(
            {
                "epochs": epochs,
                "steps": steps_per_epoch,
                "verbose": True,
                "do_validation": False,
            }
        )
        for callback in callbacks
    ]

    [callback.on_train_begin() for callback in callbacks]

    vgg = tf.keras.applications.VGG16(
        include_top=False, weights="imagenet", input_shape=(*patch_size, 3)
    )
    loss_model = tf.keras.Model(
        inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output
    )
    loss_model.trainable = False

    generator_metric = tf.keras.metrics.Mean()
    discriminator_metric = tf.keras.metrics.Mean()

    best_psnr_eval = None

    logger.info("Start training.")
    for epoch_index in range(epochs):
        logger.info(f"Epoch {epoch_index + 1} / {epochs}")
        [callback.on_epoch_begin(epoch_index) for callback in callbacks]

        progbar = tf.keras.utils.Progbar(steps_per_epoch)

        for step_index, (image_blur_batch, image_sharp_batch) in enumerate(
            train_dataset
        ):
            [callback.on_batch_begin(step_index) for callback in callbacks]

            if step_index > steps_per_epoch:
                break

            g_loss, d_loss = train_step(
                image_blur_batch=image_blur_batch,
                image_sharp_batch=image_sharp_batch,
                g=g,
                d=d,
                g_opt=g_opt,
                d_opt=d_opt,
                critic_updates=critic_updates,
                loss_model=loss_model,
            )

            generator_metric(g_loss)
            discriminator_metric(d_loss)

            progbar.update(step_index, values=[("d_loss", d_loss), ("g_loss", g_loss)])

            [callback.on_batch_end(step_index) for callback in callbacks]

        psnr_eval = evaluate_psnr(g, validation_dataset, evaluation_steps=10)

        if best_psnr_eval is None or psnr_eval > best_psnr_eval:
            best_psnr_eval = psnr_eval
            g.save_weights("best_psnr.h5")
            ckpt_manager.save()

        [
            callback.on_epoch_end(
                epoch_index,
                logs={
                    "g_loss": generator_metric.result(),
                    "d_loss": discriminator_metric.result(),
                    "psnr": psnr_eval,
                },
            )
            for callback in callbacks
        ]

        logger.info(
            f"Epoch ended. d_loss: {generator_metric.result()}, g_loss: {discriminator_metric.result()}, psnr: {psnr_eval.numpy()}, best_psnr: {best_psnr_eval.numpy()}"
        )
        generator_metric.reset_states()
        discriminator_metric.reset_states()
    [callback.on_train_end() for callback in callbacks]


@click.command()
@click.option("--batch_size", default=16, help="Size of batch")
@click.option("--log_dir", required=True, help="Path to the log_dir for Tensorboard")
@click.option("--epochs", default=4, help="Number of epochs for training")
@click.option("--critic_updates", default=5, help="Number of discriminator training")
@click.option(
    "--restore_checkpoint",
    default=None,
    type=click.Path(exists=True),
    help="Path to pre-existing checkpoint",
)
def train_command(batch_size, log_dir, epochs, critic_updates, restore_checkpoint):
    return train(batch_size, log_dir, epochs, critic_updates, restore_checkpoint)


if __name__ == "__main__":
    train_command()
