from pathlib import Path

import click
import tensorflow as tf

from deblurgan.model import discriminator_model, generator_model


def deblur(weight_path, image_path, output_dir):
    g = generator_model()
    d = discriminator_model()

    ckpt = tf.train.Checkpoint(
        generator=g,
        discriminator=d,
        generator_optimizer=tf.keras.optimizers.Adam(
            lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08
        ),
        discriminator_optimizer=tf.keras.optimizers.Adam(
            lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08
        ),
    )
    ckpt.restore(weight_path)

    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image)
    image = tf.image.convert_image_dtype(image, tf.float32)

    deblurred = g(tf.convert_to_tensor([image]))[0]

    output_path = Path(output_dir) / f"{Path(image_path).stem}.png"
    output = tf.image.convert_image_dtype(deblurred, tf.uint8)
    output = tf.image.encode_png(output)
    tf.io.write_file(output_path, output)


@click.command()
@click.option("--weight_path", help="Model weight")
@click.option("--image_path", help="Image to deblur")
@click.option("--output_dir", help="Deblurred image")
def deblur_command(weight_path, image_path, output_dir):
    return deblur(weight_path, image_path, output_dir)


if __name__ == "__main__":
    deblur_command()
