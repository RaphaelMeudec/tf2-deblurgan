from pathlib import Path

import click
import tensorflow as tf

from deblurgan.model import generator_model


def deblur(weight_path, image_path, output_dir):
    g = generator_model()
    g.load_weights(weight_path)

    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image)
    image = tf.image.convert_image_dtype(image, tf.float32)

    deblurred = g(tf.convert_to_tensor([image]))[0]

    output_path = Path(output_dir) / f"{Path(image_path).stem}.png"
    output = tf.image.convert_image_dtype(deblurred, tf.uint8)
    output = tf.image.encode_png(output)
    tf.io.write_file(str(output_path), output)


@click.command()
@click.option("--weight_path", help="Model weight")
@click.option("--image_path", help="Image to deblur")
@click.option("--output_dir", help="Deblurred image")
def deblur_command(weight_path, image_path, output_dir):
    return deblur(weight_path, image_path, output_dir)


if __name__ == "__main__":
    deblur_command()
