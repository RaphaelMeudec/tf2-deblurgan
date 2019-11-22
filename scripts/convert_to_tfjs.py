from pathlib import Path

import click
import tensorflowjs as tfjs

from deblurgan.model import generator_model


def convert_to_tfjs_model(weight_path, output_dir):
    g = generator_model()
    g.load_weights(weight_path)

    output_path = Path(output_dir) / "model"
    tfjs.converters.save_keras_model(g, str(output_path))


@click.command()
@click.option("--weight_path", required=True, type=click.Path(exists=True), help="Model weight")
@click.option("--output_dir", required=True, type=click.Path(exists=True), help="Output directory")
def convert_command(weight_path, output_dir):
    return convert_to_tfjs_model(weight_path, output_dir)


if __name__ == "__main__":
    convert_command()
