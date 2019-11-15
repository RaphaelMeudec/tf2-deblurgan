from tensorflow.keras.layers import (
    Input,
    Activation,
    Add,
    UpSampling2D,
    LeakyReLU,
    Conv2D,
    Dense,
    Flatten,
    Lambda,
    BatchNormalization,
)
from tensorflow.keras.models import Model

from deblurgan.layer_utils import ReflectionPadding2D, res_block

# TODO: Pass those elements in a config file
# the paper defined hyper-parameter:chr
channel_rate = 64
ngf = 64
ndf = 64
input_nc = 3
output_nc = 3
n_blocks_gen = 9


def generator_model(input_shape=(256, 256, 3)):
    """Build generator architecture."""
    # Current version : ResNet block
    inputs = Input(shape=input_shape)

    x = ReflectionPadding2D((3, 3))(inputs)
    x = Conv2D(filters=ngf, kernel_size=(7, 7), padding="valid")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    n_downsampling = 2
    for i in range(n_downsampling):
        mult = 2 ** i
        x = Conv2D(
            filters=ngf * mult * 2, kernel_size=(3, 3), strides=2, padding="same"
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

    mult = 2 ** n_downsampling
    for i in range(n_blocks_gen):
        x = res_block(x, ngf * mult, use_dropout=True)

    for i in range(n_downsampling):
        mult = 2 ** (n_downsampling - i)
        # x = Conv2DTranspose(filters=int(ngf * mult / 2), kernel_size=(3, 3), strides=2, padding='same')(x)
        x = UpSampling2D()(x)
        x = Conv2D(filters=int(ngf * mult / 2), kernel_size=(3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

    x = ReflectionPadding2D((3, 3))(x)
    x = Conv2D(filters=output_nc, kernel_size=(7, 7), padding="valid")(x)
    x = Activation("tanh")(x)

    outputs = Add()([x, inputs])
    outputs = Lambda(lambda z: z / 2)(outputs)

    model = Model(inputs=inputs, outputs=outputs, name="Generator")
    return model


def discriminator_model(input_shape=(256, 256, 3)):
    """Build discriminator architecture."""
    n_layers, use_sigmoid = 3, False
    inputs = Input(shape=input_shape)

    x = Conv2D(filters=ndf, kernel_size=(4, 4), strides=2, padding="same")(inputs)
    x = LeakyReLU(0.2)(x)

    nf_mult, nf_mult_prev = 1, 1
    for n in range(n_layers):
        nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
        x = Conv2D(
            filters=ndf * nf_mult, kernel_size=(4, 4), strides=2, padding="same"
        )(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

    nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
    x = Conv2D(filters=ndf * nf_mult, kernel_size=(4, 4), strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=1, kernel_size=(4, 4), strides=1, padding="same")(x)
    if use_sigmoid:
        x = Activation("sigmoid")(x)

    x = Flatten()(x)
    x = Dense(1024, activation="tanh")(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=x, name="Discriminator")
    return model


def generator_containing_discriminator(generator, discriminator, image_shape=(256, 256, 3)):
    inputs = Input(shape=image_shape)
    generated_image = generator(inputs)
    outputs = discriminator(generated_image)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def generator_containing_discriminator_multiple_outputs(generator, discriminator, image_shape=(256, 256, 3)):
    inputs = Input(shape=image_shape)
    generated_image = generator(inputs)
    outputs = discriminator(generated_image)
    model = Model(inputs=inputs, outputs=[generated_image, outputs])
    return model


if __name__ == "__main__":
    g = generator_model()
    g.summary()
    d = discriminator_model()
    d.summary()
    m = generator_containing_discriminator(generator_model(), discriminator_model())
    m.summary()
