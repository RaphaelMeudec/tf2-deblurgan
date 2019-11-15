import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model


def l1_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))


def perceptual_loss(
    y_true, y_pred, sample_weight=None, input_shape=(256, 256, 3), loss_factor=1
):
    vgg = VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output)
    loss_model.trainable = False
    return loss_factor * K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(
        gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape))
    )

    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = K.square(1 - gradient_l2_norm)

    return K.mean(gradient_penalty)
