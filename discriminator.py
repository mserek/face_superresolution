import tensorflow as tf


def _get_discriminator_block(x, filters, strides):
    x = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=3, strides=strides, padding="same"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def get_discriminator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3], name="input_image")
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(
        inputs
    )
    x = tf.keras.layers.LeakyReLU()(x)

    x = _get_discriminator_block(x, filters=64, strides=2)

    x = _get_discriminator_block(x, filters=128, strides=1)

    x = _get_discriminator_block(x, filters=128, strides=2)

    x = _get_discriminator_block(x, filters=256, strides=1)

    x = _get_discriminator_block(x, filters=256, strides=2)

    x = _get_discriminator_block(x, filters=512, strides=1)

    x = _get_discriminator_block(x, filters=512, strides=2)

    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs=inputs, outputs=out)


_binary_crossentropy = tf.keras.losses.BinaryCrossentropy()


def discriminator_loss(disc_target, disc_fakes):
    real_loss = _binary_crossentropy(tf.ones_like(disc_target), disc_target)

    generated_loss = _binary_crossentropy(tf.zeros_like(disc_fakes), disc_fakes)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss
