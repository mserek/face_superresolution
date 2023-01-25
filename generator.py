import tensorflow as tf


def _get_generator_block(x):
    skip = x
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, skip])
    return x


def get_generator(num_of_blocks=12):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=9, strides=1, padding="same")(
        inputs
    )
    x = tf.keras.layers.PReLU()(x)
    skip = x
    for i in range(num_of_blocks):
        x = _get_generator_block(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, skip])

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(x)
    x = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)
    x = tf.keras.layers.PReLU()(x)

    out = tf.keras.layers.Conv2D(filters=3, kernel_size=9, strides=1, padding="same")(x)
    return tf.keras.Model(inputs=inputs, outputs=out)


_binary_crossentropy = tf.keras.losses.BinaryCrossentropy()


def generator_loss(disc_fakes, fakes, targets):
    ssim = tf.math.reduce_mean(tf.image.ssim(fakes + 5, targets + 5, max_val=2))
    gan_loss = _binary_crossentropy(tf.ones_like(disc_fakes), disc_fakes)
    return (1 - ssim) + 0.01 * gan_loss
