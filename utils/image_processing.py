from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .vgg_preprocessing import preprocess_image


def _process_image(encoded_image,
                   height,
                   width,
                   is_training,
                   image_format="jpeg"):
    """Decode an image and resize.

    Args:
        encoded_image: String Tensor containing the image.
        is_training: Whether preprocessing for training.
        height: Height of the output image.
        width: Width of the output image.
        image_format: "jpeg" or "png".

    Returns:
        A float32 Tensor of shape [heigh, width, 3] with values in [-1, 1].
    """
    with tf.name_scope("decode", values=[encoded_image]):
        if image_format == "jpeg":
            image = tf.image.decode_jpeg(encoded_image, channels=3)
        elif image_format == "png":
            image = tf.image.decode_png(encoded_image, channels=3)
        else:
            raise ValueError("Invalid image format: %s" % image_format)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = preprocess_image(image, height, width, is_training)
    return image


def process_images(encoded_images,
                   height,
                   width,
                   is_training,
                   image_format="jpeg"):
    """Applies _process_image() to a collection of images."""
    f = lambda x: _process_image(x, height, width, is_training, image_format)
    return tf.map_fn(f, encoded_images, dtype=tf.float32)

