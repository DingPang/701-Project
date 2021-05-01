import tensorflow as tf
import os

image_size = 256

def load_img(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = resize(img, min_size = image_size)
    img = tf.cast(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

# def resize(img, min_size):
#     width, height, _ = tf.unstack(tf.shape(img), num=3)
#     if height < width:
#         new_height = min_size
#         new_width = int (width * new_height / height)
#     else:
#         new_width = min_size
#         new_height = int (height * new_width / width)
#     img = tf.image.resize(img, size = (new_width, new_height))
#     return img

def resize(img, min_size):
    img = tf.image.resize(img, size = (min_size, min_size))
    return img

def content_pre_proc(file_path, min_size=image_size):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels = 3)
    img = resize(img, min_size = min_size)
    img = tf.image.random_crop(
        img,
        size = (image_size, image_size, 3)
    )
    img = tf.cast(img, tf.float32)
    return img
