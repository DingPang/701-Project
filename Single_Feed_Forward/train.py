import os
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_datasets as tfds

from vgg import VGG
from transformer import TransferNet

#Paths
content_paths = ["../content1.jpg"]
style_paths = ["../style2.jpeg"]
#Layers
content_layer = "block4_conv1"
style_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
]

# test_content_images = tf.concat(
#     [load_img(f"images/content/{f}") for f in content_paths], axis=0
# )
# test_style_images = tf.concat(
#     [load_img(f"images/style/{f}") for f in style_paths], axis=0
# )

vgg = VGG(content_layer, style_layers)
transformer = TransferNet(content_layer)
