import io
import os

import skimage.io
import tensorflow.compat.v1 as tf

Style_Path= "../style2.jpg"

style_files = tf.gfile.Glob(Style_Path)
