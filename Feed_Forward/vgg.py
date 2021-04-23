import tensorflow as tf
import os
from tensorflow.keras.applications import vgg19, VGG19
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Only show TensorFlow warnings and errors

# This is a class extened on tf.keras.models.Model
class VGG(tf.keras.models.Model):
  def __init__(self, content_layer, style_layers):
      super(VGG, self).__init__()
      # get vgg19
      vgg = VGG19(include_top=False, weights="imagenet")

      content_features = vgg.get_layer(content_layer).output
      style_features = [vgg.get_layer(name).output for name in style_layers]

      # Define this model with our selected inputs/outputs
      self.vgg = tf.keras.Model([vgg.input], [content_features, style_features])
      # Not trainable, just for features extraction
      self.vgg.trainable = False

  def call(self, inputs):
      preprocessed_input = vgg19.preprocess_input(inputs)
      content_features, style_features = self.vgg(preprocessed_input)
      return content_features, style_features
