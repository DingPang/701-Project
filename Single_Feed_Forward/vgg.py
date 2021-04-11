import tensorflow as tf
from tensorflow.keras.applications import vgg19, VGG19

class VGG(tf.keras.models.Model):
  def __init__(self, content_layer, style_layers):
      super(VGG, self).__init__()
      vgg = VGG19(include_top=False, weights="imagenet")

      content_output = vgg.get_layer(content_layer).output
      style_outputs = [vgg.get_layer(name).output for name in style_layers]

      self.vgg = tf.keras.Model(
          [vgg.input], [content_output, style_outputs]
      )
      # Not trainable
      self.vgg.trainable = False

  def call(self, inputs):
      preprocessed_input = vgg19.preprocess_input(inputs)
      content_outputs, style_outputs = self.vgg(preprocessed_input)
      return content_outputs, style_outputs
